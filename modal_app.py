"""Generate new, identified answers for the committed pilot or confirmation plan.

Run with Modal's CLI: modal run modal_app.py::main --phase pilot --models qwen15
Model weights and datasets are downloaded inside the GPU container, never locally.
"""

import gzip
import hashlib
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

ROOT = Path(__file__).parent
app = modal.App("eval-power-prospective")
timeout_seconds = int(os.environ.get("EVAL_TIMEOUT_S", "1800"))
container_limit = int(os.environ.get("EVAL_CONTAINERS", "2"))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.10.2",
        "transformers==4.55.2",
        "huggingface-hub==0.34.4",
        "datasets==4.1.1",
        "hf-transfer==0.1.9",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "OMP_NUM_THREADS": "4"})
    .add_local_python_source("eval_power")
)
cache = modal.Volume.from_name("eval-power-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=8192,
    timeout=timeout_seconds,
    max_containers=container_limit,
    single_use_containers=True,
    volumes={"/cache": cache},
)
def generate(model, protocol, item_indices, phase):
    import os

    os.environ["HF_HOME"] = "/cache/huggingface"
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    from eval_power.grading import (
        generation_seed,
        grade_answer,
        grade_strict_answer,
        make_prompt,
    )

    start = time.monotonic()
    started_utc = datetime.now(UTC).isoformat()
    llm = LLM(
        model=model["id"],
        revision=model["revision"],
        dtype="half",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        max_num_seqs=96,
        enforce_eager=True,
        seed=protocol["generation_seed"],
        trust_remote_code=False,
    )
    tokenizer = llm.get_tokenizer()
    stop_audit = SamplingParams()
    stop_audit.update_from_generation_config(
        llm.llm_engine.processor.generation_config_fields, tokenizer.eos_token_id
    )
    native_stop_ids = set(stop_audit.all_stop_token_ids)
    marker = "EVAL_POWER_ASSISTANT_CONTENT"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "_"}, {"role": "assistant", "content": marker}],
        tokenize=False,
        add_generation_prompt=False,
    )
    suffix = rendered.split(marker, 1)[1]
    chat_end_ids = {
        tokenizer.convert_tokens_to_ids(token)
        for token in tokenizer.all_special_tokens
        if token in suffix
    }
    load_seconds = time.monotonic() - start
    results = {}
    model_index = next(i for i, x in enumerate(protocol["models"]) if x["slug"] == model["slug"])
    for bench, indices in item_indices.items():
        source = protocol["benchmarks"][bench]
        dataset = load_dataset(
            source["id"], source["config"], revision=source["revision"], split=source["split"]
        )
        prompts, parameters, references, hashes = [], [], [], []
        for idx in indices:
            row = dataset[idx]
            user, reference = make_prompt(bench, row)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
            references.append(reference)
            hashes.append(hashlib.sha256(user.encode()).hexdigest())
            guided = (
                GuidedDecodingParams(
                    choice=[f"Answer: {label}" for label in row["choices"]["label"]],
                )
                if source.get("decoding") == "guided_direct_choice"
                else None
            )
            item_seed = generation_seed(protocol, model_index, bench, idx, phase)
            prompts.extend([prompt, prompt])
            parameters.extend(
                [
                    SamplingParams(
                        temperature=0, max_tokens=source["max_tokens"], guided_decoding=guided
                    ),
                    SamplingParams(
                        n=protocol["k"],
                        temperature=protocol["temperature"],
                        top_p=protocol["top_p"],
                        max_tokens=source["max_tokens"],
                        seed=item_seed,
                        guided_decoding=guided,
                    ),
                ]
            )
        if not prompts:
            results[bench] = []
            continue
        outputs = llm.generate(prompts, parameters, use_tqdm=False)
        rows = []
        for j, idx in enumerate(indices):

            def record(output, *, seed=None, benchmark=bench, reference=references[j]):
                prediction, correct = grade_answer(benchmark, output.text, reference)
                strict_prediction, strict_correct = grade_strict_answer(
                    benchmark, output.text, reference
                )
                return {
                    "text": output.text,
                    "prediction": prediction,
                    "correct": correct,
                    "strict_prediction": strict_prediction,
                    "strict_correct": strict_correct,
                    "finish_reason": output.finish_reason,
                    "output_tokens": len(output.token_ids),
                    "stop_reason": output.stop_reason,
                    "sample_index": output.index,
                    "seed": seed,
                }

            rows.append(
                {
                    "item_id": str(idx),
                    "prompt_sha256": hashes[j],
                    "reference": references[j],
                    "greedy": record(outputs[2 * j].outputs[0]),
                    "samples": [
                        record(o, seed=parameters[2 * j + 1].seed + o.index)
                        for o in outputs[2 * j + 1].outputs
                    ],
                }
            )
        results[bench] = rows
    elapsed = time.monotonic() - start
    output_tokens = sum(
        row["greedy"]["output_tokens"] + sum(x["output_tokens"] for x in row["samples"])
        for rows in results.values()
        for row in rows
    )
    return {
        "model": model["id"],
        "slug": model["slug"],
        "revision": model["revision"],
        "benchmarks": results,
        "metadata": {
            "phase": phase,
            "started_utc": started_utc,
            "gpu": "L4",
            "load_seconds": load_seconds,
            "elapsed_seconds": elapsed,
            "output_tokens": output_tokens,
            "output_tokens_per_second_including_load": output_tokens / elapsed,
            "vllm": "0.10.2",
            "k": protocol["k"],
            "temperature": protocol["temperature"],
            "seed_stride": protocol["seed_stride"],
            "max_model_len": 2048,
            "dtype": "half",
            "arc_decoding": protocol["benchmarks"]["arc"].get("decoding", "unconstrained"),
            "stop_token_audit": {
                "native_stop_ids": sorted(native_stop_ids),
                "chat_template_end_ids": sorted(chat_end_ids),
                "native_plus_chat_end_ids": sorted(native_stop_ids | chat_end_ids),
                "additional_chat_end_ids_needed": sorted(chat_end_ids - native_stop_ids),
                "note": "Audit only: I did not override native EOS or chat-end token handling.",
            },
        },
    }


def item_order(protocol, benchmark):
    indices = list(range(protocol["benchmarks"][benchmark]["size"]))
    random.Random(protocol["item_seed"] + (1 if benchmark == "arc" else 0)).shuffle(indices)
    return indices


@app.local_entrypoint()
def main(
    phase: str = "pilot",
    models: str = "qwen15",
    throughput: bool = False,
    output_dir: str = "results/prospective/primary",
    benchmarks: str = "gsm8k,arc",
):
    protocol_bytes = (ROOT / "results/prospective/protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    selected = [m for m in protocol["models"] if m["slug"] in models.split(",")]
    if not selected or len(selected) != len(models.split(",")):
        raise ValueError("Unknown or duplicate model slug")
    selected_benchmarks = benchmarks.split(",")
    if any(benchmark not in protocol["benchmarks"] for benchmark in selected_benchmarks):
        raise ValueError("Unknown benchmark")
    if phase == "confirm" and set(selected_benchmarks) != set(protocol["benchmarks"]):
        raise ValueError("Confirmation must collect every planned benchmark")
    if phase not in ("pilot", "confirm"):
        raise ValueError("phase must be pilot or confirm")
    if throughput and phase != "pilot":
        raise ValueError("throughput pilot cannot confirm comparisons")
    plan = None
    plan_bytes = None
    directory = ROOT / output_dir
    if phase == "confirm":
        plan_bytes = (directory / "plan.json").read_bytes()
        plan = json.loads(plan_bytes)
    out = directory / ("throughput" if throughput else phase)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for model in selected:
        destination = out / f"{model['slug']}.json.gz"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite completed answers: {destination}")
        if phase == "pilot":
            n = 8 if throughput else protocol["pilot_items"]
            indices = {b: item_order(protocol, b)[:n] for b in selected_benchmarks}
        else:
            indices = {}
            for bench in protocol["benchmarks"]:
                ids = set()
                for pair in plan["pairs"]:
                    if (
                        pair["benchmark"] == bench
                        and pair["status"] == "feasible"
                        and model["slug"] in (pair["model_a"], pair["model_b"])
                    ):
                        ids.update(int(i) for i in pair["item_ids"])
                indices[bench] = sorted(ids)
        jobs.append((destination, generate.spawn(model, protocol, indices, phase)))
    failures = []
    for destination, job in jobs:
        try:
            result = job.get()
        except Exception as error:
            failures.append(f"{destination.name}: {error}")
            continue
        result["metadata"]["protocol_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
        if plan is not None:
            result["metadata"]["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
        destination.write_bytes(gzip.compress(json.dumps(result).encode(), mtime=0))
        print(f"Saved {destination.name}: {result['metadata']}")
    if failures:
        raise RuntimeError("Some models did not finish:\n" + "\n".join(failures))


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    cpu=1,
    memory=1024,
    timeout=120,
    max_containers=1,
    volumes={"/cache": cache},
)
def remove_weight_cache(model_ids):
    import shutil

    removed = []
    total_bytes = 0
    for model_id in model_ids:
        directory = Path("/cache/huggingface/hub") / ("models--" + model_id.replace("/", "--"))
        if not directory.exists():
            continue
        total_bytes += sum(
            path.stat().st_size for path in (directory / "blobs").glob("*") if path.is_file()
        )
        shutil.rmtree(directory)
        removed.append(model_id)
    cache.commit()
    return {"removed_model_caches": removed, "removed_blob_bytes": total_bytes}


@app.local_entrypoint()
def clean_cache():
    """Remove only this experiment's cached model weights after all runs finish."""
    protocol = json.loads((ROOT / "results/prospective/protocol.json").read_text())
    result = remove_weight_cache.remote([model["id"] for model in protocol["models"]])
    (ROOT / "results/prospective/cache_cleanup.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result)
