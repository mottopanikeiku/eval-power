"""Generate new, identified answers for the committed pilot or confirmation plan.

Run with Modal's CLI: modal run modal_app.py --phase pilot --models qwen15
Model weights and datasets are downloaded inside the GPU container, never locally.
"""

import gzip
import hashlib
import json
import os
import random
import time
from pathlib import Path

import modal

ROOT = Path(__file__).parent
app = modal.App("eval-power-prospective")
timeout_seconds = int(os.environ.get("EVAL_TIMEOUT_S", "1800"))
container_limit = int(os.environ.get("EVAL_CONTAINERS", "2"))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.2", "datasets==4.1.1", "hf-transfer==0.1.9")
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
    volumes={"/cache": cache},
)
def generate(model, protocol, item_indices, phase):
    import os

    os.environ["HF_HOME"] = "/cache/huggingface"
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    from eval_power.grading import grade_answer, make_prompt

    start = time.monotonic()
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
            item_seed = (
                protocol["generation_seed"]
                + model_index * 1000000
                + (100000 if phase == "confirm" else 0)
                + (10000 if bench == "arc" else 0)
                + idx
            )
            prompts.extend([prompt, prompt])
            parameters.extend(
                [
                    SamplingParams(temperature=0, max_tokens=source["max_tokens"]),
                    SamplingParams(
                        n=protocol["k"],
                        temperature=protocol["temperature"],
                        top_p=protocol["top_p"],
                        max_tokens=source["max_tokens"],
                        seed=item_seed,
                    ),
                ]
            )
        if not prompts:
            results[bench] = []
            continue
        outputs = llm.generate(prompts, parameters, use_tqdm=False)
        rows = []
        for j, idx in enumerate(indices):
            def record(output):
                prediction, correct = grade_answer(bench, output.text, references[j])
                return {
                    "text": output.text,
                    "prediction": prediction,
                    "correct": correct,
                    "finish_reason": output.finish_reason,
                    "output_tokens": len(output.token_ids),
                }

            rows.append(
                {
                    "item_id": str(idx),
                    "prompt_sha256": hashes[j],
                    "reference": references[j],
                    "greedy": record(outputs[2 * j].outputs[0]),
                    "samples": [record(o) for o in outputs[2 * j + 1].outputs],
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
            "gpu": "L4",
            "load_seconds": load_seconds,
            "elapsed_seconds": elapsed,
            "output_tokens": output_tokens,
            "output_tokens_per_second_including_load": output_tokens / elapsed,
            "vllm": "0.10.2",
            "k": protocol["k"],
            "temperature": protocol["temperature"],
            "max_model_len": 2048,
            "dtype": "half",
        },
    }


def item_order(protocol, benchmark):
    indices = list(range(protocol["benchmarks"][benchmark]["size"]))
    random.Random(protocol["item_seed"] + (1 if benchmark == "arc" else 0)).shuffle(indices)
    return indices


@app.local_entrypoint()
def main(phase: str = "pilot", models: str = "qwen15", throughput: bool = False):
    protocol = json.loads((ROOT / "results/prospective/protocol.json").read_text())
    selected = [m for m in protocol["models"] if m["slug"] in models.split(",")]
    if not selected or len(selected) != len(models.split(",")):
        raise ValueError("Unknown or duplicate model slug")
    if phase not in ("pilot", "confirm"):
        raise ValueError("phase must be pilot or confirm")
    if throughput and phase != "pilot":
        raise ValueError("throughput pilot cannot confirm comparisons")
    plan = None
    if phase == "confirm":
        plan = json.loads((ROOT / "results/prospective/plan.json").read_text())
    out = ROOT / "results/prospective" / ("throughput" if throughput else phase)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for model in selected:
        destination = out / f"{model['slug']}.json.gz"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite completed answers: {destination}")
        if phase == "pilot":
            n = 8 if throughput else protocol["pilot_items"]
            indices = {b: item_order(protocol, b)[:n] for b in protocol["benchmarks"]}
        else:
            indices = {}
            for bench in protocol["benchmarks"]:
                ids = set()
                for pair in plan["pairs"]:
                    if (
                        pair["benchmark"] == bench
                        and pair["status"] == "planned"
                        and model["slug"] in (pair["model_a"], pair["model_b"])
                    ):
                        ids.update(int(i) for i in pair["item_ids"])
                indices[bench] = sorted(ids)
        jobs.append((destination, generate.spawn(model, protocol, indices, phase)))
    for destination, job in jobs:
        result = job.get()
        result["metadata"]["protocol_sha256"] = hashlib.sha256(
            (ROOT / "results/prospective/protocol.json").read_bytes()
        ).hexdigest()
        if plan is not None:
            result["metadata"]["plan_sha256"] = hashlib.sha256(
                (ROOT / "results/prospective/plan.json").read_bytes()
            ).hexdigest()
        destination.write_bytes(gzip.compress(json.dumps(result).encode(), mtime=0))
        print(f"Saved {destination.name}: {result['metadata']}")
