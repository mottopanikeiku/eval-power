#!/usr/bin/env python3
"""Import five binary historical leaderboard matrices, without responses or weights.

The only downloaded artifact is a checksum-pinned tinyBenchmarks NumPy pickle.
Its global model order is preserved. Item IDs identify source-array positions,
not original documents; MMLU alignment remains a shared-order assumption.
"""

import argparse
import hashlib
import json
import os
import pickle
import tempfile
import urllib.request
from pathlib import Path
from typing import BinaryIO

import numpy as np

SOURCE_REVISION = "e9a8b1031b0340571beb6c9ca3a27891be09a8fd"
SOURCE_URL = (
    "https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/"
    f"{SOURCE_REVISION}/tutorials/data/lb.pickle"
)
SOURCE_SHA256 = "34f44d6a819512ef74d00a95288d252fa679288a10ca167cd97fdbc3aae66437"
SOURCE_GIT_BLOB = "c6f010d63c08779624ad6b95b518e7d58f369283"
SOURCE_BYTES = 90_593_109
MODEL_COUNT = 395
EXPECTED_ITEMS = {"gsm8k": 1319, "winogrande": 1267, "arc": 1172, "hellaswag": 10042, "mmlu": 14042}
MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)
TASKS = {
    "gsm8k": ("harness_gsm8k_5",),
    "winogrande": ("harness_winogrande_5",),
    "arc": ("harness_arc_challenge_25",),
    "hellaswag": ("harness_hellaswag_10",),
    "mmlu": tuple(f"harness_hendrycksTest_{subject}_5" for subject in MMLU_SUBJECTS),
}
METRICS = {
    "gsm8k": "acc",
    "winogrande": "acc",
    "arc": "acc_norm",
    "hellaswag": "acc_norm",
    "mmlu": "acc",
}
RESEARCH_REVISION = "c9df1547a460d5f1d2ad1ad88b6a117979858fbc"
RESEARCH_BASE = (
    "https://raw.githubusercontent.com/felipemaiapolo/efficbench/"
    f"{RESEARCH_REVISION}/generating_data/download-openllmleaderboard/"
)


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Allow only the constructors used by NumPy's array pickle protocols."""

    def find_class(self, module: str, name: str):
        # Old pickles refer to numpy.core; NumPy 2 uses numpy._core internally.
        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("numpy.core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
            ("numpy._core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
        }
        try:
            return allowed[module, name]
        except KeyError as error:
            raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}") from error


def verify_source(
    stream: BinaryIO,
    *,
    size: int = SOURCE_BYTES,
    sha256: str = SOURCE_SHA256,
    git_blob: str = SOURCE_GIT_BLOB,
) -> None:
    """Check both byte and Git-object identities, then rewind the same file."""
    stream.seek(0)
    raw_hash = hashlib.sha256()
    blob_hash = hashlib.sha1(f"blob {size}\0".encode())
    actual_size = 0
    while chunk := stream.read(1024 * 1024):
        actual_size += len(chunk)
        raw_hash.update(chunk)
        blob_hash.update(chunk)
    if actual_size != size:
        raise ValueError(f"Source size mismatch: expected {size}, got {actual_size}")
    if raw_hash.hexdigest() != sha256:
        raise ValueError("Source SHA256 mismatch")
    if blob_hash.hexdigest() != git_blob:
        raise ValueError("Source Git blob hash mismatch")
    stream.seek(0)


def default_cache() -> Path:
    directory = Path(os.environ.get("EVAL_POWER_CACHE", Path.home() / ".cache" / "eval-power"))
    return directory / "lb-e9a8b103.pickle"


def ensure_cached(cache: Path) -> None:
    """Reuse a cache file; download only this single known artifact if absent."""
    if cache.exists():
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache.parent, delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > SOURCE_BYTES:
                        raise ValueError("Download exceeds pinned source size")
                    output.write(chunk)
        with temporary.open("rb") as stream:
            verify_source(stream)
        temporary.replace(cache)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_source(cache: Path) -> dict:
    """Never deserialize a file until both pinned checksums match."""
    with cache.open("rb") as stream:
        verify_source(stream)
        data = NumpyOnlyUnpickler(stream).load()
    if not isinstance(data, dict):
        raise ValueError("Source root must be a dictionary")
    return data


def build_matrices(data: dict) -> dict[str, dict[str, np.ndarray]]:
    """Validate the source's item-by-model orientation and preserve its order."""
    names = data.get("models")
    if not isinstance(names, (list, tuple, np.ndarray)) or len(names) != MODEL_COUNT:
        raise ValueError(f"Expected {MODEL_COUNT} model labels")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Model labels must be nonempty strings")
    if len(set(names)) != MODEL_COUNT:
        raise ValueError("Model labels must be unique")
    models = np.asarray(names, dtype=str)
    blocks = data.get("data")
    if not isinstance(blocks, dict):
        raise ValueError("Source data must be a dictionary of task blocks")
    matrices = {}
    for benchmark, tasks in TASKS.items():
        correctness, item_ids, subjects = [], [], []
        for index, task in enumerate(tasks):
            block = blocks.get(task)
            if not isinstance(block, dict):
                raise ValueError(f"Missing task block: {task}")
            # The real export has only global labels. If block labels exist,
            # do not permit an unnoticed column permutation.
            if "models" in block and not np.array_equal(block["models"], models):
                raise ValueError(f"Model alignment mismatch: {task}")
            values = block.get("correctness")
            if not isinstance(values, np.ndarray) or values.ndim != 2:
                raise ValueError(f"Correctness must be a two-dimensional array: {task}")
            if values.shape[1] != MODEL_COUNT or values.shape[0] == 0:
                raise ValueError(f"Expected nonempty item x {MODEL_COUNT} matrix: {task}")
            if values.dtype.kind not in "buif":
                raise ValueError(f"Correctness must be real numeric values: {task}")
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite correctness: {task}")
            if not ((values == 0) | (values == 1)).all():
                raise ValueError(f"Nonbinary correctness: {task}")
            ids = [f"{task}:{offset}" for offset in range(values.shape[0])]
            correctness.append(values.astype(np.uint8))
            item_ids.extend(ids)
            subjects.extend(
                [MMLU_SUBJECTS[index]] * values.shape[0] if benchmark == "mmlu" else ids
            )
        matrix = np.concatenate(correctness, axis=0)
        if matrix.shape[0] != EXPECTED_ITEMS[benchmark]:
            raise ValueError(f"Unexpected {benchmark} item count: {matrix.shape[0]}")
        matrices[benchmark] = {
            "correctness": matrix,
            "models": models,
            "item_ids": np.asarray(item_ids, dtype=str),
            "subjects": np.asarray(subjects, dtype=str),
        }
    return matrices


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def license_metadata() -> dict:
    """License-reference revisions are not historical evaluation revisions."""
    return {
        "reference_revision_scope": (
            "The revisions below pin opened LICENSE METADATA only, not the dataset revisions "
            "used by the historical source evaluations; those are absent from the artifact."
        ),
        "tinyBenchmarks": {
            "license": "MIT",
            "copyright": "Felipe Maia Polo 2024",
            "revision": SOURCE_REVISION,
            "url": f"https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/"
            f"{SOURCE_REVISION}/LICENSE",
            "local_notice": "LICENSE-tinyBenchmarks.txt",
            "local_notice_sha256": (
                "03b276e45e47b9e0a02e40935eb0d2c9efe3b4259833f960edb2bfb64cdefd4a"
            ),
            "scope": "Repository-level license; no separate matrix-specific grant found.",
        },
        "gsm8k": {
            "license": "MIT",
            "copyright": "OpenAI 2021",
            "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
            "url": "https://raw.githubusercontent.com/openai/grade-school-math/"
            "3101c7d5072418e28b9008a6636bde82a006892c/LICENSE",
            "evidence": "Original license text opened.",
        },
        "mmlu": {
            "license": "MIT",
            "copyright": "Dan Hendrycks 2020",
            "revision": "4450500f923c49f1fb1dd3d99108a0bd9717b660",
            "url": "https://raw.githubusercontent.com/hendrycks/test/"
            "4450500f923c49f1fb1dd3d99108a0bd9717b660/LICENSE",
            "evidence": "Original license text opened; CAIS author metadata agrees.",
        },
        "winogrande": {
            "license": "CC-BY",
            "license_version": None,
            "revision": "727e837f77521ef38bcc56df3b275c8da43f45af",
            "url": "https://raw.githubusercontent.com/allenai/winogrande/"
            "727e837f77521ef38bcc56df3b275c8da43f45af/README.md",
            "evidence": "Official README declares dataset CC-BY, version unspecified; "
            "the separate code license is Apache-2.0, not the dataset license.",
        },
        "arc": {
            "license": "CC-BY-SA-4.0",
            "revision": "210d026faf9955653af8916fad021475a3f00453",
            "url": "https://huggingface.co/datasets/allenai/ai2_arc/raw/"
            "210d026faf9955653af8916fad021475a3f00453/README.md",
            "evidence": "Official AllenAI dataset card opened.",
        },
        "hellaswag": {
            "license": "MIT (author dataset card declaration)",
            "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
            "url": "https://huggingface.co/datasets/Rowan/hellaswag/raw/"
            "218ec52e09a7e7462a5400043bb9a69a41d06b76/README.md",
            "homepage": "https://rowanzellers.com/hellaswag/",
            "evidence": "Author card opened; original repository/license was inaccessible "
            "(HTTP451/404). ActivityNet/WikiHow constituent rights were not audited.",
        },
        "derived_matrix_terms": (
            "Numerical correctness and labels only; no prompts/question text. Distinct "
            "benchmark terms are retained, not relicensed collectively as MIT. Applicability "
            "of upstream share-alike terms to these derived numerical arrays is not established."
        ),
    }


def write_outputs(matrices: dict[str, dict[str, np.ndarray]], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for benchmark, arrays in matrices.items():
        path = output_dir / f"{benchmark}.npz"
        np.savez_compressed(path, **arrays)
        subjects, counts = np.unique(arrays["subjects"], return_counts=True)
        outputs[benchmark] = {
            "file": path.name,
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "shape": list(arrays["correctness"].shape),
            "metric": METRICS[benchmark],
            "source_tasks": list(TASKS[benchmark]),
            "subject_count": len(subjects),
            "subject_sizes": dict(zip(subjects.tolist(), counts.tolist(), strict=True))
            if benchmark == "mmlu"
            else None,
        }
    model_ids = next(iter(matrices.values()))["models"].tolist()
    manifest = {
        "schema_version": 1,
        "source": {
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "git_blob_sha1": SOURCE_GIT_BLOB,
            "sha256": SOURCE_SHA256,
            "bytes": SOURCE_BYTES,
            "artifact": "tutorials/data/lb.pickle",
            "kind": "Original per-item Open LLM Leaderboard metrics, not IRT predictions.",
            "historical_population": "Selected January-2024-era Open LLM Leaderboard models.",
            "selection": "Author snapshot: MMLU > 30, then every fifth listed model; "
            "drop incomplete evaluations and mindy-labs/mindy-7b (limited responses).",
            "ordering": "Model axis follows source models list, descending GSM8K evaluation date.",
            "model_count": len(model_ids),
            "model_ids": model_ids,
            "original_hf_per_model_revisions": None,
            "original_benchmark_revisions": None,
            "processing_sources": {
                "download": RESEARCH_BASE + "download_leaderboard.ipynb",
                "processing": RESEARCH_BASE + "process_lb_data.ipynb",
                "ordering_check": (
                    RESEARCH_BASE + "download_leaderboard_sanity_check_ordering.ipynb"
                ),
                "task_mapping": "https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/"
                f"{SOURCE_REVISION}/tutorials/utils.py",
            },
        },
        "array_contract": {
            "correctness": "uint8; item x model; finite exact 0/1 only",
            "models": "Unicode full source HF details-dataset identifiers, source order",
            "item_ids": "Unicode task:source-offset; not original document IDs",
            "subjects": "Unicode natural MMLU subject; unique per-item label for other benchmarks",
            "benchmark_order": list(TASKS),
            "mmlu_subject_order": list(MMLU_SUBJECTS),
            "weighting": "Rows retain source counts; aggregate accuracy is micro-item weighted.",
        },
        "alignment": {
            "non_mmlu": (
                "Historical source notebook compares full_prompt order across 406 rows "
                "and reports equality 1.0 for checked non-MMLU scenarios; not re-executed here."
            ),
            "mmlu": "Omitted from the source alignment check. Shared item order is inherited "
            "from the exporter; no independent document-ID audit is possible here.",
            "import_checks": "All blocks have the same 395 columns and unique global labels; "
            "optional per-block labels must match. This is not an item-ID audit.",
        },
        "exclusions": {
            "harness_truthfulqa_mc_0": "Nonbinary mc2 metric; excluded without thresholding."
        },
        "licenses": license_metadata(),
        "outputs": outputs,
        "limitations": [
            "Many selected historical models are related fine-tunes/merges, "
            "not independent families.",
            "The source omits original document IDs, prompt hashes and "
            "original HF per-model revisions.",
            "No current-frontier, repeated-generation, or future-subject power claim "
            "follows from these arrays.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=default_cache(),
        help="Path to the cached monolithic lb.pickle; downloaded only if absent",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    ensure_cached(args.cache)
    matrices = build_matrices(load_source(args.cache))
    manifest = write_outputs(matrices, args.output_dir)
    for benchmark, output in manifest["outputs"].items():
        print(f"{benchmark}: {output['shape']} {output['sha256']}")


if __name__ == "__main__":
    main()
