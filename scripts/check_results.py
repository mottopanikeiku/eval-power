"""CI checks for matrix provenance, complete hypothesis families, and result consistency."""

import csv
import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from eval_power.analysis import BENCHMARKS
from eval_power.grading import grade_answer
from eval_power.prospective import analyze_plan, validate_sources
from eval_power.stats import holm_adjust


def rows(name: str) -> list[dict]:
    path = Path("results") / name
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_prospective() -> None:
    directory = Path("results/prospective")
    protocol_path, plan_path = directory / "protocol.json", directory / "plan.json"
    protocol = json.loads(protocol_path.read_text())
    plan = json.loads(plan_path.read_text())
    summary = json.loads((directory / "summary.json").read_text())
    stages = {}
    item_references = {}
    for stage in ("pilot", "confirm"):
        documents = {}
        for path in sorted((directory / stage).glob("*.json.gz")):
            with gzip.open(path, "rt") as stream:
                document = json.load(stream)
            slug = path.name.removesuffix(".json.gz")
            assert document["slug"] == slug
            documents[slug] = document
            for benchmark, items in document["benchmarks"].items():
                for item in items:
                    identity = (benchmark, item["item_id"])
                    reference = (item["prompt_sha256"], item["reference"])
                    assert item_references.setdefault(identity, reference) == reference
                    for answer in (item["greedy"], *item["samples"]):
                        prediction, correct = grade_answer(
                            benchmark, answer["text"], item["reference"]
                        )
                        assert answer["prediction"] == prediction
                        assert answer["correct"] == correct
        stages[stage] = documents
    validate_sources(
        plan,
        protocol,
        stages["pilot"],
        stages["confirm"],
        hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    recalculated = analyze_plan(plan, stages["pilot"], stages["confirm"])
    for key, value in recalculated.items():
        assert summary[key] == value, f"Prospective summary differs: {key}"
    for group in ("protocol", "plan"):
        source = summary["sources"][group]
        assert (
            source["sha256"]
            == hashlib.sha256((directory / source["file"]).read_bytes()).hexdigest()
        )
    for stage in ("pilot", "confirm"):
        for source in summary["sources"][stage]:
            assert (
                source["sha256"]
                == hashlib.sha256((directory / source["file"]).read_bytes()).hexdigest()
            )
    assert ET.parse("figures/prospective.svg").getroot().tag.endswith("svg")


def main() -> None:
    manifest = json.loads(Path("data/manifest.json").read_text())
    configuration = json.loads(Path("results/configuration.json").read_text())
    audit = {r["benchmark"]: r for r in rows("audit.csv")}
    adjacency = rows("adjacent.csv.gz")
    with np.load("results/all_pair_tests.npz", allow_pickle=False) as tests:
        for benchmark in BENCHMARKS:
            output = manifest["outputs"][benchmark]
            path = Path("data") / output["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == output["sha256"]
            with np.load(path, allow_pickle=False) as source:
                y, models, subjects = source["correctness"], source["models"], source["subjects"]
                assert list(y.shape) == output["shape"]
                assert y.dtype == np.uint8 and np.all((y == 0) | (y == 1))
                assert len(models) == len(set(models)) == 395
                assert models.tolist() == manifest["source"]["model_ids"]
                assert len(source["item_ids"]) == len(set(source["item_ids"])) == len(y)
                assert len(subjects) == len(y)
                assert len(np.unique(subjects)) == (57 if benchmark == "mmlu" else len(y))
                ordering = np.lexsort((models, -y.mean(axis=0)))
            i, j = tests[f"{benchmark}_i"], tests[f"{benchmark}_j"]
            expected = np.triu_indices(395, 1)
            np.testing.assert_array_equal(i, expected[0])
            np.testing.assert_array_equal(j, expected[1])
            np.testing.assert_allclose(
                tests[f"{benchmark}_holm"], holm_adjust(tests[f"{benchmark}_p"]), rtol=1e-13
            )
            a = audit[benchmark]
            assert int(a["all_pair_family"]) == 395 * 394 // 2
            assert int(a["items"]) == output["shape"][0]
            sub = [r for r in adjacency if r["benchmark"] == benchmark]
            assert len(sub) == 394
            assert [r["model_a"] for r in sub] == models[ordering[:-1]].tolist()
            assert [r["model_b"] for r in sub] == models[ordering[1:]].tolist()
            significant = sum(float(r["full_family_holm_p"]) <= 0.05 for r in sub)
            assert significant == int(a["paired_full_holm_distinguishable"])
            assert (394 - significant) / 394 == float(
                a["paired_full_holm_not_distinguishable_fraction"]
            )
            if benchmark == "mmlu":
                np.testing.assert_allclose(
                    tests["mmlu_cluster_holm"], holm_adjust(tests["mmlu_cluster_p"]), rtol=1e-13
                )
    seen = set()
    for row in rows("validation_curves.csv.gz"):
        key = tuple(
            row[field] for field in ("benchmark", "pilot_n", "split", "pair", "design", "n")
        )
        assert key not in seen
        seen.add(key)
        assert int(row["trials"]) == configuration["trials_per_curve_point"]
        assert int(row["n"]) <= int(row["heldout_n"])
        rate = float(row["observed_paired"])
        assert 0 <= float(row["mc_low"]) <= rate + 1e-14
        assert rate - 1e-14 <= float(row["mc_high"]) <= 1
        assert abs(rate * int(row["trials"]) - round(rate * int(row["trials"]))) < 1e-9
    for name in (
        "pilot_planning.svg",
        "leaderboard_noise.svg",
        "minimum_difference.svg",
        "pilot_calibration.svg",
    ):
        assert ET.parse(Path("figures") / name).getroot().tag.endswith("svg")
    assert Path("README.md").read_text().rstrip().endswith("Written with AI coding assistance.")
    check_prospective()
    print(
        "Validated five pinned binary matrices, all-pair Holm families, "
        "calibration rows, fresh-item response counts and grades, and SVGs."
    )


if __name__ == "__main__":
    main()
