"""Analyze a committed pilot plan and fresh-item repeated-decoding confirmation."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import scipy

from eval_power.prospective import analyze_plan, validate_sources

matplotlib.use("Agg")


def load_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def source_pin(path, label):
    return {"file": label, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_models(directory):
    models, pins = {}, []
    for path in sorted([*directory.glob("*.json"), *directory.glob("*.json.gz")]):
        document = load_json(path)
        slug = document.get("slug")
        expected = path.name.removesuffix(".gz").removesuffix(".json")
        if slug != expected or slug in models:
            raise ValueError(f"model slug must match its unique filename: {path.name}")
        models[slug] = document
        pins.append(source_pin(path, f"{directory.name}/{path.name}"))
    return models, pins


def make_figure(summary, destination):
    """Plot finite-pilot variance estimates and fresh-item confidence intervals."""
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "eval-power-prospective"
    pairs = summary["pairs"]
    height = max(4, len(pairs) * 0.25 + 2)
    figure, (variance_axis, confirm_axis) = plt.subplots(
        1, 2, figsize=(13, height), sharey=True, layout="constrained"
    )
    y = np.arange(len(pairs))
    labels = [f"{row['benchmark']}: {row['model_a']} − {row['model_b']}" for row in pairs]
    item = np.array([row["pilot"]["item_variance"] for row in pairs])
    sampling = np.array([row["pilot"]["sampling_variance"] / summary["k"] for row in pairs])
    unclipped = [row["pilot"]["item_variance_unclipped"] for row in pairs]
    variance_axis.barh(y, item, label="Item estimate (clipped)", color="#426b8a")
    variance_axis.barh(y, sampling, left=item, label="Decoding estimate / k", color="#e4a05c")
    variance_axis.scatter(unclipped, y, marker="x", color="#8d3444", label="Unclipped item")
    variance_axis.axvline(0, color="0.5", linewidth=0.8)
    variance_axis.set_yticks(y, labels)
    variance_axis.invert_yaxis()
    variance_axis.set_xlabel("Pilot variance estimate")
    variance_axis.set_title("Noisy item / decoding split")
    variance_axis.legend(fontsize=8, loc="best")
    confirm_axis.axvline(0, color="0.5", linewidth=0.8)
    for index, row in enumerate(pairs):
        confirmation = row["confirmation"]
        if confirmation is None:
            confirm_axis.annotate(
                row["status"],
                (0, index),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
            continue
        mean = confirmation["mean_difference"]
        low, high = confirmation["confidence_interval"]
        confirm_axis.errorbar(
            mean,
            index,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            capsize=2,
            color="#426b8a" if confirmation["detected"] else "0.55",
        )
    confirm_axis.set_xlabel("Fresh-item mean accuracy difference (A − B)")
    confirm_axis.set_title(f"Paired t intervals ({100 * (1 - summary['alpha']):g}%)")
    detection = summary["detection"]
    if detection["tested"]:
        low, high = detection["wilson_interval"]
        caption = (
            f"Observed detections: {detection['detected']}/{detection['tested']}; "
            f"descriptive Wilson interval [{low:.2f}, {high:.2f}]."
        )
    else:
        caption = "No feasible comparisons were tested."
    figure.suptitle(
        caption + "\nShared models/items are not independent trials or single-pair power.",
        fontsize=11,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg", metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("results/prospective/plan.json"))
    parser.add_argument("--protocol", type=Path, default=Path("results/prospective/protocol.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("results/prospective"))
    parser.add_argument("--summary", type=Path, default=Path("results/prospective/summary.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/prospective.svg"))
    args = parser.parse_args()
    plan, protocol = load_json(args.plan), load_json(args.protocol)
    pilot, pilot_pins = load_models(args.raw_dir / "pilot")
    confirm, confirm_pins = load_models(args.raw_dir / "confirm")
    protocol_pin = source_pin(args.protocol, args.protocol.name)
    plan_pin = source_pin(args.plan, args.plan.name)
    validate_sources(plan, protocol, pilot, confirm, protocol_pin["sha256"], plan_pin["sha256"])
    summary = analyze_plan(plan, pilot, confirm)
    summary["sources"] = {
        "protocol": protocol_pin,
        "plan": plan_pin,
        "pilot": pilot_pins,
        "confirm": confirm_pins,
    }
    summary["model_sources"] = protocol["models"]
    summary["benchmark_sources"] = protocol["benchmarks"]
    summary["software"] = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    # All raw/plan checks complete before writing either result file.
    make_figure(summary, args.figure)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
