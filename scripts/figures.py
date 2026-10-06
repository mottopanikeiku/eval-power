"""Create SVG figures from committed result tables; no model inference."""

import csv
import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_power.analysis import BENCHMARKS

LABELS = {
    "arc": "ARC-C",
    "gsm8k": "GSM8K",
    "winogrande": "WinoGrande",
    "hellaswag": "HellaSwag",
    "mmlu": "MMLU",
}
COLORS = {"paired": "#2166ac", "unpaired": "#b35806", "128": "#2166ac", "256": "#b35806"}


def read_csv(name: str) -> list[dict]:
    path = Path("results") / name
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    """Crop the SVG canvas to its artists, with only a small outer margin."""
    fig.savefig(
        Path("figures") / name,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"Date": None},
    )
    plt.close(fig)


def main() -> None:
    Path("figures").mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.hashsalt": "eval-power",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    calibration = read_csv("calibration.csv")
    fig, ax = plt.subplots(figsize=(8, 4.3), layout="constrained")
    x = np.arange(len(BENCHMARKS))
    for pilot, offset in (("128", -0.16), ("256", 0.16)):
        rows = [
            next(r for r in calibration if r["benchmark"] == b and r["pilot_n"] == pilot)
            for b in BENCHMARKS
        ]
        center = np.array([float(r["iid_power_median"]) for r in rows])
        lo = np.array([float(r["iid_power_q25"]) for r in rows])
        hi = np.array([float(r["iid_power_q75"]) for r in rows])
        ax.errorbar(
            x + offset,
            center,
            yerr=[center - lo, hi - center],
            fmt="o",
            capsize=4,
            color=COLORS[pilot],
            label=f"{pilot}-item pilot: median and IQR",
        )
    ax.axhline(0.8, color="#444444", ls="--", lw=1, label="Planned power: 80%")
    ax.set(
        xticks=x,
        xticklabels=[LABELS[b] for b in BENCHMARKS],
        ylim=(0, 1.03),
        ylabel="Heldout empirical-IID detection rate",
        title="A noisy pilot gap is not an 80%-power guarantee",
    )
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.text(
        0.5,
        -0.03,
        "Fixed name-only pairs; disjoint pilot/heldout items; "
        "heldout resampling, not new model runs.",
        ha="center",
        fontsize=8,
    )
    save(fig, "pilot_planning.svg")

    audit = read_csv("audit.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    unresolved = [100 * float(r["paired_full_holm_not_distinguishable_fraction"]) for r in audit]
    axes[0].barh([LABELS[r["benchmark"]] for r in audit], unresolved, color="#2166ac")
    axes[0].set(
        xlim=(0, 100),
        xlabel="Adjacent gaps not distinguishable (%)",
        title="Exact McNemar; whole-family Holm",
    )
    for index, value in enumerate(unresolved):
        axes[0].text(value - 2, index, f"{value:.1f}%", ha="right", va="center", color="white")
    ratios = [float(r["median_nonzero_adjacent_paired_over_unpaired_se"]) for r in audit]
    axes[1].barh([LABELS[r["benchmark"]] for r in audit], ratios, color="#b35806")
    axes[1].axvline(1, ls="--", lw=1, color="#555555")
    axes[1].set(
        xlim=(0, 1.05),
        xlabel="Median paired / unpaired standard error",
        title="Same-item pairing reduces noise",
    )
    fig.suptitle(
        "Selected 2024 Open LLM Leaderboard models, not current frontier models", fontsize=11
    )
    save(fig, "leaderboard_noise.svg")

    mde = read_csv("mde.csv")
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.1), sharey=True, layout="constrained")
    for ax, benchmark in zip(axes, BENCHMARKS, strict=True):
        rows = [r for r in mde if r["benchmark"] == benchmark]
        counts = [int(r["items_per_model"]) for r in rows]
        for method in ("paired", "unpaired"):
            ax.loglog(
                counts,
                [100 * float(r[f"{method}_mde"]) for r in rows],
                color=COLORS[method],
                label=method,
            )
        ax.set(
            title=LABELS[benchmark],
            xlabel="Items / model",
            xticks=[100, 1000, 10000],
            ylim=(0.1, 30),
        )
        ax.grid(alpha=0.15, which="both")
    axes[0].set_ylabel("80%-power minimum difference (pp)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Fixed empirical variance: median non-identical adjacent pairs; α = 0.05", fontsize=10
    )
    save(fig, "minimum_difference.svg")

    curves = read_csv("validation_curves.csv.gz")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), sharex=True, sharey=True, layout="constrained")
    for ax, design, key, label in (
        (axes[0], "empirical_iid", "pilot_predicted_iid", "Empirical IID"),
        (axes[1], "without_replacement", "pilot_predicted_finite", "Finite heldout subsampling"),
    ):
        ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1)
        for pilot in ("128", "256"):
            valid = [
                r
                for r in curves
                if r["design"] == design and r["pilot_n"] == pilot and r[key] != ""
            ]
            predicted = np.array([float(r[key]) for r in valid])
            observed = np.array([float(r["observed_paired"]) for r in valid])
            means_x, means_y = [], []
            for lo in np.arange(0, 1, 0.1):
                chosen = (predicted >= lo) & (predicted < lo + 0.1 + 1e-12)
                if chosen.any():
                    means_x.append(predicted[chosen].mean())
                    means_y.append(observed[chosen].mean())
            ax.plot(means_x, means_y, "o-", color=COLORS[pilot], label=f"Pilot n={pilot}")
        ax.set(
            title=label, xlabel="Pilot-predicted detection probability", xlim=(0, 1), ylim=(0, 1)
        )
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Observed exact-McNemar rejection rate")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Out-of-pilot calibration; probability-bin means across five historical benchmarks",
        fontsize=10,
    )
    save(fig, "pilot_calibration.svg")


if __name__ == "__main__":
    main()
