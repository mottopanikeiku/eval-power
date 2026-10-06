"""Historical accuracy audit and disjoint-pilot calibration, without model inference."""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm, t

from eval_power.stats import (
    gaussian_power,
    holm_adjust,
    minimum_detectable_difference,
    required_items,
)

BENCHMARKS = ("arc", "gsm8k", "winogrande", "hellaswag", "mmlu")
ALPHA = 0.05
TARGET_POWER = 0.8


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write measured rows with an explicit header, including compressed audit tables."""
    if not rows:
        raise ValueError(f"no rows for {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pairwise_metrics(y: np.ndarray, groups: np.ndarray | None = None) -> dict:
    """All unordered pairs from sufficient statistics; no K×K×N tensor."""
    if y.ndim != 2 or len(y) < 2 or not np.all((y == 0) | (y == 1)):
        raise ValueError("correctness must be a binary item × model matrix")
    n, k = y.shape
    i, j = np.triu_indices(k, 1)
    x = y.astype(np.float64)
    successes = x.sum(axis=0)
    both = x.T @ x
    wins = np.rint(successes[i] - both[i, j]).astype(np.int64)
    losses = np.rint(successes[j] - both[i, j]).astype(np.int64)
    gap = (wins - losses) / n
    discordance = (wins + losses) / n
    variance = (discordance - gap**2) * n / (n - 1)
    p = successes / n
    unpaired = p[i] * (1 - p[i]) + p[j] * (1 - p[j])
    denominator = np.sqrt(p[i] * (1 - p[i]) * p[j] * (1 - p[j]))
    covariance = both[i, j] / n - p[i] * p[j]
    rho = np.divide(
        covariance, denominator, out=np.full(len(i), np.nan), where=denominator > 0
    )
    exact_p = np.minimum(1, 2 * binom.cdf(np.minimum(wins, losses), wins + losses, 0.5))
    result = {
        "i": i,
        "j": j,
        "accuracy": p,
        "gap": gap,
        "variance": np.maximum(0, variance),
        "unpaired_variance": unpaired,
        "rho": rho,
        "wins": wins,
        "losses": losses,
        "p": exact_p,
        "holm": holm_adjust(exact_p),
    }
    if groups is not None:
        if len(groups) != n:
            raise ValueError("groups must align with items")
        labels, inverse = np.unique(groups, return_inverse=True)
        g = len(labels)
        if g < 2:
            raise ValueError("cluster inference requires at least two real groups")
        sizes = np.bincount(inverse)
        cluster_totals = np.zeros((g, k))
        np.add.at(cluster_totals, inverse, x)
        residuals = cluster_totals - sizes[:, None] * p[None, :]
        cross = residuals.T @ residuals
        cluster_variance = g / (g - 1) * (
            np.diag(cross)[i] + np.diag(cross)[j] - 2 * cross[i, j]
        ) / n**2
        cluster_se = np.sqrt(np.maximum(0, cluster_variance))
        statistic = np.divide(
            np.abs(gap), cluster_se, out=np.zeros(len(i)), where=cluster_se > 0
        )
        cluster_p = 2 * t.sf(statistic, g - 1)
        # Zero residual variation cannot support empirical generalization.
        cluster_p[cluster_se == 0] = 1.0
        result.update(
            cluster_se=cluster_se,
            cluster_p=cluster_p,
            cluster_holm=holm_adjust(cluster_p),
            group_count=g,
        )
    return result


def finite_detection_power(delta: float, variance: float, n: int, population: int) -> float:
    """Conditional detection of an IID Wald test under finite-pool subsampling.

    The test threshold keeps its IID standard error. Only the repeated-subsample
    distribution receives the finite population correction. This is NOT a
    superpopulation power guarantee, and is only a normal approximation.
    """
    if variance <= 0 or not 0 < n <= population:
        raise ValueError("positive variance and 0 < n <= population required")
    z = float(norm.ppf(1 - ALPHA / 2))
    threshold = z * math.sqrt(variance / n)
    spread = math.sqrt(variance / n * (1 - n / population))
    if spread == 0:
        return float(abs(delta) > threshold)
    return float(norm.cdf((-threshold - abs(delta)) / spread) + norm.sf(
        (threshold - abs(delta)) / spread
    ))


def rejection_rates(counts: np.ndarray, n: int) -> tuple[float, float]:
    """Exact paired McNemar and naive unpaired Wald rejection on identical draws."""
    both_wrong, b_only, a_only, both_right = counts.T
    m = a_only + b_only
    paired_p = np.minimum(1, 2 * binom.cdf(np.minimum(a_only, b_only), m, 0.5))
    a = (a_only + both_right) / n
    b = (b_only + both_right) / n
    se = np.sqrt((a * (1 - a) + b * (1 - b)) / n)
    gap = np.abs(a - b)
    statistic = np.divide(gap, se, out=np.zeros(len(se)), where=se > 0)
    unpaired_p = 2 * norm.sf(statistic)
    unpaired_p[(se == 0) & (gap > 0)] = 0
    return float(np.mean(paired_p <= ALPHA)), float(np.mean(unpaired_p <= ALPHA))


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Monte Carlo uncertainty for one conditional rejection rate."""
    p = successes / trials
    z = float(norm.ppf(0.975))
    divisor = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / divisor
    radius = z / divisor * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2))
    return center - radius, center + radius


def cell_counts(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Counts [both wrong, B only, A only, both right]."""
    return np.bincount(2 * a.astype(np.int8) + b.astype(np.int8), minlength=4)


def calibrate(
    y: np.ndarray,
    benchmark: str,
    models: np.ndarray,
    pairs: np.ndarray,
    splits: int,
    trials: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Freeze name-only pairs; fit pilots, then sample only disjoint heldout items.

    Count-space hypergeometric draws are exactly the same experiment as drawing
    aligned heldout rows without replacement. Multinomial draws are the
    empirical IID (with-replacement) comparison, not newly evaluated questions.
    """
    curves, plans, fixed_targets = [], [], []
    for pilot_n in (128, 256):
        for split in range(splits):
            rng = np.random.default_rng(np.random.SeedSequence([seed, pilot_n, split]))
            order = rng.permutation(len(y))
            pilot, heldout = order[:pilot_n], order[pilot_n:]
            available = len(heldout)
            sizes = [n for n in (64, 128, 256, 512, 1024) if n <= available]
            for pair_index, (i, j) in enumerate(pairs):
                a, b = y[:, i], y[:, j]
                pilot_d = a[pilot].astype(float) - b[pilot]
                heldout_d = a[heldout].astype(float) - b[heldout]
                delta, variance = float(pilot_d.mean()), float(pilot_d.var(ddof=1))
                heldout_delta = float(heldout_d.mean())
                heldout_variance = float(heldout_d.var(ddof=1))
                cells = cell_counts(a[heldout], b[heldout])
                p_a, p_b = float(a[pilot].mean()), float(b[pilot].mean())
                unpaired_variance = p_a * (1 - p_a) + p_b * (1 - p_b)
                identity = {
                    "benchmark": benchmark,
                    "pilot_n": pilot_n,
                    "split": split,
                    "pair": pair_index,
                    "model_a": str(models[i]),
                    "model_b": str(models[j]),
                    "pilot_gap": delta,
                    "heldout_gap": heldout_delta,
                    "pilot_variance": variance,
                    "heldout_variance": heldout_variance,
                    "heldout_n": available,
                }
                if variance > 0:
                    fixed_targets.append({
                        **identity,
                        "target_difference": 0.02,
                        "paired_items": required_items(0.02, variance),
                        "unpaired_items": required_items(0.02, unpaired_variance),
                        "heldout_variance_reference_items": (
                            required_items(0.02, heldout_variance)
                            if heldout_variance > 0 else ""
                        ),
                    })
                for n in sizes:
                    finite_counts = rng.multivariate_hypergeometric(cells, n, size=trials)
                    iid_counts = rng.multinomial(n, cells / available, size=trials)
                    finite_rate, finite_unpaired = rejection_rates(finite_counts, n)
                    iid_rate, iid_unpaired = rejection_rates(iid_counts, n)
                    for design, rate, naive_rate in (
                        ("without_replacement", finite_rate, finite_unpaired),
                        ("empirical_iid", iid_rate, iid_unpaired),
                    ):
                        low, high = wilson_interval(round(rate * trials), trials)
                        curves.append({
                            **identity,
                            "design": design,
                            "n": n,
                            "trials": trials,
                            "pilot_predicted_iid": (
                                gaussian_power(delta, variance, n) if variance > 0 else ""
                            ),
                            "pilot_predicted_finite": (
                                finite_detection_power(delta, variance, n, available)
                                if variance > 0 else ""
                            ),
                            "heldout_oracle_iid": (
                                gaussian_power(heldout_delta, heldout_variance, n)
                                if heldout_variance > 0 else ""
                            ),
                            "heldout_oracle_finite": (
                                finite_detection_power(
                                    heldout_delta, heldout_variance, n, available
                                ) if heldout_variance > 0 else ""
                            ),
                            "observed_paired": rate,
                            "observed_unpaired": naive_rate,
                            "mc_low": low,
                            "mc_high": high,
                        })
                # This separate diagnostic uses the noisy pilot gap as a design
                # alternative; it is NOT evidence that the original pair differs.
                plan_n = required_items(delta, variance) if delta != 0 and variance > 0 else None
                record = {
                    **identity,
                    "requested_power": TARGET_POWER,
                    "planned_n": plan_n if plan_n is not None else "",
                    "finite_attainable": plan_n is not None and plan_n <= available,
                    "iid_evaluated": plan_n is not None and plan_n <= 1_000_000,
                    "status": "zero_pilot_gap_or_variance" if plan_n is None else "planned",
                    "observed_finite": "",
                    "observed_iid": "",
                    "mc_low_iid": "",
                    "mc_high_iid": "",
                    "trials": trials,
                }
                if record["iid_evaluated"]:
                    draws = rng.multinomial(plan_n, cells / available, size=trials)
                    rate, _ = rejection_rates(draws, plan_n)
                    low, high = wilson_interval(round(rate * trials), trials)
                    record.update(observed_iid=rate, mc_low_iid=low, mc_high_iid=high)
                if record["finite_attainable"]:
                    draws = rng.multivariate_hypergeometric(cells, plan_n, size=trials)
                    rate, _ = rejection_rates(draws, plan_n)
                    record["observed_finite"] = rate
                plans.append(record)
    return curves, plans, fixed_targets


def summarize_calibration(curves: list[dict], plans: list[dict]) -> list[dict]:
    result = []
    for benchmark in BENCHMARKS:
        for pilot_n in (128, 256):
            subset = [p for p in plans if p["benchmark"] == benchmark and p["pilot_n"] == pilot_n]
            fitted = [p for p in subset if p["planned_n"] != ""]
            evaluated = [p for p in subset if p["iid_evaluated"]]
            finite = [p for p in subset if p["finite_attainable"]]
            relevant = [
                c for c in curves
                if c["benchmark"] == benchmark and c["pilot_n"] == pilot_n
                and c["design"] == "empirical_iid" and c["pilot_predicted_iid"] != ""
            ]
            finite_curves = [
                c for c in curves
                if c["benchmark"] == benchmark and c["pilot_n"] == pilot_n
                and c["design"] == "without_replacement" and c["pilot_predicted_finite"] != ""
            ]
            result.append({
                "benchmark": benchmark,
                "pilot_n": pilot_n,
                "pair_split_plans": len(subset),
                "plannable": len(fitted),
                "zero_gap_or_variance": len(subset) - len(fitted),
                "finite_attainable": len(finite),
                "finite_attainable_fraction": len(finite) / len(subset),
                "iid_evaluated": len(evaluated),
                "iid_power_median": float(np.median([p["observed_iid"] for p in evaluated])),
                "iid_power_q25": float(np.quantile([p["observed_iid"] for p in evaluated], 0.25)),
                "iid_power_q75": float(np.quantile([p["observed_iid"] for p in evaluated], 0.75)),
                "iid_below_target_95mc_fraction": (
                    sum(p["mc_high_iid"] < TARGET_POWER for p in evaluated) / len(evaluated)
                ),
                "iid_above_target_95mc_fraction": (
                    sum(p["mc_low_iid"] > TARGET_POWER for p in evaluated) / len(evaluated)
                ),
                "finite_power_median": (
                    float(np.median([p["observed_finite"] for p in finite])) if finite else ""
                ),
                "iid_curve_mean_absolute_error": float(np.mean([
                    abs(c["pilot_predicted_iid"] - c["observed_paired"]) for c in relevant
                ])),
                "iid_curve_within_10pp_fraction": float(np.mean([
                    abs(c["pilot_predicted_iid"] - c["observed_paired"]) <= 0.10
                    for c in relevant
                ])),
                "iid_oracle_mean_absolute_error": float(np.mean([
                    abs(c["heldout_oracle_iid"] - c["observed_paired"])
                    for c in relevant if c["heldout_oracle_iid"] != ""
                ])),
                "finite_curve_mean_absolute_error": float(np.mean([
                    abs(c["pilot_predicted_finite"] - c["observed_paired"])
                    for c in finite_curves
                ])),
                "finite_curve_within_10pp_fraction": float(np.mean([
                    abs(c["pilot_predicted_finite"] - c["observed_paired"]) <= 0.10
                    for c in finite_curves
                ])),
                "finite_oracle_mean_absolute_error": float(np.mean([
                    abs(c["heldout_oracle_finite"] - c["observed_paired"])
                    for c in finite_curves if c["heldout_oracle_finite"] != ""
                ])),
            })
    return result


def run_analysis(
    data_dir: Path,
    result_dir: Path,
    pair_count: int = 40,
    splits: int = 5,
    trials: int = 1000,
    seed: int = 20261006,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    audit, adjacent, lookup, mde, curves, plans, fixed_targets = [], [], [], [], [], [], []
    hypothesis_arrays = {}
    for benchmark in BENCHMARKS:
        with np.load(data_dir / f"{benchmark}.npz", allow_pickle=False) as source:
            y, models, subjects = source["correctness"], source["models"], source["subjects"]
        n, k = y.shape
        metrics = pairwise_metrics(y, subjects if benchmark == "mmlu" else None)
        i, j = metrics["i"], metrics["j"]
        score_order = np.lexsort((models, -metrics["accuracy"]))
        pair_lookup = {(int(a), int(b)): row for row, (a, b) in enumerate(zip(i, j, strict=True))}
        rows = [pair_lookup[tuple(sorted((int(a), int(b))))] for a, b in zip(
            score_order[:-1], score_order[1:], strict=True
        )]
        adjacent_p = metrics["p"][rows]
        adjacent_holm = holm_adjust(adjacent_p)
        positive = np.array([row for row in rows if metrics["variance"][row] > 0])
        ratios = np.sqrt(metrics["variance"][positive] / metrics["unpaired_variance"][positive])
        correlations = metrics["rho"][rows]
        audit.append({
            "benchmark": benchmark,
            "models": k,
            "items": n,
            "all_pair_family": len(i),
            "adjacent_pairs": len(rows),
            "tied_adjacent_scores": int(np.sum(metrics["gap"][rows] == 0)),
            "identical_adjacent_vectors": len(rows) - len(positive),
            "paired_uncorrected_distinguishable": int(np.sum(adjacent_p <= ALPHA)),
            "paired_adjacent_only_holm_distinguishable": int(np.sum(adjacent_holm <= ALPHA)),
            "paired_full_holm_distinguishable": int(np.sum(metrics["holm"][rows] <= ALPHA)),
            "paired_full_holm_not_distinguishable_fraction": float(np.mean(
                metrics["holm"][rows] > ALPHA
            )),
            "median_adjacent_item_correlation": float(np.nanmedian(correlations)),
            "adjacent_correlation_q25": float(np.nanquantile(correlations, 0.25)),
            "adjacent_correlation_q75": float(np.nanquantile(correlations, 0.75)),
            "median_nonzero_adjacent_paired_over_unpaired_se": float(np.median(ratios)),
            "median_nonzero_adjacent_item_saving_fraction": float(np.median(1 - ratios**2)),
            "median_all_pair_correlation": float(np.nanmedian(metrics["rho"])),
            "cluster_groups": metrics.get("group_count", ""),
            "cluster_full_holm_distinguishable": (
                int(np.sum(metrics["cluster_holm"][rows] <= ALPHA))
                if "cluster_holm" in metrics else ""
            ),
            "median_adjacent_cluster_over_paired_se": (
                float(np.median(metrics["cluster_se"][positive]
                                / np.sqrt(metrics["variance"][positive] / n)))
                if "cluster_se" in metrics else ""
            ),
        })
        for rank, row in enumerate(rows):
            a, b = score_order[rank : rank + 2]
            adjacent.append({
                "benchmark": benchmark,
                "rank_a": rank + 1,
                "model_a": str(models[a]),
                "model_b": str(models[b]),
                "accuracy_a": float(metrics["accuracy"][a]),
                "accuracy_b": float(metrics["accuracy"][b]),
                "difference": abs(float(metrics["gap"][row])),
                "paired_se": math.sqrt(float(metrics["variance"][row]) / n),
                "unpaired_se": math.sqrt(float(metrics["unpaired_variance"][row]) / n),
                "correlation": float(metrics["rho"][row]),
                "mcnemar_p": float(metrics["p"][row]),
                "adjacent_only_holm_p": float(adjacent_holm[rank]),
                "full_family_holm_p": float(metrics["holm"][row]),
                "cluster_se": (
                    float(metrics["cluster_se"][row]) if "cluster_se" in metrics else ""
                ),
                "cluster_full_holm_p": (
                    float(metrics["cluster_holm"][row]) if "cluster_holm" in metrics else ""
                ),
            })
        for difference in (0.005, 0.01, 0.02, 0.05):
            paired_counts = [required_items(difference, float(metrics["variance"][r])) for r in positive]
            unpaired_counts = [
                required_items(difference, float(metrics["unpaired_variance"][r])) for r in positive
            ]
            lookup.append({
                "benchmark": benchmark,
                "target_difference": difference,
                "alpha": ALPHA,
                "power": TARGET_POWER,
                "nonzero_adjacent_pairs": len(positive),
                "paired_items_median": float(np.median(paired_counts)),
                "paired_items_q25": float(np.quantile(paired_counts, 0.25)),
                "paired_items_q75": float(np.quantile(paired_counts, 0.75)),
                "unpaired_items_median": float(np.median(unpaired_counts)),
                "unpaired_items_q25": float(np.quantile(unpaired_counts, 0.25)),
                "unpaired_items_q75": float(np.quantile(unpaired_counts, 0.75)),
            })
        # Both normal tails are retained; the usual z_alpha + z_power is approximate.
        unit_noncentrality = minimum_detectable_difference(1.0, 1)
        for count in (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000):
            mde.append({
                "benchmark": benchmark,
                "items_per_model": count,
                "paired_mde": unit_noncentrality * math.sqrt(
                    float(np.median(metrics["variance"][positive])) / count
                ),
                "unpaired_mde": unit_noncentrality * math.sqrt(
                    float(np.median(metrics["unpaired_variance"][positive])) / count
                ),
            })
        for name in ("i", "j", "p", "holm", "cluster_p", "cluster_holm"):
            if name in metrics:
                hypothesis_arrays[f"{benchmark}_{name}"] = metrics[name]
        # Pair identities depend on names/indices and seed only, never observed scores.
        selection_rng = np.random.default_rng(seed)
        selected = selection_rng.choice(len(i), size=pair_count, replace=False)
        pair_ids = np.column_stack((i[selected], j[selected]))
        c, p, f = calibrate(y, benchmark, models, pair_ids, splits, trials, seed)
        curves.extend(c)
        plans.extend(p)
        fixed_targets.extend(f)
    write_csv(result_dir / "audit.csv", audit)
    write_csv(result_dir / "adjacent.csv.gz", adjacent)
    write_csv(result_dir / "lookup.csv", lookup)
    write_csv(result_dir / "mde.csv", mde)
    write_csv(result_dir / "validation_curves.csv.gz", curves)
    write_csv(result_dir / "pilot_plans.csv.gz", plans)
    write_csv(result_dir / "fixed_target_plans.csv.gz", fixed_targets)
    write_csv(result_dir / "calibration.csv", summarize_calibration(curves, plans))
    np.savez_compressed(result_dir / "all_pair_tests.npz", **hypothesis_arrays)
    configuration = {
        "seed": seed,
        "benchmarks": BENCHMARKS,
        "model_pair_selection": "uniform unordered index pairs, independent of scores",
        "pair_count_per_benchmark": pair_count,
        "pilot_sizes": [128, 256],
        "item_splits": splits,
        "trials_per_curve_point": trials,
        "alpha": ALPHA,
        "target_power": TARGET_POWER,
        "power_test": "exact two-sided conditional McNemar",
        "planning_formula": "two-tail Gaussian paired-difference approximation",
        "iid_plan_cap": 1_000_000,
        "cluster_estimand": "item-weighted accuracy; independently sampled MMLU subjects",
        "cluster_test": "CR1 standard error and t(56), Holm across all unordered model pairs",
        "multiplicity_family": "all unordered model pairs, separately within each benchmark",
        "finite_validation": "disjoint heldout items, without replacement; no future-item claim",
        "iid_validation": "with-replacement empirical heldout distribution, not new items",
        "pilot_gap_plans": "diagnostic transfer experiment, not evidence of model superiority",
    }
    (result_dir / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
