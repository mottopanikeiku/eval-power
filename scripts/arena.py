"""Project public Arena vote columns, then analyze committed numeric aggregates.

Import: uv run --with pyarrow==23.0.1 python scripts/arena.py --import-votes
Analyze: uv run python scripts/arena.py --bootstrap 500 --trials 200
No conversation, prompt, user identifier, or original row identifier is saved.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib
import numpy as np
import scipy
from scipy.stats import norm

from eval_power.arena import (
    ELO_SCALE,
    contrast,
    fit_bt,
    planned_votes,
    resample_counts,
    split_counts,
)

DATASET = "lmarena-ai/arena-human-preference-55k"
SOURCE_REVISION = "18c298340948c0e7f7727399fd459cca6ce0ca6f"
PARQUET_REVISION = "a0b4f36210e6a22263ccba29acbe87310c2383d0"
PARQUET_SIZE = 101531895
PARQUET_SHA256 = "9795a97ace213cb6e0c01adffa2d9070e550435b733260f2b9554397f3f46f46"
URL = (
    f"https://huggingface.co/datasets/{DATASET}/resolve/"
    f"{PARQUET_REVISION}/default/train/0000.parquet"
)
COLUMNS = ["model_a", "model_b", "winner_model_a", "winner_model_b", "winner_tie"]


class RangeReader(io.RawIOBase):
    """Seekable HTTP range reader; never fetch unselected conversation columns."""

    def __init__(self):
        super().__init__()
        self.position = 0
        self.ranges = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = (0, self.position, PARQUET_SIZE)[whence]
        self.position = base + offset
        if self.position < 0:
            raise ValueError("negative seek")
        return self.position

    def read(self, size=-1):
        size = min(size if size >= 0 else PARQUET_SIZE, PARQUET_SIZE - self.position)
        if size <= 0:
            return b""
        start, stop = self.position, self.position + size - 1
        request = Request(URL + f"?range_start={start}", headers={"Range": f"bytes={start}-{stop}"})
        with urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise ValueError(
                    "source must support range projection; refusing full text download"
                )
            expected = f"bytes {start}-{stop}/{PARQUET_SIZE}"
            if response.headers.get("Content-Range") != expected:
                raise ValueError("unexpected source byte range")
            value = response.read(size + 1)
        if len(value) != size:
            raise ValueError("unexpected source byte count")
        self.ranges.append(
            {"start": start, "bytes": size, "sha256": hashlib.sha256(value).hexdigest()}
        )
        self.position += size
        return value


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty result table")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def import_votes(args):
    import pyarrow.parquet as pq

    reader = RangeReader()
    parquet = pq.ParquetFile(reader)
    table = parquet.read(columns=COLUMNS, use_threads=False)
    columns = {name: table[name].to_pylist() for name in COLUMNS}
    names = sorted(set(columns["model_a"]) | set(columns["model_b"]))
    index = {name: i for i, name in enumerate(names)}
    a, b = np.triu_indices(len(names), 1)
    pairs = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(a, b, strict=True))}
    counts = np.zeros((len(a), 3), dtype=np.int64)
    self_votes = 0
    for ma, mb, wa, wb, tie in zip(*(columns[name] for name in COLUMNS), strict=True):
        outcome = [int(wa), int(wb), int(tie)]
        if sum(outcome) != 1 or any(value not in (0, 1) for value in outcome):
            raise ValueError("invalid winner flags")
        i, j = index[ma], index[mb]
        if i == j:
            self_votes += 1
            continue
        if i > j:
            i, j = j, i
            outcome[0], outcome[1] = outcome[1], outcome[0]
        counts[pairs[i, j]] += outcome
    pilot, heldout = split_counts(counts, np.random.default_rng(args.seed))
    aggregate = args.result_dir / "votes.npz"
    np.savez_compressed(aggregate, models=np.asarray(names), pilot=pilot, heldout=heldout)
    with urlopen("https://www.apache.org/licenses/LICENSE-2.0.txt", timeout=60) as response:
        (args.result_dir / "LICENSE.txt").write_bytes(response.read())
    save_json(
        args.manifest,
        {
            "dataset": DATASET,
            "source_revision": SOURCE_REVISION,
            "source_card_url": (
                f"https://huggingface.co/datasets/{DATASET}/blob/{SOURCE_REVISION}/README.md"
            ),
            "license": "Apache-2.0",
            "license_evidence": "license: apache-2.0 in the pinned source card",
            "attribution": "LMSYS / LMArena; Chiang et al., Chatbot Arena (2024), arXiv:2403.04132",
            "license_copy": "results/arena/LICENSE.txt",
            "converted_revision": PARQUET_REVISION,
            "parquet_url": URL,
            "parquet_size_bytes": PARQUET_SIZE,
            "parquet_lfs_sha256": PARQUET_SHA256,
            "checksum_scope": (
                "LFS full-file hash from pinned API tree; downloaded ranges checked separately"
            ),
            "projected_columns": COLUMNS,
            "downloaded_bytes": sum(item["bytes"] for item in reader.ranges),
            "downloaded_ranges": reader.ranges,
            "source_votes": len(columns["model_a"]),
            "self_comparison_votes_excluded": self_votes,
            "retained_votes": int(counts.sum()),
            "models": len(names),
            "split_seed": args.seed,
            "pilot_votes": int(pilot.sum()),
            "heldout_votes": int(heldout.sum()),
            "split": "independent Bernoulli(0.5) allocation of each vote; disjoint counts",
            "aggregate_file": "results/arena/votes.npz",
            "aggregate_sha256": hashlib.sha256(aggregate.read_bytes()).hexdigest(),
            "privacy": (
                "Only model names and unordered-pair outcome counts; no user or row IDs or text"
            ),
        },
    )


def analyze(args):
    with np.load(args.result_dir / "votes.npz", allow_pickle=False) as data:
        names = data["models"].tolist()
        pilot, heldout = data["pilot"], data["heldout"]
    models = len(names)
    a, b = np.triu_indices(models, 1)
    full = pilot + heldout
    degree = np.bincount(a, full.sum(axis=1), minlength=models)
    degree += np.bincount(b, full.sum(axis=1), minlength=models)
    # Eligibility depends on exposure only, never fitted ranks, gaps, or winners.
    eligible = np.flatnonzero(degree >= args.minimum_votes)
    mask = np.isin(a, eligible) & np.isin(b, eligible)
    names = [names[i] for i in eligible]
    pilot, heldout, full = pilot[mask], heldout[mask], full[mask]
    models = len(names)
    if models < 2:
        raise ValueError("fewer than two exposure-eligible models")
    fit = fit_bt(full, models)
    if fit.status != "ok":
        raise ValueError(f"full-data fit is {fit.status}; no global ranking is estimable")
    pilot_fit = fit_bt(pilot, models)
    heldout_fit = fit_bt(heldout, models)
    rng = np.random.default_rng(args.seed + 1)
    bootstrap_scores = []
    bootstrap_statuses = Counter()
    for _ in range(args.bootstrap):
        boot = fit_bt(resample_counts(full, int(full.sum()), rng), models)
        bootstrap_statuses[boot.status] += 1
        if boot.status == "ok":
            bootstrap_scores.append(boot.scores)
    boots = np.asarray(bootstrap_scores).reshape(-1, models)
    np.savez_compressed(args.result_dir / "bootstrap.npz", scores=boots)
    ranking = np.argsort(-fit.scores, kind="stable")
    ranked = []
    for rank, i in enumerate(ranking, start=1):
        ranked.append(
            {
                "rank": rank,
                "model": names[i],
                "score_log_odds": fit.scores[i],
                "score_elo_centered": fit.scores[i] * ELO_SCALE,
            }
        )
    save_csv(args.result_dir / "ranking.csv", ranked)
    adjacent = []
    for i, j in zip(ranking[:-1], ranking[1:], strict=True):
        gap, se = contrast(fit, i, j)
        low, high = (
            np.quantile(boots[:, i] - boots[:, j], [0.025, 0.975]) if len(boots) else (None, None)
        )
        votes, status = planned_votes(gap, se, int(full.sum()))
        adjacent.append(
            {
                "model_a": names[i],
                "model_b": names[j],
                "gap_log_odds": gap,
                "gap_elo": gap * ELO_SCALE,
                "wald_low_elo": (gap - 1.96 * se) * ELO_SCALE,
                "wald_high_elo": (gap + 1.96 * se) * ELO_SCALE,
                "bootstrap_low_elo": None if low is None else low * ELO_SCALE,
                "bootstrap_high_elo": None if high is None else high * ELO_SCALE,
                "total_votes_for_80pct": votes,
                "plan_status": status,
            }
        )
    save_csv(args.result_dir / "adjacent.csv", adjacent)
    # Fixed evenly spaced alphabetical neighbors, selected without score information.
    starts = np.unique(np.linspace(0, models - 2, min(args.pairs, models - 1), dtype=int))
    planning = []
    z = norm.ppf(0.975)
    for i in starts:
        j = i + 1
        gap, se = contrast(pilot_fit, i, j)
        n, status = planned_votes(gap, se, int(pilot.sum()))
        held_gap, held_se = contrast(heldout_fit, i, j)
        failures = Counter()
        rejected = 0
        successful = 0
        if status == "ok":
            for _ in range(args.trials):
                trial = fit_bt(resample_counts(heldout, n, rng), models)
                failures[trial.status] += 1
                if trial.status == "ok":
                    successful += 1
                    delta, trial_se = contrast(trial, i, j)
                    rejected += int(abs(delta / trial_se) > z)
        planning.append(
            {
                "model_a": names[i],
                "model_b": names[j],
                "pilot_fit_status": pilot_fit.status,
                "pilot_gap_log_odds": gap,
                "pilot_se": se,
                "planned_total_votes": n,
                "plan_status": status,
                "heldout_gap_log_odds": held_gap,
                "heldout_se": held_se,
                "trials": args.trials if status == "ok" else 0,
                "successful_fits": successful,
                "rejections": rejected,
                "rejection_rate_all_trials": rejected / args.trials if status == "ok" else None,
                "rejection_rate_conditional": rejected / successful if successful else None,
                "mc_se": np.sqrt(
                    (rejected / args.trials) * (1 - rejected / args.trials) / args.trials
                )
                if status == "ok"
                else None,
                "disconnected": failures["disconnected"],
                "separated": failures["separated"],
                "other_fit_failures": sum(
                    v for k, v in failures.items() if k not in ("ok", "disconnected", "separated")
                ),
            }
        )
    save_csv(args.result_dir / "pilot_planning.csv", planning)
    wald_excludes = sum(int(row["wald_low_elo"] > 0) for row in adjacent)
    bootstrap_excludes = sum(
        int(row["bootstrap_low_elo"] is not None and row["bootstrap_low_elo"] > 0)
        for row in adjacent
    )
    plan_sizes = [
        row["total_votes_for_80pct"] for row in adjacent if row["total_votes_for_80pct"] is not None
    ]
    rates = [
        row["rejection_rate_all_trials"]
        for row in planning
        if row["rejection_rate_all_trials"] is not None
    ]
    save_json(
        args.result_dir / "summary.json",
        {
            "eligible_models": models,
            "minimum_model_exposure": args.minimum_votes,
            "excluded_models": len(degree) - models,
            "retained_votes": int(full.sum()),
            "pilot_votes": int(pilot.sum()),
            "heldout_votes": int(heldout.sum()),
            "fit_status": fit.status,
            "pilot_fit_status": pilot_fit.status,
            "heldout_fit_status": heldout_fit.status,
            "adjacent_pairs": len(adjacent),
            "wald_intervals_excluding_zero": wald_excludes,
            "bootstrap_intervals_excluding_zero": bootstrap_excludes,
            "bootstrap_statuses": {
                status: bootstrap_statuses[status]
                for status in (
                    "ok",
                    "disconnected",
                    "separated",
                    "singular",
                    "line_search_failed",
                    "not_converged",
                )
            },
            "bootstrap_successes": len(boots),
            "bootstrap_attempts": args.bootstrap,
            "median_adjacent_total_votes_for_80pct": (
                float(np.median(plan_sizes)) if plan_sizes else None
            ),
            "adjacent_zero_or_nonestimable_plans": len(adjacent) - len(plan_sizes),
            "score_independent_planned_pairs": len(planning),
            "median_heldout_resampling_rejection_rate": float(np.median(rates)) if rates else None,
            "seed": args.seed,
            "methods": {
                "tie": (
                    "one vote with fractional outcome 0.5; ties and both-bad ties "
                    "are pooled upstream"
                ),
                "fit": "unpenalized Bradley-Terry, sum-zero log-odds scores; Elo=400/log(10)*score",
                "wald": "pointwise 95% inverse-Hessian curvature intervals; not sandwich corrected",
                "bootstrap": "pointwise percentile 95% IID vote bootstrap over pair/outcome counts",
                "bootstrap_failures": (
                    "reported by status; intervals conditional on successful fits"
                ),
                "adjacency": (
                    "descriptive full-data rank neighbors, not an independently selected test set"
                ),
                "selection": (
                    "exposure >= minimum; planning pairs evenly spaced alphabetic neighbors"
                ),
                "planning": (
                    "80% normal-approximation two-sided alpha=.05, total votes at pilot pair mix"
                ),
                "evaluation": (
                    "multinomial resampling of disjoint heldout votes at each pilot plan; "
                    "NOT new votes"
                ),
                "failure_denominator": (
                    "all-trial rejection rate counts nonestimable fits as nonrejections; "
                    "conditional also saved"
                ),
                "limitations": [
                    "IID votes; user/prompt clustering unavailable in retained aggregates",
                    "historical nonrepresentative competition subset, not a current leaderboard",
                    "BT assumes a single transitive preference strength across matchups",
                    "pointwise intervals; no multiplicity or rank-selection correction",
                    "heldout resampling measures empirical stability, not prospective power",
                    "eligibility is exposure-only but uses full sample exposures",
                ],
            },
        },
    )
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "arena-votes"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    x = np.arange(1, len(adjacent) + 1)
    for prefix, offset, color, label in (
        ("wald", -0.14, "#27689d", "Inverse Hessian"),
        ("bootstrap", 0.14, "#be6230", "Vote bootstrap"),
    ):
        lows = np.array([row[f"{prefix}_low_elo"] for row in adjacent], dtype=float)
        highs = np.array([row[f"{prefix}_high_elo"] for row in adjacent], dtype=float)
        axes[0].vlines(x + offset, lows, highs, color=color, alpha=0.75, label=label)
    axes[0].scatter(x, [row["gap_elo"] for row in adjacent], s=10, color="black")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set(
        xlabel="Neighbor pair in fitted rank order",
        ylabel="Elo-scale gap (95% intervals)",
        title="Many adjacent gaps include zero",
    )
    axes[0].legend(fontsize=8)
    valid = [row for row in planning if row["planned_total_votes"] is not None]
    axes[1].scatter(
        [row["planned_total_votes"] for row in valid],
        [row["rejection_rate_all_trials"] for row in valid],
        s=26,
    )
    axes[1].axhline(0.8, color="gray", linestyle="--", label="Planning target: 80%")
    axes[1].set(
        xscale="log",
        ylim=(-0.03, 1.03),
        xlabel="Pilot-planned total Arena votes",
        ylabel="Heldout resampling rejection rate",
        title="Historical resampling, not new votes",
    )
    axes[1].legend(fontsize=8)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, metadata={"Date": None})
    plt.close(fig)
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    save_json(
        args.result_dir / "environment.json",
        {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "os": platform.platform(),
            "cpu": cpu,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "threads": {
                name: os.environ.get(name, "unset")
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "command": "uv run python " + " ".join(sys.argv),
            "paid_compute_usd": 0,
            "timing_claim": "none",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-votes", action="store_true")
    parser.add_argument("--result-dir", type=Path, default=Path("results/arena"))
    parser.add_argument("--manifest", type=Path, default=Path("data/arena_manifest.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/arena_votes.svg"))
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--minimum-votes", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--pairs", type=int, default=12)
    args = parser.parse_args()
    if min(args.minimum_votes, args.bootstrap, args.trials, args.pairs) < 1:
        parser.error("minimum-votes, bootstrap, trials, and pairs must be positive")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.import_votes:
        import_votes(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
