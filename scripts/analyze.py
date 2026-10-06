"""Recompute audit, pilot calibration, and sample-size lookup from derived matrices."""

import argparse
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import scipy

from eval_power.analysis import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    if min(args.pairs, args.splits, args.trials) < 1:
        parser.error("pairs, splits, and trials must be positive")
    run_analysis(args.data_dir, args.result_dir, args.pairs, args.splits, args.trials, args.seed)
    cpu = (
        next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
        if Path("/proc/cpuinfo").exists()
        else platform.processor()
    )
    environment = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "cpu": cpu,
        "os": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "threads": {
            name: os.environ.get(name, "not set")
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "analysis_command": "uv run python " + " ".join(sys.argv),
        "paid_compute_usd": 0,
        "model_inference": "not run",
        "timing_benchmark": "not run; no wall-clock performance claim",
    }
    (args.result_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")


if __name__ == "__main__":
    main()
