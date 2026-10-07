"""Export the committed Python planner's values for the browser test grid."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from eval_power.stats import (
    correlation,
    gaussian_power,
    paired_variance,
    required_items,
    unpaired_variance,
)


def fixture():
    cases = []
    grid = product(
        [0.001, 0.005, 0.01, 0.05, 0.2],
        [0.001, 0.02, 0.2, 0.5],
        [0.01, 0.05, 0.1],
        [0.6, 0.8, 0.95],
    )
    inputs = list(grid) + [
        (0.00001, 0.5, 0.000001, 0.999999),
        (-0.02, 0.2, 0.05, 0.8),
        (1.0, 0.0001, 0.05, 0.8),
        (0.01, 0.25, 0.5, 0.51),
        (0.01, 0.25, 0.05, 0.0501),
        (0.001, 0.25, 1e-8, 0.99),
    ]
    for delta, variance, alpha, power in inputs:
        n = required_items(delta, variance, alpha, power)
        cases.append(
            {
                "delta": delta,
                "variance": variance,
                "alpha": alpha,
                "power": power,
                "items": n,
                "power_at_items": gaussian_power(delta, variance, n, alpha),
                "power_before_items": (
                    gaussian_power(delta, variance, n - 1, alpha) if n > 1 else None
                ),
            }
        )
    comparisons = []
    # Counts for (both correct, A only, B only, neither correct).
    for both, a_only, b_only, neither in [
        (60, 10, 12, 18),
        (30, 40, 20, 10),
        (20, 0, 30, 50),
        (0, 30, 40, 30),
        (80, 0, 10, 10),
        (0, 0, 25, 75),
        (0, 100, 0, 0),
        (50, 25, 25, 0),
    ]:
        a = np.array([1] * both + [1] * a_only + [0] * b_only + [0] * neither)
        b = np.array([1] * both + [0] * a_only + [1] * b_only + [0] * neither)
        comparisons.append(
            {
                "pa": float(a.mean()),
                "pb": float(b.mean()),
                "rho": correlation(a, b),
                "discordance": (a_only + b_only) / len(a),
                # The UI takes population assumptions; undo sample ddof=1.
                "paired_variance": paired_variance(a, b) * (len(a) - 1) / len(a),
                "unpaired_variance": unpaired_variance(a, b),
            }
        )
    return {
        "source": "src/eval_power/stats.py",
        "item_tolerance": 1,
        "power_absolute_tolerance": 1e-12,
        "variance_absolute_tolerance": 1e-12,
        "cases": cases,
        "comparisons": comparisons,
    }


def check_fixture(reference, committed):
    """Check numerical agreement, allowing the documented cross-platform rounding."""
    if reference.keys() != committed.keys():
        raise SystemExit("Fixture fields changed; regenerate the fixture")
    for key in reference.keys() - {"cases", "comparisons"}:
        if reference[key] != committed[key]:
            raise SystemExit(f"Fixture metadata changed: {key}")
    for group in ("cases", "comparisons"):
        if len(reference[group]) != len(committed[group]):
            raise SystemExit(f"Fixture grid changed: {group}")
        for index, (actual, expected) in enumerate(
            zip(reference[group], committed[group], strict=True)
        ):
            if actual.keys() != expected.keys():
                raise SystemExit(f"Fixture fields changed: {group}[{index}]")
            for key, value in actual.items():
                old = expected[key]
                tolerance = 0
                if key == "items":
                    tolerance = reference["item_tolerance"]
                elif key.startswith("power_"):
                    tolerance = reference["power_absolute_tolerance"]
                elif key in ("rho", "paired_variance", "unpaired_variance"):
                    tolerance = reference["variance_absolute_tolerance"]
                if value is None or old is None:
                    matches = value is old
                else:
                    matches = abs(value - old) <= tolerance
                if not matches:
                    raise SystemExit(
                        f"Fixture is stale: {group}[{index}].{key}: {old} vs {value}; "
                        "run python site/export_fixture.py"
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Compare against the committed fixture"
    )
    args = parser.parse_args()
    destination = Path(__file__).with_name("python-fixture.json")
    reference = fixture()
    content = json.dumps(reference, indent=2, allow_nan=False) + "\n"
    if args.check:
        check_fixture(reference, json.loads(destination.read_text()))
        print("Python fixture matches the current planner within documented tolerances")
    else:
        destination.write_text(content)
        print(f"Exported {len(reference['cases'])} planner cases")


if __name__ == "__main__":
    main()
