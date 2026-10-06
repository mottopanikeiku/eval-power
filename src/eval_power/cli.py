"""Plan IID item counts using a fixed target difference and optional paired pilot."""

import argparse
import json

import numpy as np

from .stats import (
    clustered_se,
    correlation,
    paired_variance,
    required_items,
    unpaired_variance,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--difference", type=float, required=True, help="fixed target accuracy gap, not a pilot gap"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--variance", type=float, help="positive variance of one item difference")
    source.add_argument("--pilot", help="NumPy .npy array of binary outcomes, shape (n, 2)")
    parser.add_argument("--groups", help="aligned .npy group labels; only with --pilot")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    args = parser.parse_args(argv)
    if args.groups is not None and args.pilot is None:
        parser.error("--groups requires --pilot")

    result = {
        "difference": args.difference,
        "alpha": args.alpha,
        "target_power": args.power,
        "assumptions": [
            "Two-sided Gaussian approximation, not exact binary-test power.",
            "The target difference is fixed explicitly, not estimated from the pilot gap.",
            "The supplied or pilot-estimated variance transfers to the future evaluation.",
            "IID item sampling is assumed for item-count planning.",
            "Counts are items per model, not the total over both models.",
        ],
    }
    try:
        if args.pilot is None:
            result["variance"] = args.variance
            result["items_per_model"] = required_items(
                args.difference, args.variance, args.alpha, args.power
            )
        else:
            pilot = np.load(args.pilot, allow_pickle=False)
            if not isinstance(pilot, np.ndarray) or pilot.ndim != 2 or pilot.shape[1] != 2:
                raise ValueError("pilot must be a binary .npy array with shape (n, 2)")
            a, b = pilot[:, 0], pilot[:, 1]
            paired = paired_variance(a, b)
            unpaired = unpaired_variance(a, b)
            result.update(
                pilot_n=len(a),
                paired_variance=paired,
                unpaired_variance=unpaired,
                paired_correlation=correlation(a, b),
                paired_items_per_model=required_items(
                    args.difference, paired, args.alpha, args.power
                ),
                unpaired_items_per_model=required_items(
                    args.difference, unpaired, args.alpha, args.power
                ),
            )
            result["assumptions"].extend(
                [
                    "Paired planning uses same-item differences "
                    "and their sample variance (ddof=1).",
                    "Unpaired planning assumes independent model outcomes and binomial marginals.",
                ]
            )
            if args.groups is not None:
                groups = np.load(args.groups, allow_pickle=False)
                result["cluster_se"] = clustered_se(a, b, groups)
                result["group_count"] = len(np.unique(groups))
                result["assumptions"].append(
                    "Cluster SE is CR1 for the item-weighted gap and independent groups; "
                    "the reported IID item counts do not extrapolate cluster sampling. "
                    "More items within existing groups are not more independent groups."
                )
    except (OSError, ValueError, TypeError, EOFError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
