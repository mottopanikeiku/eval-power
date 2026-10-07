"""Plan fresh-item counts from the completed pilot, before fresh inference."""

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

from eval_power.prospective import aligned_samples, plan_pair, validate_sources

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results/prospective/primary")
    directory = parser.parse_args().raw_dir
    destination = directory / "plan.json"
    if destination.exists():
        raise FileExistsError("Refusing to replace a committed experiment plan")
    protocol_path = ROOT / "results/prospective/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    raw = {
        model["slug"]: json.loads(
            gzip.decompress((directory / "pilot" / f"{model['slug']}.json.gz").read_bytes())
        )
        for model in protocol["models"]
    }
    pairs = []
    pilot_ids = {}
    for benchmark, source in protocol["benchmarks"].items():
        order = list(range(source["size"]))
        random.Random(protocol["item_seed"] + (1 if benchmark == "arc" else 0)).shuffle(order)
        pilot = [str(i) for i in order[: protocol["pilot_items"]]]
        pilot_ids[benchmark] = pilot
        fresh = order[protocol["pilot_items"] : protocol["pilot_items"] + protocol["fresh_pool"]]
        for model_a, model_b in protocol["pairs"]:
            a, b = aligned_samples(
                raw[model_a]["benchmarks"][benchmark],
                raw[model_b]["benchmarks"][benchmark],
                pilot,
                protocol["k"],
                exact=True,
            )
            stats = plan_pair(
                a,
                b,
                fresh_pool=protocol["fresh_pool"],
                alpha=protocol["alpha"],
                target_power=protocol["target_power"],
            )
            required = stats["required_items"]
            n = None if required is None else max(protocol["minimum_confirm_items"], required)
            status = (
                "nonestimable" if n is None else "feasible" if n <= len(fresh) else "infeasible"
            )
            pairs.append(
                {
                    "benchmark": benchmark,
                    "model_a": model_a,
                    "model_b": model_b,
                    "status": status,
                    "fresh_pool": len(fresh),
                    "required_items": required,
                    "n_items": n if status == "feasible" else None,
                    "item_ids": [str(i) for i in fresh[:n]] if status == "feasible" else [],
                    "pilot_difference": stats["mean_difference"],
                    "pilot_variance": stats["total_variance"],
                    "pilot_decomposition": stats,
                }
            )
    plan = {
        "protocol_file": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "alpha": protocol["alpha"],
        "target_power": protocol["target_power"],
        "metric": (
            "Flexible numeric-match GSM8K and guided direct-label ARC-Challenge accuracy. "
            "ARC is not conventional option-likelihood scoring. Strict-format scores "
            "remain a secondary analysis of the same answers."
        ),
        "k": protocol["k"],
        "item_seed": protocol["item_seed"],
        "generation_seed": protocol["generation_seed"],
        "generation_seed_rule": (
            "base + model_index*1000000 + (100000 for confirmation, else 0) "
            "+ (10000 for ARC, else 0) + dataset_index*k; child j uses parent_seed+j. "
            "Greedy decoding uses temperature zero."
        ),
        "minimum_items": protocol["minimum_confirm_items"],
        "pilot_items": pilot_ids,
        "pairs": pairs,
        "interpretation": (
            "One fresh comparison per feasible model pair and benchmark. Shared model answers "
            "make detections dependent; observed detection fraction is not an independent "
            "estimate of a single pair's power."
        ),
    }
    validate_sources(plan, protocol, raw, {}, plan["protocol_sha256"], "")
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                s: sum(p["status"] == s for p in pairs)
                for s in ("feasible", "infeasible", "nonestimable")
            }
        )
    )


if __name__ == "__main__":
    main()
