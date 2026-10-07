"""Plan fresh-item counts from the completed pilot, before fresh inference."""

import gzip
import hashlib
import json
import random
import subprocess
from pathlib import Path

from eval_power.prospective import aligned_samples, plan_pair

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "results/prospective"
    destination = directory / "plan.json"
    if destination.exists():
        raise FileExistsError("Refusing to replace a committed experiment plan")
    protocol_path = directory / "protocol.json"
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
        "protocol_commit": subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(protocol_path.relative_to(ROOT))],
            cwd=ROOT,
            text=True,
        ).strip(),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "alpha": protocol["alpha"],
        "target_power": protocol["target_power"],
        "k": protocol["k"],
        "minimum_items": protocol["minimum_confirm_items"],
        "pilot_items": pilot_ids,
        "pairs": pairs,
        "interpretation": (
            "One fresh comparison per feasible model pair and benchmark. Shared model answers "
            "make detections dependent; observed detection fraction is not an independent "
            "estimate of a single pair's power."
        ),
    }
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
