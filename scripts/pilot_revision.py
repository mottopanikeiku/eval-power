"""Explain the pre-confirmation repairs without replacing any inference files."""

import copy
import gzip
import hashlib
import json
from pathlib import Path

from eval_power.grading import grade_answer, grade_strict_answer
from eval_power.prospective import indexed_rows, model_summary

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/prospective"


def load(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def pin(path):
    return {
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def regrade(document, strict=False):
    result = copy.deepcopy(document)
    grader = grade_strict_answer if strict else grade_answer
    for benchmark, rows in result["benchmarks"].items():
        for row in rows:
            for answer in (row["greedy"], *row["samples"]):
                answer["prediction"], answer["correct"] = grader(
                    benchmark, answer["text"], row["reference"]
                )
    return result


def seed_counts(documents, protocol):
    seeds = []
    indices = {model["slug"]: i for i, model in enumerate(protocol["models"])}
    for slug, document in documents.items():
        for benchmark, rows in document["benchmarks"].items():
            for row in rows:
                base = (
                    protocol["generation_seed"]
                    + indices[slug] * 1000000
                    + (10000 if benchmark == "arc" else 0)
                    + int(row["item_id"]) * protocol.get("seed_stride", 1)
                )
                seeds.extend(base + j for j in range(protocol["k"]))
    return {
        "requested_child_streams": len(seeds),
        "distinct_child_seeds": len(set(seeds)),
        "reused_child_streams": len(seeds) - len(set(seeds)),
    }


def build_revision():
    settings = {
        "initial": ("protocol_initial.json", "pilot"),
        "long_arc": ("protocol_arc_long_collection.json", "arc_revised/pilot"),
        "guided_before_seed_fix": ("protocol_arc_direct_before_seed_fix.json", "arc_direct/pilot"),
        "primary": ("protocol.json", "primary/pilot"),
    }
    protocols, collections, sources = {}, {}, []
    for name, (protocol_file, directory) in settings.items():
        protocol_path = DIRECTORY / protocol_file
        protocol = json.loads(protocol_path.read_text())
        protocols[name] = protocol
        documents = {}
        for model in protocol["models"]:
            path = DIRECTORY / directory / f"{model['slug']}.json.gz"
            document = load(path)
            if document["model"] != model["id"] or document["revision"] != model["revision"]:
                raise ValueError("Pilot model identity changed")
            if document["metadata"]["protocol_sha256"] != pin(protocol_path)["sha256"]:
                raise ValueError("Pilot collection protocol does not match its source")
            documents[model["slug"]] = document
            sources.append(pin(path))
        collections[name] = documents
    primary_protocol = protocols["primary"]
    for name, protocol in protocols.items():
        for key in (
            "models",
            "k",
            "temperature",
            "top_p",
            "pilot_items",
            "pairs",
            "item_seed",
            "generation_seed",
            "alpha",
            "target_power",
            "minimum_confirm_items",
            "fresh_pool",
        ):
            if protocol[key] != primary_protocol[key]:
                raise ValueError(f"The measurement repair must not change {key}: {name}")
    for slug, original in collections["initial"].items():
        for documents in collections.values():
            for benchmark, rows in documents[slug]["benchmarks"].items():
                old = indexed_rows(original["benchmarks"][benchmark])
                if set(old) != set(indexed_rows(rows)):
                    raise ValueError("Pilot item selection changed")
                for row in rows:
                    if any(
                        row[key] != old[row["item_id"]][key]
                        for key in ("prompt_sha256", "reference")
                    ):
                        raise ValueError("Pilot question or reference changed")
    quality = {
        name: model_summary({slug: regrade(doc) for slug, doc in documents.items()}, 5)
        for name, documents in collections.items()
    }
    initial_strict = model_summary(
        {slug: regrade(doc, strict=True) for slug, doc in collections["initial"].items()}, 5
    )
    primary_strict = model_summary(
        {slug: regrade(doc, strict=True) for slug, doc in collections["primary"].items()}, 5
    )
    audits = {
        slug: document["metadata"]["stop_token_audit"]
        for slug, document in collections["primary"].items()
    }
    if any(audit["additional_chat_end_ids_needed"] for audit in audits.values()):
        raise ValueError("Native generation still lacks a declared assistant end token")
    return {
        "changed_after_pilot_before_confirmation": True,
        "reason": primary_protocol["pilot_revision"]["reason"],
        "protocols": {name: pin(DIRECTORY / files[0]) for name, files in settings.items()},
        "sources": sources,
        "primary_gsm8k_and_arc_recollected": True,
        "same_items_prompts_references": True,
        "arc_decoding": "guided_direct_choice",
        "stop_token_audit_by_model": audits,
        "seed_counts": {
            name: seed_counts(documents, protocols[name]) for name, documents in collections.items()
        },
        "models": {
            slug: {
                benchmark: {
                    "initial_strict": initial_strict[slug]["benchmarks"][benchmark],
                    "initial_flexible": quality["initial"][slug]["benchmarks"][benchmark],
                    "primary_strict": primary_strict[slug]["benchmarks"][benchmark],
                    "primary_flexible": quality["primary"][slug]["benchmarks"][benchmark],
                    **(
                        {
                            "long_unconstrained_flexible": quality["long_arc"][slug]["benchmarks"][
                                "arc"
                            ],
                            "guided_before_seed_fix": quality["guided_before_seed_fix"][slug][
                                "benchmarks"
                            ]["arc"],
                        }
                        if benchmark == "arc"
                        else {}
                    ),
                }
                for benchmark in primary_protocol["benchmarks"]
            }
            for slug in collections["primary"]
        },
        "interpretation": (
            "Initial strict versus flexible GSM8K isolates a grader change on identical text. "
            "The primary pilot is recollected after spacing parent seeds by k: vLLM uses "
            "parent_seed+child_index, so the earlier stride of one reused streams. ARC also "
            "changes its decoding distribution to guided direct-label generation; these "
            "are not conventional option-likelihood ARC scores or pure grader effects. "
            "Native EOS handling already included Phi's end-of-turn token and was not changed."
        ),
    }


def main():
    result = build_revision()
    (DIRECTORY / "pilot_revision.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Recorded all pilot sources, method changes, stop tokens and child-seed counts.")


if __name__ == "__main__":
    main()
