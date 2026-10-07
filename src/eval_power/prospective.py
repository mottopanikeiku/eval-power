"""Repeated-decoding pilot estimates and fresh-item paired confirmation.

The variance split assumes conditionally independent decodes within each model
and across models. A finite pilot does not identify its components perfectly:
the unclipped item estimate is reported even when it is negative.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from scipy.stats import norm, t

from .stats import paired_variance


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _alpha(value):
    value = _real(value, "alpha")
    if not 0 < value < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return value


def _samples(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] < 2 or a.shape[1] < 5:
        raise ValueError("a and b must have matching shape [items >= 2, k >= 5]")
    for values in (a, b):
        if values.dtype.kind not in "buif" or not np.all((values == 0) | (values == 1)):
            raise ValueError("samples must contain only binary values 0 and 1")
    return a.astype(np.float64, copy=False), b.astype(np.float64, copy=False)


def variance_decomposition(a, b):
    """Estimate item and independent-decoding variance for paired binary arrays.

    ``sampling_variance`` is mean(var(A_i) + var(B_i)), before division by k.
    ``total_variance`` is the planning variance of a single item's mean gap.
    Clipping a negative item estimate is a planning convention, not evidence
    that there is no item heterogeneity.
    """
    a, b = _samples(a, b)
    n, k = a.shape
    differences = a.mean(axis=1) - b.mean(axis=1)
    observed = float(differences.var(ddof=1))
    sampling = float((a.var(axis=1, ddof=1) + b.var(axis=1, ddof=1)).mean())
    unclipped = observed - sampling / k
    item = max(0.0, unclipped)
    total = item + sampling / k
    return {
        "n_items": n,
        "k": k,
        "mean_difference": float(differences.mean()),
        "observed_variance": observed,
        "sampling_variance": sampling,
        "item_variance_unclipped": unclipped,
        "item_variance": item,
        "total_variance": total,
        "sampling_fraction": sampling / k / total if total else None,
    }


def required_items(delta, total_variance, alpha=0.05, target_power=0.8):
    """Ceiling of (z_(1-alpha/2) + z_power)^2 * variance / gap^2.

    A zero pilot gap is nonestimable. A zero variance gives the literal count
    zero; protocol planning treats that pilot as nonestimable separately. This is
    the requested normal approximation, not exact t-test power.
    """
    delta = _real(delta, "delta")
    variance = _real(total_variance, "total_variance")
    alpha = _alpha(alpha)
    power = _real(target_power, "target_power")
    if abs(delta) > 1 or variance < 0:
        raise ValueError("absolute gap must be <= 1 and variance must be nonnegative")
    if not alpha < power < 1:
        raise ValueError("target_power must lie strictly between alpha and 1")
    if delta == 0:
        return None
    # Divide sequentially, avoiding underflow of delta**2 for small pilot gaps.
    count = (float(norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2 * variance / abs(delta)
             / abs(delta))
    if not math.isfinite(count):
        raise ValueError("required item count exceeds floating-point planning range")
    return math.ceil(count)


def plan_pair(a, b, fresh_pool, alpha=0.05, target_power=0.8):
    """Report the unconstrained item requirement and finite-pool feasibility."""
    fresh_pool = _integer(fresh_pool, "fresh_pool", minimum=0)
    result = variance_decomposition(a, b)
    count = required_items(result["mean_difference"], result["total_variance"], alpha, target_power)
    if result["total_variance"] == 0:
        count = None
    status = "nonestimable" if count is None else "feasible" if count <= fresh_pool else "infeasible"
    return {
        **result,
        "required_items": count,
        "fresh_pool": fresh_pool,
        "feasible": status == "feasible",
        "status": status,
    }


def paired_t_test(a, b, alpha=0.05):
    """Two-sided one-sample t-test on paired item means, with n-1 degrees of freedom.

    Identical zero differences give p=1; constant nonzero differences give p=0
    as the zero-standard-error limit. The statistic is null for either degenerate
    case so JSON never contains infinity or NaN.
    """
    a, b = _samples(a, b)
    alpha = _alpha(alpha)
    differences = a.mean(axis=1) - b.mean(axis=1)
    n = len(differences)
    mean = float(differences.mean())
    se = math.sqrt(float(differences.var(ddof=1)) / n)
    if se == 0:
        statistic, pvalue = None, 1.0 if mean == 0 else 0.0
        degenerate = "all_zero" if mean == 0 else "constant_nonzero"
    else:
        statistic = mean / se
        pvalue = float(2 * t.sf(abs(statistic), n - 1))
        degenerate = None
    radius = float(t.ppf(1 - alpha / 2, n - 1)) * se
    return {
        "n_items": n,
        "k": a.shape[1],
        "mean_difference": mean,
        "standard_error": se,
        "degrees_of_freedom": n - 1,
        "statistic": statistic,
        "degenerate": degenerate,
        "pvalue": pvalue,
        "detected": pvalue < alpha,
        "confidence_interval": [mean - radius, mean + radius],
        "alpha": alpha,
        "test": "two-sided paired t-test on item means",
    }


def wilson_interval(detected, total, alpha=0.05):
    """Wilson interval for an observed detection proportion; undefined for zero tests.

    Heterogeneous, overlapping model pairs are not independent replicates.
    This descriptive interval does not establish power for a single pair.
    """
    detected = _integer(detected, "detected", minimum=0)
    total = _integer(total, "total", minimum=0)
    alpha = _alpha(alpha)
    if detected > total:
        raise ValueError("detected must not exceed total")
    if total == 0:
        return None
    z2 = float(norm.ppf(1 - alpha / 2)) ** 2
    proportion = detected / total
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    radius = math.sqrt(z2 * (proportion * (1 - proportion) / total + z2 / (4 * total**2)))
    radius /= denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _item_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError("item_id must be a string or integer")
    if isinstance(value, str) and not value:
        raise ValueError("item_id must not be empty")
    return str(value)


def indexed_rows(rows):
    """Index list rows or JSON objects keyed by explicit item ID; reject duplicates."""
    if isinstance(rows, Mapping):
        entries = rows.items()
    elif isinstance(rows, list):
        entries = ((None, row) for row in rows)
    else:
        raise ValueError("benchmark rows must be a list or an item-ID-keyed object")
    indexed = {}
    for key, row in entries:
        if not isinstance(row, Mapping):
            raise ValueError("every benchmark row must be an object")
        if "item_id" not in row and key is None:
            raise ValueError("every row must have an explicit item_id")
        item_id = _item_id(row.get("item_id", key))
        if key is not None and _item_id(key) != item_id:
            raise ValueError("row item_id does not match its object key")
        if item_id in indexed:
            raise ValueError(f"duplicate item_id: {item_id}")
        indexed[item_id] = row
    return indexed


def aligned_samples(rows_a, rows_b, item_ids, k, *, exact=False, pilot_item_ids=()):
    """Select exactly the explicit IDs and k decodes; never truncate sample rows.

    exact=True requires both models' complete ID sets to equal the requested set.
    Otherwise rows outside this pair's selection can serve other planned pairs.
    All provided confirmation rows must be disjoint from the pilot IDs.
    """
    k = _integer(k, "k", minimum=5)
    a, b = indexed_rows(rows_a), indexed_rows(rows_b)
    ids = [_item_id(value) for value in item_ids]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("item_ids must contain at least two distinct explicit IDs")
    selected = set(ids)
    pilot = {_item_id(value) for value in pilot_item_ids}
    if (set(a) | set(b)) & pilot:
        raise ValueError("confirmation item IDs overlap the pilot")
    if not selected <= set(a) or not selected <= set(b):
        raise ValueError("a planned item_id is missing from a model")
    if exact and (set(a) != selected or set(b) != selected):
        raise ValueError("model item IDs do not match the requested item IDs")
    output = []
    for rows in (a, b):
        samples = []
        for item_id in ids:
            values = rows[item_id].get("samples")
            if not isinstance(values, list) or len(values) != k:
                raise ValueError(f"item {item_id} must contain exactly k={k} samples")
            if any(not isinstance(value, Mapping) or "correct" not in value for value in values):
                raise ValueError(f"item {item_id} samples need explicit correctness values")
            samples.append([value["correct"] for value in values])
        output.append(np.asarray(samples))
    return _samples(*output)


def _benchmark(document, name):
    try:
        return document["benchmarks"][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"raw model output is missing benchmark {name}") from exc


def _pilot_ids_setting(setting, benchmark, ids):
    if isinstance(setting, Mapping):
        if benchmark not in setting:
            raise ValueError(f"pilot_items is missing benchmark {benchmark}")
        setting = setting[benchmark]
    if isinstance(setting, list):
        expected = [_item_id(value) for value in setting]
        if len(set(expected)) != len(expected) or set(expected) != set(ids):
            raise ValueError("pilot item IDs differ from the plan")
    elif _integer(setting, "pilot_items", minimum=2) != len(ids):
        raise ValueError("pilot item count differs from the plan")


def _greedy_comparison(rows_a, rows_b, ids, fixed_gap, alpha, power):
    values = []
    for rows in (indexed_rows(rows_a), indexed_rows(rows_b)):
        try:
            values.append(np.asarray([rows[item_id]["greedy"]["correct"] for item_id in ids]))
        except (KeyError, TypeError) as exc:
            raise ValueError("pilot rows must include greedy correctness") from exc
    variance = paired_variance(*values)
    gap = float(values[0].mean() - values[1].mean())
    return {
        "mean_difference": gap,
        "paired_variance": variance,
        "required_items_same_fixed_gap": required_items(fixed_gap, variance, alpha, power),
        "fixed_gap": fixed_gap,
        "interpretation": (
            "Descriptive hypothetical greedy count at the repeated-decoding pilot gap, "
            "not greedy power at its own observed gap."
        ),
    }


def _answer_quality(answers):
    correctness = np.asarray([answer["correct"] for answer in answers])
    if (correctness.dtype.kind not in "buif"
            or not np.all((correctness == 0) | (correctness == 1))):
        raise ValueError("greedy/sample correctness must be binary")
    n = len(answers)
    missing_prediction = sum("prediction" not in answer for answer in answers)
    missing_finish = sum("finish_reason" not in answer for answer in answers)
    malformed = sum("prediction" in answer and answer["prediction"] is None for answer in answers)
    truncated = sum(answer.get("finish_reason") == "length" for answer in answers)
    return {
        "n_answers": n,
        "correct_count": int(correctness.sum()),
        "accuracy": float(correctness.mean()),
        "malformed_prediction_count": malformed,
        "malformed_prediction_rate": malformed / n if not missing_prediction else None,
        "truncated_count": truncated,
        "truncation_rate": truncated / n if not missing_finish else None,
        "missing_prediction_field_count": missing_prediction,
        "missing_finish_reason_field_count": missing_finish,
    }


def model_summary(models, k):
    """Describe accuracy, parsing, and truncation counts without changing grading."""
    result = {}
    for slug, document in models.items():
        benchmarks = {}
        for benchmark, raw_rows in document["benchmarks"].items():
            rows = indexed_rows(raw_rows)
            aligned_samples(raw_rows, raw_rows, list(rows), k, exact=True)
            try:
                greedy = _answer_quality([row["greedy"] for row in rows.values()])
                samples = _answer_quality([
                    answer for row in rows.values() for answer in row["samples"]
                ])
            except (KeyError, TypeError) as exc:
                raise ValueError("raw output lacks explicit greedy/sample correctness") from exc
            benchmarks[benchmark] = {
                "n_items": len(rows), "k": k,
                "greedy_accuracy": greedy["accuracy"],
                "mean_sampled_accuracy": samples["accuracy"],
                "greedy": greedy, "samples": samples,
            }
        result[slug] = {
            "model": document["model"], "revision": document["revision"], "benchmarks": benchmarks
        }
    return result


def validate_collection(plan, pilot_models, confirm_models):
    """Require each confirmation collection to equal its planned item-ID union."""
    expected, pilot_ids = {}, {}
    for document in pilot_models.values():
        for benchmark, rows in document["benchmarks"].items():
            pilot_ids.setdefault(benchmark, set()).update(indexed_rows(rows))
    for pair in plan["pairs"]:
        ids = {_item_id(value) for value in pair.get("item_ids", [])}
        for slug in (pair["model_a"], pair["model_b"]):
            expected.setdefault((slug, pair["benchmark"]), set()).update(ids)
    observed = {}
    for slug, document in confirm_models.items():
        if slug not in pilot_models:
            raise ValueError("unplanned confirmation model")
        for benchmark, rows in document["benchmarks"].items():
            ids = set(indexed_rows(rows))
            if ids != expected.get((slug, benchmark), set()):
                raise ValueError("confirmation item IDs differ from the planned union")
            if ids & pilot_ids.get(benchmark, set()):
                raise ValueError("confirmation item IDs overlap the pilot")
            if ids:
                aligned_samples(rows, rows, list(ids), plan["k"], exact=True)
            observed[slug, benchmark] = ids
    for identity, ids in expected.items():
        if observed.get(identity, set()) != ids:
            raise ValueError("confirmation output is missing planned union items")


def validate_sources(plan, protocol, pilot_models, confirm_models, protocol_sha256, plan_sha256):
    """Check raw-file protocol/plan hashes and pinned model identities."""
    for key in ("alpha", "target_power", "k"):
        if plan[key] != protocol[key]:
            raise ValueError(f"plan {key} differs from the pilot protocol")
    minimum = plan.get("minimum_items", plan.get("minimum_confirm_items", 16))
    if minimum != protocol["minimum_confirm_items"]:
        raise ValueError("plan minimum item count differs from the protocol")
    models = {model["slug"]: model for model in protocol["models"]}
    if set(pilot_models) != set(models):
        raise ValueError("pilot models differ from the protocol model set")
    planned_pairs = {
        (pair["benchmark"], *sorted((pair["model_a"], pair["model_b"]))) for pair in plan["pairs"]
    }
    protocol_pairs = {
        (benchmark, *sorted(pair))
        for benchmark in protocol["benchmarks"] for pair in protocol["pairs"]
    }
    if planned_pairs != protocol_pairs:
        raise ValueError("plan comparisons differ from the protocol")
    for stage, documents in (("pilot", pilot_models), ("confirm", confirm_models)):
        for slug, document in documents.items():
            model = models.get(slug)
            if (model is None or document.get("slug") != slug
                    or document.get("model") != model["id"]
                    or document.get("revision") != model["revision"]):
                raise ValueError("raw model identity/revision differs from the protocol")
            metadata = document.get("metadata", {})
            if metadata.get("protocol_sha256") != protocol_sha256:
                raise ValueError("raw protocol_sha256 differs from the protocol file")
            if stage == "confirm" and metadata.get("plan_sha256") != plan_sha256:
                raise ValueError("raw plan_sha256 differs from the plan file")
            if set(document["benchmarks"]) - set(protocol["benchmarks"]):
                raise ValueError("raw output includes an unplanned benchmark")
            if stage == "pilot":
                if set(document["benchmarks"]) != set(protocol["benchmarks"]):
                    raise ValueError("pilot benchmark set differs from the protocol")
                for benchmark, rows in document["benchmarks"].items():
                    ids = list(indexed_rows(rows))
                    _pilot_ids_setting(plan["pilot_items"], benchmark, ids)
                    if "pilot_items" in protocol and len(ids) != protocol["pilot_items"]:
                        raise ValueError("pilot item count differs from the protocol")
    for pair in plan["pairs"]:
        if pair.get("fresh_pool", pair.get("fresh_pool_items")) != protocol["fresh_pool"]:
            raise ValueError("plan fresh pool differs from the protocol")


def analyze_plan(plan, pilot_models, confirm_models):
    """Analyze only planned comparisons, with strict raw-data alignment checks.

    Dictionaries of raw documents are keyed by their explicit model slugs.
    Infeasible/nonestimable plans remain untested; missing feasible outputs error.
    """
    validate_collection(plan, pilot_models, confirm_models)
    alpha = _alpha(plan["alpha"])
    power = _real(plan["target_power"], "target_power")
    k = _integer(plan["k"], "k", minimum=5)
    minimum_items = _integer(
        plan.get("minimum_items", plan.get("minimum_confirm_items", 16)), "minimum_items", minimum=2
    )
    if not alpha < power < 1:
        raise ValueError("target_power must lie strictly between alpha and 1")
    if not isinstance(plan.get("protocol_commit"), str) or not plan["protocol_commit"]:
        raise ValueError("plan must identify protocol_commit")
    results, seen = [], set()
    for pair in plan["pairs"]:
        benchmark, name_a, name_b = pair["benchmark"], pair["model_a"], pair["model_b"]
        identity = (benchmark, *sorted((name_a, name_b)))
        if name_a == name_b or identity in seen:
            raise ValueError("planned comparisons must be distinct model pairs")
        seen.add(identity)
        try:
            pilot_a, pilot_b = pilot_models[name_a], pilot_models[name_b]
        except KeyError as exc:
            raise ValueError("a planned model has no pilot output") from exc
        for name, document in ((name_a, pilot_a), (name_b, pilot_b)):
            if (document.get("slug") != name or not document.get("model")
                    or not document.get("revision")):
                raise ValueError("pilot slug, model name, or pinned revision is missing/mismatched")
        rows_a, rows_b = _benchmark(pilot_a, benchmark), _benchmark(pilot_b, benchmark)
        ids = list(indexed_rows(rows_a))
        _pilot_ids_setting(plan["pilot_items"], benchmark, ids)
        a, b = aligned_samples(rows_a, rows_b, ids, k, exact=True)
        pilot = variance_decomposition(a, b)
        for field, value in (("pilot_difference", pilot["mean_difference"]),
                             ("pilot_variance", pilot["total_variance"])):
            if field in pair and not math.isclose(_real(pair[field], field), value,
                                                  rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{field} differs from the raw pilot")
        count = (required_items(pilot["mean_difference"], pilot["total_variance"], alpha, power)
                 if pilot["total_variance"] > 0 else None)
        scheduled_count = max(minimum_items, count) if count is not None else None
        result = {"benchmark": benchmark, "model_a": name_a, "model_b": name_b,
                  "pilot": pilot, "required_items": count, "planned_n_items": pair["n_items"]}
        result["required_items_k1"] = required_items(
            pilot["mean_difference"], pilot["item_variance"] + pilot["sampling_variance"],
            alpha, power,
        ) if pilot["total_variance"] > 0 else None
        result["required_items_k5"] = required_items(
            pilot["mean_difference"], pilot["item_variance"] + pilot["sampling_variance"] / 5,
            alpha, power,
        ) if pilot["total_variance"] > 0 else None
        result["greedy"] = _greedy_comparison(
            rows_a, rows_b, ids, pilot["mean_difference"], alpha, power
        )
        fresh_pool = pair.get("fresh_pool", pair.get("fresh_pool_items"))
        if fresh_pool is not None:
            fresh_pool = _integer(fresh_pool, "fresh_pool", minimum=0)
            result["fresh_pool"] = fresh_pool
        status = ("nonestimable" if count is None else
                  "infeasible" if fresh_pool is not None and scheduled_count > fresh_pool
                  else "feasible")
        if pair.get("status", status) != status:
            raise ValueError("planned feasibility status differs from the raw pilot")
        result["status"] = status
        if status != "feasible":
            if pair["n_items"] not in (None, 0) or pair.get("item_ids"):
                raise ValueError("infeasible/nonestimable comparisons cannot have confirmatory items")
            result["confirmation"] = None
            results.append(result)
            continue
        n = _integer(pair["n_items"], "n_items", minimum=2)
        if n != scheduled_count:
            raise ValueError("planned n_items differs from the uncapped requirement/minimum")
        item_ids = pair["item_ids"]
        if not isinstance(item_ids, list) or len(item_ids) != n:
            raise ValueError("planned item_ids must contain exactly n_items IDs")
        try:
            confirm_a, confirm_b = confirm_models[name_a], confirm_models[name_b]
        except KeyError as exc:
            raise ValueError("a feasible planned model has no confirmation output") from exc
        for name, document, pilot_doc in ((name_a, confirm_a, pilot_a),
                                           (name_b, confirm_b, pilot_b)):
            if (document.get("slug") != name or document.get("model") != pilot_doc["model"]
                    or document.get("revision") != pilot_doc["revision"]):
                raise ValueError("confirmation model or revision differs from the pilot")
        a, b = aligned_samples(_benchmark(confirm_a, benchmark), _benchmark(confirm_b, benchmark),
                               item_ids, k, pilot_item_ids=ids)
        result["confirmation"] = paired_t_test(a, b, alpha)
        results.append(result)
    tests = [row["confirmation"] for row in results if row["confirmation"] is not None]
    detected = sum(row["detected"] for row in tests)
    return {
        "protocol_commit": plan["protocol_commit"],
        "alpha": alpha,
        "target_power": power,
        "k": k,
        "pairs": results,
        "pilot_models": model_summary(pilot_models, k),
        "detection": {
            "detected": detected,
            "tested": len(tests),
            "proportion": detected / len(tests) if tests else None,
            "wilson_interval": wilson_interval(detected, len(tests), alpha),
            "interpretation": (
                "Observed detection across heterogeneous planned pairs, not single-pair power. "
                "Pairs share models and items and are not independent replicates; the Wilson "
                "interval is descriptive, not a dependence-adjusted coverage guarantee."
            ),
        },
        "limitations": [
            "The pilot variance split assumes conditionally independent decodes within and "
            "across models; its finite-pilot components are uncertain, not perfectly identified.",
            "Negative unclipped item estimates are retained; clipping to zero only sets the "
            "planning variance used in the normal approximation.",
            "Required items use a two-sided normal approximation; confirmation uses item-level "
            "paired t-tests. Pilot effect and variance estimates can be noisy.",
            "Detection uses nominal per-comparison alpha, without multiplicity correction.",
            "Formatting failures and output truncation can change measured accuracy; per-model "
            "pilot summaries report null predictions and length-limited outputs separately.",
            "Hypothetical k=1 and k=5 budgets use the same sampled pilot gap and estimated "
            "components, not separate empirical single-decode measurements.",
            "Zero-gap or zero-variance pilots are nonestimable under the protocol and are not "
            "confirmed; zero pilot variance does not prove zero population variance.",
        ],
    }
