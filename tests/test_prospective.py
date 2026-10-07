"""Synthetic references for repeated-decoding planning and item-level confirmation."""

import copy
import math

import numpy as np
import pytest
from scipy.stats import norm, ttest_rel

from eval_power.prospective import (
    aligned_samples,
    analyze_plan,
    indexed_rows,
    paired_t_test,
    plan_pair,
    required_items,
    validate_sources,
    variance_decomposition,
    wilson_interval,
)


A = [[1, 1, 0, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
B = [[0, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0]]


def scalar_variance(values):
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def make_rows(ids, samples):
    return [
        {"item_id": item_id,
         "greedy": {"correct": row[0], "prediction": str(row[0]), "finish_reason": "stop"},
         "samples": [
             {"correct": value, "prediction": str(value), "finish_reason": "stop"} for value in row
         ]}
        for item_id, row in zip(ids, samples, strict=True)
    ]


def test_decomposition_matches_scalar_reference_and_uint8():
    actual = variance_decomposition(np.asarray(A, dtype=np.uint8), np.asarray(B, dtype=np.uint8))
    differences = [(sum(a) - sum(b)) / 5 for a, b in zip(A, B, strict=True)]
    sampling = sum(scalar_variance(a) + scalar_variance(b)
                   for a, b in zip(A, B, strict=True)) / 4
    observed = scalar_variance(differences)
    unclipped = observed - sampling / 5
    assert actual["n_items"] == 4
    assert actual["k"] == 5
    assert actual["mean_difference"] == pytest.approx(sum(differences) / 4)
    assert actual["observed_variance"] == pytest.approx(observed)
    assert actual["sampling_variance"] == pytest.approx(sampling)
    assert actual["item_variance_unclipped"] == pytest.approx(unclipped)
    assert actual["item_variance"] == pytest.approx(max(0, unclipped))
    assert actual["total_variance"] == pytest.approx(max(0, unclipped) + sampling / 5)


def test_negative_component_is_retained_not_hidden():
    samples = [[1, 1, 0, 0, 0]] * 3
    actual = variance_decomposition(samples, samples)
    assert actual["observed_variance"] == 0
    assert actual["item_variance_unclipped"] == pytest.approx(-0.12)
    assert actual["item_variance"] == 0
    assert actual["sampling_variance"] == pytest.approx(0.6)
    assert actual["total_variance"] == pytest.approx(0.12)


@pytest.mark.parametrize("k", [5, 7, 10])
def test_k_is_used_without_truncation(k):
    a = [[1] * k, [0] * k, [1] * k]
    b = [[0] * k] * 3
    assert variance_decomposition(a, b)["k"] == k
    aligned = aligned_samples(make_rows(["a", "b", "c"], a),
                              make_rows(["a", "b", "c"], b), ["a", "b", "c"], k)
    assert aligned[0].shape == (3, k)


@pytest.mark.parametrize("k", [0, 1, 4])
def test_decomposition_rejects_too_few_decodes(k):
    with pytest.raises(ValueError):
        variance_decomposition(np.zeros((3, k)), np.zeros((3, k)))


@pytest.mark.parametrize("a,b", [([[0] * 5], [[0] * 5]),
                                  ([[0] * 5] * 2, [[0] * 6] * 2),
                                  ([[2] * 5] * 2, [[0] * 5] * 2),
                                  ([[float("nan")] * 5] * 2, [[0] * 5] * 2)])
def test_bad_sample_arrays_error(a, b):
    with pytest.raises(ValueError):
        variance_decomposition(a, b)


def test_normal_item_requirement_matches_requested_formula():
    delta, variance = -0.08, 0.21
    expected = math.ceil((norm.ppf(0.975) + norm.ppf(0.8)) ** 2 * variance / delta**2)
    assert required_items(delta, variance) == expected
    assert required_items(-delta, variance) == expected
    assert required_items(0, variance) is None
    assert required_items(delta, 0) == 0


@pytest.mark.parametrize("delta,variance,alpha,power", [
    (2, 0.1, 0.05, 0.8), (0.1, -1, 0.05, 0.8),
    (0.1, 0.1, 0, 0.8), (0.1, 0.1, 0.05, 1),
])
def test_invalid_planning_inputs_error(delta, variance, alpha, power):
    with pytest.raises(ValueError):
        required_items(delta, variance, alpha, power)


def test_finite_pool_infeasibility_never_caps_requirement():
    estimate = variance_decomposition(A, B)
    count = required_items(estimate["mean_difference"], estimate["total_variance"])
    infeasible = plan_pair(A, B, count - 1)
    assert infeasible["required_items"] == count
    assert infeasible["status"] == "infeasible"
    assert infeasible["feasible"] is False
    assert plan_pair(A, B, count)["status"] == "feasible"
    assert plan_pair(A, A, 1000)["status"] == "nonestimable"


def test_explicit_ids_align_not_position_and_accept_keyed_rows():
    a = make_rows(["x", "y", "z", "w"], A)
    b = make_rows(["x", "y", "z", "w"], B)
    b = {row["item_id"]: row for row in reversed(b)}
    actual_a, actual_b = aligned_samples(a, b, ["z", "x"], 5)
    np.testing.assert_array_equal(actual_a, [A[2], A[0]])
    np.testing.assert_array_equal(actual_b, [B[2], B[0]])
    keyed = {"x": {"samples": [{"correct": 0}] * 5}}
    assert list(indexed_rows(keyed)) == ["x"]


def test_mismatched_ids_duplicates_and_pilot_overlap_error():
    a = make_rows(["x", "y", "z", "w"], A)
    b = make_rows(["x", "y", "other", "w"], B)
    with pytest.raises(ValueError, match="missing"):
        aligned_samples(a, b, ["x", "z"], 5)
    with pytest.raises(ValueError, match="match"):
        aligned_samples(a, b, ["x", "y"], 5, exact=True)
    with pytest.raises(ValueError, match="distinct"):
        aligned_samples(a, a, ["x", "x"], 5)
    with pytest.raises(ValueError, match="duplicate"):
        indexed_rows(a + a[:1])
    with pytest.raises(ValueError, match="object key"):
        indexed_rows({"different": a[0]})
    # Reject overlap even outside this particular pair's selected subset.
    with pytest.raises(ValueError, match="overlap"):
        aligned_samples(a, a, ["x", "y"], 5, pilot_item_ids=["w"])


@pytest.mark.parametrize("count", [4, 6])
def test_row_sample_count_must_equal_planned_k(count):
    rows = make_rows(["x", "y"], [[0] * count] * 2)
    with pytest.raises(ValueError, match="exactly"):
        aligned_samples(rows, rows, ["x", "y"], 5)


def test_paired_t_matches_scipy_known_reference():
    actual = paired_t_test(A, B)
    reference = ttest_rel(np.asarray(A).mean(axis=1), np.asarray(B).mean(axis=1))
    assert actual["statistic"] == pytest.approx(reference.statistic)
    assert actual["pvalue"] == pytest.approx(reference.pvalue)
    assert actual["degrees_of_freedom"] == 3
    np.testing.assert_allclose(actual["confidence_interval"], reference.confidence_interval())


def test_constant_paired_differences_have_explicit_limit():
    zeros, ones = np.zeros((3, 5)), np.ones((3, 5))
    assert paired_t_test(zeros, zeros)["pvalue"] == 1
    assert paired_t_test(ones, zeros)["pvalue"] == 0
    assert paired_t_test(ones, zeros)["statistic"] is None
    assert paired_t_test(ones, zeros)["degenerate"] == "constant_nonzero"


def test_wilson_reference_and_empty_detection():
    # Standard 95% Wilson score interval for 5 successes in 10 observations.
    assert wilson_interval(5, 10) == pytest.approx((0.236593090512564, 0.763406909487436))
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 10)[0] == pytest.approx(0)
    assert wilson_interval(10, 10)[1] == pytest.approx(1)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def example_plan():
    pilot_ids = ["p0", "p1", "p2", "p3"]
    pilot = {
        "a": {"slug": "a", "model": "org/a", "revision": "rev-a",
              "benchmarks": {"gsm8k": make_rows(pilot_ids, A)}},
        "b": {"slug": "b", "model": "org/b", "revision": "rev-b",
              "benchmarks": {"gsm8k": make_rows(pilot_ids, B)}},
    }
    estimate = variance_decomposition(A, B)
    n = max(16, required_items(estimate["mean_difference"], estimate["total_variance"]))
    ids = [f"c{i}" for i in range(n)]
    confirm = {
        name: {**document, "benchmarks": {"gsm8k": make_rows(ids, [samples[0]] * n)}}
        for (name, document), samples in zip(pilot.items(), (A, B), strict=True)
    }
    plan = {"protocol_commit": "abc123", "alpha": 0.05, "target_power": 0.8, "k": 5,
            "pilot_items": {"gsm8k": pilot_ids}, "pairs": [{
                "benchmark": "gsm8k", "model_a": "a", "model_b": "b", "n_items": n,
                "item_ids": ids, "fresh_pool": 512,
                "pilot_difference": estimate["mean_difference"],
                "pilot_variance": estimate["total_variance"],
            }]}
    return plan, pilot, confirm


def test_planned_analysis_uses_exact_count_and_marks_detection_descriptive():
    plan, pilot, confirm = example_plan()
    summary = analyze_plan(plan, pilot, confirm)
    assert summary["pairs"][0]["confirmation"]["n_items"] == plan["pairs"][0]["n_items"]
    assert summary["detection"]["tested"] == 1
    assert "not single-pair power" in summary["detection"]["interpretation"]
    assert "not independent" in summary["detection"]["interpretation"]


@pytest.mark.parametrize("mutation", ["count", "gap", "variance", "revision", "pilot_ids"])
def test_planned_analysis_rejects_mismatch(mutation):
    plan, pilot, confirm = example_plan()
    if mutation == "count":
        plan["pairs"][0]["n_items"] += 1
    elif mutation == "gap":
        plan["pairs"][0]["pilot_difference"] += 0.1
    elif mutation == "variance":
        plan["pairs"][0]["pilot_variance"] += 0.1
    elif mutation == "revision":
        confirm["a"]["revision"] = "changed"
    else:
        pilot["b"]["benchmarks"]["gsm8k"][0]["item_id"] = "other"
    with pytest.raises(ValueError):
        analyze_plan(plan, pilot, confirm)


def test_infeasible_pair_does_not_require_confirmation():
    plan, pilot, _ = example_plan()
    pair = plan["pairs"][0]
    pair.update(fresh_pool=1, n_items=None, item_ids=[], status="infeasible")
    summary = analyze_plan(plan, pilot, {})
    assert summary["pairs"][0]["required_items"] > 1
    assert summary["pairs"][0]["confirmation"] is None
    assert summary["detection"]["proportion"] is None
    assert summary["detection"]["wilson_interval"] is None


def test_zero_gap_pair_is_nonestimable_not_tested():
    plan, pilot, _ = example_plan()
    pilot["b"]["benchmarks"] = copy.deepcopy(pilot["a"]["benchmarks"])
    estimate = variance_decomposition(A, A)
    plan["pairs"][0].update(n_items=None, item_ids=[], status="nonestimable",
                             pilot_difference=0, pilot_variance=estimate["total_variance"])
    assert analyze_plan(plan, pilot, {})["pairs"][0]["status"] == "nonestimable"


def test_budget_and_greedy_comparisons_use_same_sampled_gap():
    plan, pilot, confirm = example_plan()
    row = analyze_plan(plan, pilot, confirm)["pairs"][0]
    variance = row["pilot"]
    gap = variance["mean_difference"]
    assert row["required_items_k1"] == required_items(
        gap, variance["item_variance"] + variance["sampling_variance"]
    )
    assert row["required_items_k5"] == required_items(
        gap, variance["item_variance"] + variance["sampling_variance"] / 5
    )
    assert row["greedy"]["fixed_gap"] == gap
    assert row["greedy"]["required_items_same_fixed_gap"] == required_items(
        gap, row["greedy"]["paired_variance"]
    )
    assert variance["sampling_fraction"] == pytest.approx(
        variance["sampling_variance"] / 5 / variance["total_variance"]
    )


def test_pilot_accuracy_malformed_and_truncation_counts():
    plan, pilot, confirm = example_plan()
    rows = pilot["a"]["benchmarks"]["gsm8k"]
    rows[0]["samples"][0].update(prediction=None, finish_reason="length")
    rows[1]["greedy"].update(prediction=None, finish_reason="length")
    quality = analyze_plan(plan, pilot, confirm)["pilot_models"]["a"]["benchmarks"]["gsm8k"]
    assert quality["greedy_accuracy"] == pytest.approx(0.75)
    assert quality["mean_sampled_accuracy"] == pytest.approx(0.45)
    assert quality["samples"]["n_answers"] == 20
    assert quality["samples"]["malformed_prediction_count"] == 1
    assert quality["samples"]["malformed_prediction_rate"] == pytest.approx(1 / 20)
    assert quality["samples"]["truncated_count"] == 1
    assert quality["samples"]["truncation_rate"] == pytest.approx(1 / 20)
    assert quality["greedy"]["malformed_prediction_rate"] == pytest.approx(1 / 4)


def test_zero_pilot_variance_is_nonestimable_under_protocol():
    zeros, ones = np.zeros((3, 5)), np.ones((3, 5))
    assert plan_pair(ones, zeros, 512)["status"] == "nonestimable"


@pytest.mark.parametrize("change", ["extra", "missing", "extra_k"])
def test_confirmation_collection_must_equal_planned_union(change):
    plan, pilot, confirm = example_plan()
    rows = confirm["a"]["benchmarks"]["gsm8k"]
    if change == "extra":
        extra = copy.deepcopy(rows[0])
        extra["item_id"] = "unplanned"
        rows.append(extra)
    elif change == "missing":
        rows.pop()
    else:
        rows[0]["samples"].append(copy.deepcopy(rows[0]["samples"][0]))
    with pytest.raises(ValueError):
        analyze_plan(plan, pilot, confirm)


def example_protocol(plan, pilot, confirm):
    protocol = {
        "alpha": 0.05, "target_power": 0.8, "k": 5, "minimum_confirm_items": 16,
        "fresh_pool": 512, "benchmarks": {"gsm8k": {}}, "pairs": [["a", "b"]],
        "models": [
            {"slug": slug, "id": document["model"], "revision": document["revision"]}
            for slug, document in pilot.items()
        ],
    }
    for document in pilot.values():
        document["metadata"] = {"protocol_sha256": "protocol-hash"}
    for document in confirm.values():
        document["metadata"] = {
            "protocol_sha256": "protocol-hash", "plan_sha256": "plan-hash"
        }
    return protocol


@pytest.mark.parametrize("change", ["none", "protocol_hash", "plan_hash", "revision", "model"])
def test_source_hashes_and_pinned_model_mapping(change):
    plan, pilot, confirm = example_plan()
    protocol = example_protocol(plan, pilot, confirm)
    if change == "none":
        validate_sources(plan, protocol, pilot, confirm, "protocol-hash", "plan-hash")
        return
    if change == "protocol_hash":
        pilot["a"]["metadata"]["protocol_sha256"] = "wrong"
    elif change == "plan_hash":
        confirm["a"]["metadata"]["plan_sha256"] = "wrong"
    elif change == "revision":
        pilot["a"]["revision"] = "wrong"
    else:
        confirm["a"]["model"] = "wrong"
    with pytest.raises(ValueError):
        validate_sources(plan, protocol, pilot, confirm, "protocol-hash", "plan-hash")
