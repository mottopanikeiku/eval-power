"""Scalar references for the paired statistics and planning approximations."""

import math

import numpy as np
import pytest
from scipy.stats import binomtest, norm

from eval_power import (
    clustered_se,
    correlation,
    gaussian_power,
    holm_adjust,
    mcnemar_pvalue,
    minimum_detectable_difference,
    paired_se,
    paired_variance,
    required_items,
    unpaired_se,
    unpaired_variance,
)


def scalar_variances(a, b):
    differences = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    n = len(differences)
    mean = sum(differences) / n
    paired = sum((value - mean) ** 2 for value in differences) / (n - 1)
    pa, pb = sum(a) / n, sum(b) / n
    unpaired = pa * (1 - pa) + pb * (1 - pb)
    return paired, unpaired


def scalar_cluster_se(a, b, groups):
    differences = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    mean = sum(differences) / len(differences)
    labels = set(groups)
    squared_sums = 0.0
    for label in labels:
        group_sum = 0.0
        for index in range(len(differences)):
            if groups[index] == label:
                group_sum += differences[index] - mean
        squared_sums += group_sum**2
    return math.sqrt(len(labels) / (len(labels) - 1) * squared_sums / len(differences) ** 2)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 1]),
        ([0, 1, 0, 1], [0, 1, 0, 1]),
        ([0, 0, 0], [1, 1, 1]),
        ([1, 0], [0, 0]),
    ],
)
def test_variances_and_standard_errors_against_scalar(a, b):
    paired, unpaired = scalar_variances(a, b)
    # Unsigned source data must be converted before A-B subtraction.
    a, b = np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8)
    assert paired_variance(a, b) == pytest.approx(paired)
    assert unpaired_variance(a, b) == pytest.approx(unpaired)
    assert paired_se(a, b) == pytest.approx(math.sqrt(paired / len(a)))
    assert unpaired_se(a, b) == pytest.approx(math.sqrt(unpaired / len(a)))


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([0, 1, 0, 1], [0, 1, 0, 1], 1.0),
        ([0, 1, 0, 1], [1, 0, 1, 0], -1.0),
        ([0, 0, 1, 1], [0, 1, 0, 1], 0.0),
        ([0, 0, 0], [0, 1, 1], None),
        ([0, 1, 1], [1, 1, 1], None),
    ],
)
def test_binary_correlation(a, b, expected):
    actual = correlation(a, b)
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "groups",
    [
        ["anatomy", "anatomy", "history", "history", "history", "math", "math"],
        [0, 1, 2, 3, 4, 5, 6],
        ["large", "large", "large", "large", "large", "large", "small"],
    ],
)
def test_cluster_se_against_nested_loop(groups):
    a, b = [1, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 0]
    assert clustered_se(a, b, groups) == pytest.approx(scalar_cluster_se(a, b, groups))
    if len(set(groups)) == len(groups):
        assert clustered_se(a, b, groups) == pytest.approx(paired_se(a, b))


def test_cluster_zero_residuals():
    assert clustered_se([1, 1, 1], [0, 0, 0], ["a", "a", "b"]) == 0


@pytest.mark.parametrize("groups", [["one"] * 3, ["a", "b"], [[0], [1], [2]], [0, 1, np.nan]])
def test_cluster_rejects_invalid_groups(groups):
    with pytest.raises(ValueError):
        clustered_se([0, 1, 0], [1, 0, 1], groups)


@pytest.mark.parametrize("a_only,b_only", [(0, 0), (1, 0), (12, 1), (3, 5), (6, 6), (0, 20)])
def test_exact_mcnemar_against_binomtest(a_only, b_only):
    a = [1] * a_only + [0] * b_only + [0, 1]
    b = [0] * a_only + [1] * b_only + [0, 1]
    expected = 1 if a_only + b_only == 0 else binomtest(a_only, a_only + b_only).pvalue
    assert mcnemar_pvalue(a, b) == pytest.approx(expected)
    assert mcnemar_pvalue(b, a) == pytest.approx(expected)


def test_holm_original_order_ties_and_monotonicity():
    # Sorted values .01,.01,.03,.04 get .04,.04,.06,.06 after the running maximum.
    np.testing.assert_allclose(holm_adjust([0.04, 0.01, 0.03, 0.01]), [0.06, 0.04, 0.06, 0.04])
    np.testing.assert_allclose(holm_adjust([0, 1, 0.5]), [0, 1, 1])
    assert holm_adjust([]).shape == (0,)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-0.01], [1.01], [[0.1]]])
def test_holm_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        holm_adjust(values)


@pytest.mark.parametrize(
    "statistic",
    [paired_variance, unpaired_variance, paired_se, unpaired_se, correlation, mcnemar_pvalue],
)
@pytest.mark.parametrize(
    ("a", "b"),
    [([], []), ([0, 1], [0]), ([[0, 1]], [[1, 0]]), ([0, 2], [1, 0]), ([0, np.nan], [0, 1])],
)
def test_pair_validation(statistic, a, b):
    with pytest.raises(ValueError):
        statistic(a, b)


def test_sample_variance_needs_two_items():
    with pytest.raises(ValueError, match="at least 2"):
        paired_variance([0], [1])
    assert unpaired_variance([0], [1]) == 0


@pytest.mark.parametrize(
    "delta,variance,n,alpha",
    [(0.01, 0.15, 12000, 0.05), (0.2, 0.5, 30, 0.1)],
)
def test_power_against_two_tail_normal_reference(delta, variance, n, alpha):
    z = norm.ppf(1 - alpha / 2)
    noncentrality = abs(delta) * math.sqrt(n / variance)
    expected = norm.cdf(-z - noncentrality) + norm.cdf(noncentrality - z)
    assert gaussian_power(delta, variance, n, alpha) == pytest.approx(expected)
    assert gaussian_power(-delta, variance, n, alpha) == pytest.approx(expected)
    assert gaussian_power(0, variance, n, alpha) == pytest.approx(alpha)


@pytest.mark.parametrize(
    "delta,variance,alpha,power",
    [
        (0.01, 0.15, 0.05, 0.8),
        (-0.2, 0.9, 0.1, 0.9),
        (1, 0.001, 0.05, 0.8),
        (0.1, 0.3, 0.2, 0.3),
    ],
)
def test_required_items_is_minimal_integer(delta, variance, alpha, power):
    n = required_items(delta, variance, alpha, power)
    assert isinstance(n, int) and n >= 1
    assert gaussian_power(delta, variance, n, alpha) >= power
    if n > 1:
        assert gaussian_power(delta, variance, n - 1, alpha) < power
    assert required_items(-delta, variance, alpha, power) == n


def test_required_items_matches_exhaustive_scalar_reference():
    delta, variance, alpha, power = 0.2, 0.3, 0.15, 0.7
    z = norm.ppf(1 - alpha / 2)
    for n in range(1, 1000):
        effect = delta * math.sqrt(n / variance)
        if norm.cdf(-z - effect) + norm.cdf(effect - z) >= power:
            break
    else:
        pytest.fail("scalar reference did not reach target power")
    assert required_items(delta, variance, alpha, power) == n


def test_minimum_detectable_difference_inverts_power_and_can_exceed_one():
    mde = minimum_detectable_difference(0.4, 200)
    assert gaussian_power(mde, 0.4, 200) == pytest.approx(0.8)
    assert minimum_detectable_difference(1, 1) > 1
    assert minimum_detectable_difference(0.4, 800) == pytest.approx(mde / 2)


@pytest.mark.parametrize("variance", [0, -1, np.nan, np.inf])
def test_planning_refuses_invalid_and_zero_variance(variance):
    with pytest.raises(ValueError, match="variance"):
        required_items(0.01, variance)
    with pytest.raises(ValueError, match="variance"):
        gaussian_power(0.01, variance, 100)
    with pytest.raises(ValueError, match="variance"):
        minimum_detectable_difference(variance, 100)


def test_observed_zero_variance_is_not_one_item_promise():
    variance = paired_variance([0, 1, 0, 1], [0, 1, 0, 1])
    with pytest.raises(ValueError, match="zero-variance pilot"):
        required_items(0.01, variance)


@pytest.mark.parametrize("delta", [0, 1.01, -1.01, np.nan, np.inf])
def test_required_items_requires_nonzero_binary_target(delta):
    with pytest.raises(ValueError):
        required_items(delta, 0.2)


@pytest.mark.parametrize("n", [0, -1, 2.0, True])
def test_planning_requires_positive_integer_n(n):
    with pytest.raises(ValueError, match="positive integer"):
        gaussian_power(0.1, 0.2, n)
    with pytest.raises(ValueError, match="positive integer"):
        minimum_detectable_difference(0.2, n)


@pytest.mark.parametrize(
    "alpha,power",
    [(0, 0.8), (1, 0.8), (np.nan, 0.8), (0.05, 0.05), (0.05, 1), (0.05, np.nan)],
)
def test_planning_rejects_invalid_probabilities(alpha, power):
    with pytest.raises(ValueError):
        required_items(0.1, 0.2, alpha, power)
    with pytest.raises(ValueError):
        minimum_detectable_difference(0.2, 100, alpha, power)


def test_nontrivial_correlation_against_scalar_reference():
    a, b = [0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 1, 1]
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    denominator = math.sqrt(
        sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
    )
    assert correlation(a, b) == pytest.approx(covariance / denominator)
    assert correlation(a, [1 - y for y in b]) == pytest.approx(-covariance / denominator)


@pytest.mark.parametrize("delta", [1.01, -1.01, np.nan, np.inf])
def test_gaussian_power_validates_delta(delta):
    with pytest.raises(ValueError):
        gaussian_power(delta, 0.2, 100)


@pytest.mark.parametrize("alpha", [0, 1, np.nan, np.inf])
def test_gaussian_power_validates_alpha(alpha):
    with pytest.raises(ValueError):
        gaussian_power(0.1, 0.2, 100, alpha)
