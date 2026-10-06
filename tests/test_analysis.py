"""Fixed-input references for every sufficient-statistic/count-space shortcut."""

import itertools
import math

import numpy as np
import pytest
from scipy.stats import binomtest

from eval_power.analysis import (
    cell_counts,
    finite_detection_power,
    pairwise_metrics,
    rejection_rates,
    wilson_interval,
)
from eval_power.stats import clustered_se, holm_adjust


def test_all_pair_sufficient_statistics_match_slow_vectors():
    y = np.array([
        [1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1], [0, 0, 0, 1],
        [1, 1, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1],
    ], dtype=np.uint8)
    groups = np.array(["a", "a", "a", "b", "b", "c", "c", "c"])
    got = pairwise_metrics(y, groups)
    reference_p = []
    for row, (i, j) in enumerate(itertools.combinations(range(y.shape[1]), 2)):
        a, b = y[:, i].astype(float), y[:, j].astype(float)
        d = a - b
        wins, losses = sum(d == 1), sum(d == -1)
        p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1
        reference_p.append(p)
        assert got["gap"][row] == pytest.approx(d.mean())
        assert got["variance"][row] == pytest.approx(d.var(ddof=1))
        assert got["p"][row] == pytest.approx(p)
        assert got["cluster_se"][row] == pytest.approx(clustered_se(a, b, groups))
        if a.var() and b.var():
            assert got["rho"][row] == pytest.approx(np.corrcoef(a, b)[0, 1])
        else:
            assert math.isnan(got["rho"][row])
    np.testing.assert_allclose(got["holm"], holm_adjust(reference_p))


def test_matrix_product_does_not_overflow_byte_counts():
    y = np.ones((1024, 3), dtype=np.uint8)
    y[:256, 0] = 0
    got = pairwise_metrics(y)
    assert got["wins"][0] == 0
    assert got["losses"][0] == 256
    assert got["variance"][0] == pytest.approx(0.25 * 0.75 * 1024 / 1023)


def test_count_space_subsamples_equal_explicit_enumeration():
    # Enumerate all subsets; count-space testing must give exactly the same rate.
    a = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
    b = np.array([0, 0, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
    n = 6
    rows, reference = [], []
    for selection in itertools.combinations(range(len(a)), n):
        indices = np.array(selection)
        aa, bb = a[indices], b[indices]
        rows.append(cell_counts(aa, bb))
        wins = int(sum((aa == 1) & (bb == 0)))
        losses = int(sum((aa == 0) & (bb == 1)))
        p = binomtest(wins, wins + losses).pvalue if wins + losses else 1
        reference.append(p <= 0.05)
    paired, _ = rejection_rates(np.array(rows), n)
    assert paired == np.mean(reference)
    means = [float((a[list(s)].astype(float) - b[list(s)]).mean())
             for s in itertools.combinations(range(len(a)), n)]
    d = a.astype(float) - b
    expected_variance = (1 - n / len(a)) * d.var(ddof=1) / n
    assert np.var(means, ddof=0) == pytest.approx(expected_variance)


def test_cell_mapping_preserves_shared_item_pairing():
    assert cell_counts(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])).tolist() == [1]*4


def test_finite_correction_is_not_applied_to_test_threshold():
    assert finite_detection_power(0.01, 0.2, 1000, 1000) == 0
    assert finite_detection_power(0.1, 0.2, 1000, 1000) == 1
    with pytest.raises(ValueError):
        finite_detection_power(0.01, 0.2, 1001, 1000)


def test_monte_carlo_interval_extremes():
    low, high = wilson_interval(0, 1000)
    assert low == pytest.approx(0, abs=1e-15)
    assert 0 < high < 0.01
    low, high = wilson_interval(1000, 1000)
    assert 0.99 < low < 1
    assert high == pytest.approx(1)


def test_rejects_nonbinary_or_misaligned_clusters():
    with pytest.raises(ValueError):
        pairwise_metrics(np.array([[0.5, 1], [0, 1]]))
    with pytest.raises(ValueError):
        pairwise_metrics(np.array([[0, 1], [1, 0]]), np.array(["a"]))
