"""Scalar likelihood and analytical references for aggregated Arena statistics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import norm

from eval_power.arena import (
    ELO_SCALE,
    contrast,
    fit_bt,
    planned_votes,
    resample_counts,
    split_counts,
)


def scalar_likelihood(counts, scores):
    """Expand each tiny reference vote, including one fractional tie outcome."""
    value = 0.0
    row = 0
    for a in range(len(scores)):
        for b in range(a + 1, len(scores)):
            p = expit(scores[a] - scores[b])
            for outcome, count in zip((1.0, 0.0, 0.5), counts[row], strict=True):
                for _ in range(int(count)):
                    value += outcome * np.log(p) + (1 - outcome) * np.log1p(-p)
            row += 1
    return value


@pytest.mark.parametrize("counts", ([[30, 10, 0]], [[24, 4, 12]], [[0, 0, 40]]))
def test_two_model_fit_and_hessian_against_closed_form(counts):
    counts = np.asarray(counts)
    n = counts.sum()
    p = (counts[0, 0] + counts[0, 2] / 2) / n
    fit = fit_bt(counts, 2)
    assert fit.status == "ok"
    delta, se = contrast(fit, 0, 1)
    assert delta == pytest.approx(np.log(p / (1 - p)), abs=1e-8)
    assert se**2 == pytest.approx(1 / (n * p * (1 - p)))
    assert fit.scores.sum() == pytest.approx(0, abs=1e-12)
    assert fit.covariance.sum(axis=0) == pytest.approx([0, 0], abs=1e-12)
    assert ELO_SCALE * np.log(10) == pytest.approx(400)


def test_three_model_fit_matches_scalar_vote_likelihood():
    counts = np.array([[20, 10, 0], [40, 10, 0], [20, 10, 0]])
    fit = fit_bt(counts, 3)
    expected = np.array([np.log(2), 0, -np.log(2)])
    assert fit.status == "ok"
    assert fit.scores == pytest.approx(expected, abs=1e-8)
    optimum = scalar_likelihood(counts, fit.scores)
    for direction in np.eye(3):
        assert scalar_likelihood(counts, fit.scores + direction * 0.01) < optimum
        assert scalar_likelihood(counts, fit.scores - direction * 0.01) < optimum
    # Scalar weighted graph Laplacian, restricted to an anchored model.
    hessian = np.zeros((3, 3))
    row = 0
    for a in range(3):
        for b in range(a + 1, 3):
            p = expit(expected[a] - expected[b])
            weight = counts[row].sum() * p * (1 - p)
            direction = np.eye(3)[a] - np.eye(3)[b]
            hessian += weight * np.outer(direction, direction)
            row += 1
    anchored = np.linalg.inv(hessian[:2, :2])
    _, se = contrast(fit, 0, 1)
    assert se**2 == pytest.approx(anchored[0, 0] + anchored[1, 1] - 2 * anchored[0, 1])


@pytest.mark.parametrize(
    ("counts", "models", "status"),
    [
        ([[1, 0, 0]], 2, "separated"),
        ([[0, 0, 0]], 2, "disconnected"),
        ([[1, 1, 0], [0, 0, 0], [0, 0, 0]], 3, "disconnected"),
        ([[1, 0, 0], [1, 0, 0], [1, 0, 0]], 3, "separated"),
    ],
)
def test_graph_failures_are_not_regularized_away(counts, models, status):
    fit = fit_bt(counts, models)
    assert fit.status == status
    assert contrast(fit, 0, 1) == (None, None)


def test_ties_connect_directed_win_graph():
    fit = fit_bt([[0, 0, 2], [0, 0, 0], [0, 0, 2]], 3)
    assert fit.status == "ok"
    assert fit.scores == pytest.approx([0, 0, 0])


@pytest.mark.parametrize("counts", ([[1.5, 2, 3]], [[-1, 2, 3]], [[1, np.nan, 3]]))
def test_invalid_counts_rejected(counts):
    with pytest.raises(ValueError):
        fit_bt(counts, 2)


def test_split_is_disjoint_and_exhaustive():
    counts = np.array([[10, 20, 30], [5, 0, 2], [0, 8, 1]])
    pilot, heldout = split_counts(counts, np.random.default_rng(2026))
    assert np.array_equal(pilot + heldout, counts)
    assert np.all(pilot >= 0) and np.all(heldout >= 0)
    same, _ = split_counts(counts, np.random.default_rng(2026))
    assert np.array_equal(pilot, same)


def test_aggregate_bootstrap_matches_expanded_vote_category_probabilities():
    counts = np.array([[2, 1, 1], [1, 3, 0], [1, 0, 1]])
    expanded = []
    for category, count in enumerate(counts.ravel()):
        expanded.extend([category] * count)
    scalar_probabilities = [expanded.count(k) / len(expanded) for k in range(counts.size)]
    reference = np.random.default_rng(7).multinomial(27, scalar_probabilities)
    result = resample_counts(counts, 27, np.random.default_rng(7))
    assert np.array_equal(reference, result.ravel())
    assert result.sum() == 27
    assert result[1, 2] == 0 and result[2, 1] == 0


def test_planning_uses_two_sided_power_and_reports_zero_gap():
    votes, status = planned_votes(0.1, 0.05, 1000)
    assert status == "ok"
    z = norm.ppf(0.975)
    noncentrality = 0.1 / 0.05 * np.sqrt(votes / 1000)
    power = norm.cdf(-z - noncentrality) + norm.cdf(noncentrality - z)
    assert power >= 0.8
    previous = 0.1 / 0.05 * np.sqrt((votes - 1) / 1000)
    assert norm.cdf(-z - previous) + norm.cdf(previous - z) < 0.8
    assert planned_votes(0, 0.05, 1000) == (None, "zero_gap")
    assert planned_votes(None, None, 1000) == (None, "nonestimable")


def test_committed_votes_match_source_manifest_and_contain_no_text():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data/arena_manifest.json").read_text())
    aggregate = root / manifest["aggregate_file"]
    assert hashlib.sha256(aggregate.read_bytes()).hexdigest() == manifest["aggregate_sha256"]
    with np.load(aggregate, allow_pickle=False) as data:
        assert set(data.files) == {"models", "pilot", "heldout"}
        assert data["pilot"].sum() == manifest["pilot_votes"]
        assert data["heldout"].sum() == manifest["heldout_votes"]
        assert len(data["models"]) == manifest["models"]
        assert data["pilot"].dtype.kind in "iu"
        assert data["heldout"].dtype.kind in "iu"
    assert manifest["license"] == "Apache-2.0"
