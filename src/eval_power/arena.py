"""Aggregated Arena votes: unpenalized Bradley–Terry and IID-vote resampling.

A tie supplies half a win to each model, not two independent observations.
Scores use natural log odds, centered to sum zero. The conventional Elo scale
is 400/log(10) times this score. No prior or ridge rescues an unidentified fit.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import expit, ndtr
from scipy.stats import norm

ELO_SCALE = 400 / np.log(10)


@dataclass
class BTFit:
    status: str
    scores: np.ndarray | None = None
    covariance: np.ndarray | None = None
    iterations: int = 0


def _counts(counts, models):
    counts = np.asarray(counts)
    expected = (models * (models - 1) // 2, 3)
    if models < 2 or counts.shape != expected:
        raise ValueError("counts must cover every unordered pair and three outcomes")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be finite and nonnegative")
    if np.any(counts != np.floor(counts)):
        raise ValueError("counts must be integer vote counts")
    return counts.astype(np.float64, copy=False)


def fit_bt(counts, models, max_iterations=80):
    """Fit counts ordered by np.triu_indices(models, 1), columns A/B/tie.

    The directed win graph must be strongly connected for a finite MLE.
    Ties create both directed edges. Wald covariance is inverse curvature of
    the fractional-outcome objective, not an empirical sandwich estimate.
    """
    counts = _counts(counts, models)
    a, b = np.triu_indices(models, 1)
    n = counts.sum(axis=1)
    wins = counts[:, 0] + counts[:, 2] / 2
    rows = np.concatenate((a, b))
    cols = np.concatenate((b, a))
    edges = np.concatenate((wins > 0, n - wins > 0))
    graph = csr_matrix(
        (np.ones(int(edges.sum())), (rows[edges], cols[edges])), shape=(models, models)
    )
    if connected_components(graph, directed=False, return_labels=False) != 1:
        return BTFit("disconnected")
    if connected_components(graph, connection="strong", return_labels=False) != 1:
        return BTFit("separated")
    beta = np.zeros(models)

    def objective(value):
        delta = value[a] - value[b]
        return float(np.sum(n * np.logaddexp(0, delta) - wins * delta))

    def curvature(value):
        p = expit(value[a] - value[b])
        residual = n * p - wins
        gradient = np.bincount(a, residual, minlength=models)
        gradient -= np.bincount(b, residual, minlength=models)
        weight = n * p * (1 - p)
        hessian = np.diag(
            np.bincount(a, weight, minlength=models) + np.bincount(b, weight, minlength=models)
        )
        hessian[a, b] -= weight
        hessian[b, a] -= weight
        return gradient, hessian

    for iteration in range(max_iterations):
        gradient, hessian = curvature(beta)
        if np.max(np.abs(gradient)) < max(1e-7, float(n.sum()) * 1e-11):
            covariance = np.zeros((models, models))
            covariance[:-1, :-1] = solve(hessian[:-1, :-1], np.eye(models - 1), assume_a="pos")
            # Transform reference-model covariance to the sum-zero convention.
            centered = covariance - covariance.mean(axis=0)[None, :]
            centered -= centered.mean(axis=1)[:, None]
            return BTFit("ok", beta - beta.mean(), centered, iteration)
        try:
            step = solve(hessian[:-1, :-1], gradient[:-1], assume_a="pos")
        except np.linalg.LinAlgError:
            return BTFit("singular", iterations=iteration)
        fraction = 1.0
        old = objective(beta)
        descent = float(gradient[:-1] @ step)
        while fraction >= 2**-24:
            candidate = beta.copy()
            candidate[:-1] -= fraction * step
            if objective(candidate) <= old - 1e-4 * fraction * descent + 1e-10:
                beta = candidate
                break
            fraction /= 2
        else:
            return BTFit("line_search_failed", iterations=iteration)
    return BTFit("not_converged", iterations=max_iterations)


def contrast(fit, a, b):
    """Log-odds difference and curvature standard error for a fitted pair."""
    if fit.status != "ok":
        return None, None
    covariance = fit.covariance
    variance = covariance[a, a] + covariance[b, b] - 2 * covariance[a, b]
    return float(fit.scores[a] - fit.scores[b]), float(np.sqrt(max(0, variance)))


def resample_counts(counts, votes, rng):
    """Exact IID vote bootstrap without expanding any individual vote rows."""
    counts = np.asarray(counts)
    total = counts.sum()
    if total <= 0 or votes < 1:
        raise ValueError("resampling requires positive source and requested vote counts")
    return rng.multinomial(int(votes), counts.ravel() / total).reshape(counts.shape)


def split_counts(counts, rng, pilot_fraction=0.5):
    """Disjoint random vote allocation; pilot + heldout is exactly the source."""
    counts = np.asarray(counts, dtype=np.int64)
    if not 0 < pilot_fraction < 1:
        raise ValueError("pilot_fraction must lie strictly between zero and one")
    pilot = rng.binomial(counts, pilot_fraction)
    return pilot, counts - pilot


def planned_votes(gap, se, pilot_votes, alpha=0.05, power=0.8):
    """Total arena votes at the pilot matchup mix, using inverse-N curvature.

    This plans for a two-sided normal contrast test, not direct A/B votes.
    A zero estimated gap has no finite plan; an unidentified fit has no plan.
    """
    if gap is None or se is None:
        return None, "nonestimable"
    if not np.isfinite(gap) or not np.isfinite(se) or se <= 0 or pilot_votes < 1:
        return None, "nonestimable"
    if abs(gap) <= 1e-12:
        return None, "zero_gap"
    if not 0 < alpha < power < 1:
        raise ValueError("require 0 < alpha < power < 1")
    z = norm.ppf(1 - alpha / 2)
    root = brentq(lambda x: ndtr(-z - x) + ndtr(x - z) - power, 0, 20)
    required = pilot_votes * (root * se / abs(gap)) ** 2
    if not np.isfinite(required) or required > np.iinfo(np.int64).max:
        return None, "out_of_range"
    return max(1, int(np.ceil(required))), "ok"
