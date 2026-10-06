"""Paired binary statistics and two-sided Gaussian sample-size approximations."""

import math
from numbers import Integral, Real

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from scipy.stats import binom


def _binary_pair(a, b, minimum=1):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("a and b must be aligned one-dimensional binary arrays")
    if a.size < minimum:
        raise ValueError(f"a and b must contain at least {minimum} items")
    for values in (a, b):
        if values.dtype.kind not in "buif" or not np.all((values == 0) | (values == 1)):
            raise ValueError("a and b must contain only binary values 0 and 1")
    # Convert before subtracting: uint8 subtraction would otherwise wrap around.
    return a.astype(np.float64, copy=False), b.astype(np.float64, copy=False)


def paired_variance(a, b) -> float:
    """Sample variance (ddof=1) of aligned binary differences A minus B."""
    a, b = _binary_pair(a, b, minimum=2)
    return float(np.var(a - b, ddof=1))


def unpaired_variance(a, b) -> float:
    """Plug-in binomial variance pA(1-pA)+pB(1-pB), without dividing by n."""
    a, b = _binary_pair(a, b)
    pa, pb = float(a.mean()), float(b.mean())
    return pa * (1 - pa) + pb * (1 - pb)


def paired_se(a, b) -> float:
    """IID-item standard error of the paired mean difference."""
    variance = paired_variance(a, b)
    return math.sqrt(variance / len(a))


def unpaired_se(a, b) -> float:
    """Standard error assuming independent model outcomes and IID items."""
    variance = unpaired_variance(a, b)
    return math.sqrt(variance / len(a))


def correlation(a, b) -> float | None:
    """Empirical Pearson correlation, or None when either marginal is constant."""
    a, b = _binary_pair(a, b)
    a, b = a - a.mean(), b - b.mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    if denominator == 0:
        return None
    return float(np.clip((a @ b) / denominator, -1, 1))


def clustered_se(a, b, groups) -> float:
    """CR1 intercept SE for the item-weighted difference, with >=2 clusters.

    This estimates uncertainty for the supplied cluster design, not an IID
    item-count planning variance. More items need not mean more clusters.
    """
    a, b = _binary_pair(a, b, minimum=2)
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.shape != a.shape:
        raise ValueError("groups must be one-dimensional and aligned with a and b")
    if groups.dtype.kind in "fc" and not np.all(np.isfinite(groups)):
        raise ValueError("group labels must be finite")
    try:
        labels, inverse = np.unique(groups, return_inverse=True)
    except TypeError as exc:
        raise ValueError("group labels must be comparable") from exc
    count = len(labels)
    if count < 2:
        raise ValueError("clustered_se requires at least two distinct groups")
    differences = a - b
    sums = np.bincount(inverse, weights=differences - differences.mean())
    variance = count / (count - 1) * float(sums @ sums) / a.size**2
    return math.sqrt(variance)


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _variance_alpha(variance, alpha):
    variance = _finite_real(variance, "variance")
    if variance <= 0:
        raise ValueError(
            "variance must be strictly positive; a zero-variance pilot cannot support "
            "sample-size planning"
        )
    alpha = _finite_real(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return variance, alpha


def _item_count(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0:
        raise ValueError("n must be a positive integer")
    return int(n)


def _normal_power(noncentrality, z):
    return float(ndtr(-z - noncentrality) + ndtr(noncentrality - z))


def gaussian_power(delta, variance, n, alpha=0.05) -> float:
    """Two-tailed normal-approximation power, not exact binary-test power."""
    delta = _finite_real(delta, "delta")
    if abs(delta) > 1:
        raise ValueError("delta must have absolute value at most 1")
    variance, alpha = _variance_alpha(variance, alpha)
    n = _item_count(n)
    noncentrality = abs(delta) * math.sqrt(n) / math.sqrt(variance)
    return _normal_power(noncentrality, -ndtri(alpha / 2))


def _target_noncentrality(alpha, power):
    power = _finite_real(power, "power")
    if not alpha < power < 1:
        raise ValueError("power must be greater than alpha and strictly less than 1")
    z = -ndtri(alpha / 2)
    upper = max(1.0, float(z + ndtri(power)))
    while _normal_power(upper, z) < power:
        upper *= 2
    return float(brentq(lambda value: _normal_power(value, z) - power, 0, upper, xtol=5e-15))


def required_items(delta, variance, alpha=0.05, power=0.8) -> int:
    """Minimal integer IID items per model for a fixed target effect and variance."""
    delta = _finite_real(delta, "delta")
    if not 0 < abs(delta) <= 1:
        raise ValueError("target delta must have absolute value in (0, 1]")
    variance, alpha = _variance_alpha(variance, alpha)
    noncentrality = _target_noncentrality(alpha, power)
    try:
        upper = max(1, math.ceil((noncentrality * math.sqrt(variance) / abs(delta)) ** 2))
    except (OverflowError, ValueError) as exc:
        raise ValueError("required item count exceeds floating-point planning range") from exc
    # Correct root/rounding error with monotone integer search, including n=1.
    while gaussian_power(delta, variance, upper, alpha) < power:
        upper *= 2
    lower = 0
    while upper - lower > 1:
        middle = (upper + lower) // 2
        if gaussian_power(delta, variance, middle, alpha) >= power:
            upper = middle
        else:
            lower = middle
    return upper


def minimum_detectable_difference(variance, n, alpha=0.05, power=0.8) -> float:
    """Fixed-variance Gaussian MDE; values above 1 indicate an impossible binary gap."""
    variance, alpha = _variance_alpha(variance, alpha)
    n = _item_count(n)
    noncentrality = _target_noncentrality(alpha, power)
    return noncentrality * math.sqrt(variance) / math.sqrt(n)


def mcnemar_pvalue(a, b) -> float:
    """Exact two-sided conditional McNemar p-value for aligned binary items."""
    a, b = _binary_pair(a, b)
    b_only = int(np.count_nonzero((a == 0) & (b == 1)))
    a_only = int(np.count_nonzero((a == 1) & (b == 0)))
    discordance = a_only + b_only
    if discordance == 0:
        return 1.0
    return min(1.0, float(2 * binom.cdf(min(a_only, b_only), discordance, 0.5)))


def holm_adjust(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values in original order; no independence needed."""
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("pvalues must be a one-dimensional array of finite values")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("pvalues must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(values.size, 0, -1)))
    result = np.empty_like(values)
    result[order] = adjusted
    return result
