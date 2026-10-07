"""Keep cross-platform rounding tolerance separate from a changed input grid."""

from copy import deepcopy

import pytest
from export_fixture import check_fixture

REFERENCE = {
    "source": "src/eval_power/stats.py",
    "item_tolerance": 1,
    "power_absolute_tolerance": 1e-12,
    "variance_absolute_tolerance": 1e-12,
    "cases": [{"delta": 0.01, "items": 1000, "power_at_items": 0.8}],
    "comparisons": [{"pa": 0.7, "rho": 0.5, "paired_variance": 0.2}],
}


def test_cross_platform_rounding():
    rounded = deepcopy(REFERENCE)
    rounded["cases"][0]["power_at_items"] += 1e-13
    rounded["comparisons"][0]["paired_variance"] += 1e-13
    rounded["cases"][0]["items"] += 1
    check_fixture(REFERENCE, rounded)


@pytest.mark.parametrize(
    ("group", "key", "difference"),
    [
        ("cases", "delta", 1e-13),
        ("cases", "items", 2),
        ("cases", "power_at_items", 2e-12),
        ("comparisons", "paired_variance", 2e-12),
        ("comparisons", "rho", 2e-12),
    ],
)
def test_changed_values_are_rejected(group, key, difference):
    changed = deepcopy(REFERENCE)
    changed[group][0][key] += difference
    with pytest.raises(SystemExit, match="Fixture is stale"):
        check_fixture(REFERENCE, changed)


def test_changed_grid_is_rejected():
    changed = deepcopy(REFERENCE)
    changed["cases"].append(changed["cases"][0])
    with pytest.raises(SystemExit, match="grid changed"):
        check_fixture(REFERENCE, changed)
