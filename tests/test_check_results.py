"""Cross-platform float rounding must pass; any other change to saved results must fail."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
assert_matches = importlib.import_module("check_results").assert_matches

REFERENCE = {"pairs": [{"pvalue": 0.0049637111773446095, "n_items": 32, "detected": True}]}


def test_last_bit_float_differences_pass():
    assert_matches(
        {"pairs": [{"pvalue": 0.004963711177344609, "n_items": 32, "detected": True}]}, REFERENCE
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"pairs": [{"pvalue": 0.00496372, "n_items": 32, "detected": True}]},
        {"pairs": [{"pvalue": 0.0049637111773446095, "n_items": 33, "detected": True}]},
        {"pairs": [{"pvalue": 0.0049637111773446095, "n_items": 32.0, "detected": True}]},
        {"pairs": [{"pvalue": 0.0049637111773446095, "n_items": 32, "detected": 1}]},
        {"pairs": [{"pvalue": 0.0049637111773446095, "n_items": 32}]},
        {"pairs": []},
    ],
)
def test_other_changes_fail(changed):
    with pytest.raises(AssertionError):
        assert_matches(changed, REFERENCE)
