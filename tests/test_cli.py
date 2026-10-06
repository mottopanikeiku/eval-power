"""CLI planning stays separate from observed-pilot effect diagnostics."""

import json

import numpy as np
import pytest

from eval_power import clustered_se, correlation, paired_variance, required_items, unpaired_variance
from eval_power.cli import main


@pytest.fixture
def pilot_path(tmp_path):
    path = tmp_path / "pilot.npy"
    # Nonzero variances but zero observed mean difference.
    np.save(path, np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]], dtype=np.uint8))
    return path


def test_direct_variance_json(capsys):
    assert main(["--difference", ".01", "--variance", ".15"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["items_per_model"] == required_items(0.01, 0.15)
    assert result["difference"] == 0.01
    assert result["variance"] == 0.15
    assert result["alpha"] == 0.05 and result["target_power"] == 0.8
    assumptions = " ".join(result["assumptions"])
    assert "Gaussian approximation" in assumptions
    assert "not exact binary-test power" in assumptions
    assert "transfers" in assumptions and "items per model" in assumptions
    assert "IID" in assumptions


def test_pilot_reports_both_counts_for_explicit_target(pilot_path, capsys):
    assert main(["--difference", ".05", "--pilot", str(pilot_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    pilot = np.load(pilot_path)
    a, b = pilot[:, 0], pilot[:, 1]
    assert float(a.mean() - b.mean()) == 0
    assert result["difference"] == 0.05
    assert result["pilot_n"] == len(a)
    assert result["paired_variance"] == pytest.approx(paired_variance(a, b))
    assert result["unpaired_variance"] == pytest.approx(unpaired_variance(a, b))
    assert result["paired_items_per_model"] == required_items(0.05, paired_variance(a, b))
    assert result["unpaired_items_per_model"] == required_items(0.05, unpaired_variance(a, b))
    assert result["paired_correlation"] == pytest.approx(correlation(a, b))
    assert "items_per_model" not in result
    assert "observed_power" not in result
    assert any("not estimated from the pilot gap" in value for value in result["assumptions"])


def test_groups_report_cr1_without_cluster_item_count(pilot_path, tmp_path, capsys):
    group_path = tmp_path / "subjects.npy"
    groups = np.array(["anatomy", "anatomy", "math", "history", "history", "history"])
    np.save(group_path, groups)
    assert main(
        ["--difference", ".05", "--pilot", str(pilot_path), "--groups", str(group_path)]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    pilot = np.load(pilot_path)
    assert result["group_count"] == 3
    assert result["cluster_se"] == pytest.approx(clustered_se(pilot[:, 0], pilot[:, 1], groups))
    assert "cluster_items_per_model" not in result
    assert any("do not extrapolate cluster sampling" in value for value in result["assumptions"])


def test_pilot_constant_marginal_reports_null_correlation(tmp_path, capsys):
    path = tmp_path / "constant.npy"
    np.save(path, [[0, 0], [0, 1], [0, 1], [0, 0]])
    main(["--difference", ".1", "--pilot", str(path)])
    assert json.loads(capsys.readouterr().out)["paired_correlation"] is None


def test_cli_custom_target_and_probabilities(capsys):
    main(["--difference", "-.2", "--variance", ".3", "--alpha", ".1", "--power", ".9"])
    result = json.loads(capsys.readouterr().out)
    assert result["difference"] == -0.2
    assert result["items_per_model"] == required_items(-0.2, 0.3, 0.1, 0.9)


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--variance", ".15"],
        ["--difference", ".01"],
        ["--difference", "zero", "--variance", ".15"],
        ["--difference", "0", "--variance", ".15"],
        ["--difference", "1.1", "--variance", ".15"],
        ["--difference", "nan", "--variance", ".15"],
        ["--difference", ".01", "--variance", "0"],
        ["--difference", ".01", "--variance", "inf"],
        ["--difference", ".01", "--variance", ".15", "--alpha", "0"],
        ["--difference", ".01", "--variance", ".15", "--power", "1"],
        ["--difference", ".01", "--variance", ".15", "--power", ".01"],
        ["--difference", ".01", "--variance", ".15", "--groups", "unused.npy"],
        ["--difference", ".01", "--variance", ".15", "--pilot", "unused.npy"],
    ],
)
def test_cli_invalid_arguments_are_errors_without_tracebacks(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert "error:" in output.err
    assert "Traceback" not in output.err
    assert output.out == ""


def test_pilot_never_defaults_to_observed_difference(pilot_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--pilot", str(pilot_path)])
    assert exc.value.code == 2
    assert "--difference" in capsys.readouterr().err


@pytest.mark.parametrize(
    "pilot",
    [
        [0, 1],
        [[0, 1, 0], [1, 0, 1]],
        [[0, 2], [1, 0]],
        [[0, np.nan], [1, 0]],
        [[0, 1]],
        [[0, 0], [1, 1]],
    ],
)
def test_cli_refuses_invalid_or_zero_variance_pilots(pilot, tmp_path, capsys):
    path = tmp_path / "invalid.npy"
    np.save(path, np.asarray(pilot))
    with pytest.raises(SystemExit) as exc:
        main(["--difference", ".01", "--pilot", str(path)])
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert "error:" in output.err and "Traceback" not in output.err
    assert output.out == ""


@pytest.mark.parametrize("groups", [["only"] * 6, ["a", "b"], [["a"]] * 6])
def test_cli_invalid_groups(pilot_path, tmp_path, groups, capsys):
    path = tmp_path / "groups.npy"
    np.save(path, np.asarray(groups))
    with pytest.raises(SystemExit) as exc:
        main(["--difference", ".01", "--pilot", str(pilot_path), "--groups", str(path)])
    assert exc.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("contents", [None, b"not a numpy array"])
def test_cli_missing_or_malformed_pilot(tmp_path, contents, capsys):
    path = tmp_path / "missing.npy"
    if contents is not None:
        path.write_bytes(contents)
    with pytest.raises(SystemExit) as exc:
        main(["--difference", ".01", "--pilot", str(path)])
    assert exc.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
