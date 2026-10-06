"""Importer safety, alignment and numerical round-trip checks (no network required)."""

import hashlib
import importlib.util
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

IMPORTER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "import_data.py"
SPEC = importlib.util.spec_from_file_location("import_data", IMPORTER_PATH)
import_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(import_data)


@pytest.fixture
def source(monkeypatch):
    """Small real-shaped blocks; all 57 subject names still participate."""
    monkeypatch.setattr(import_data, "MODEL_COUNT", 3)
    monkeypatch.setattr(
        import_data,
        "EXPECTED_ITEMS",
        {"gsm8k": 2, "winogrande": 2, "arc": 2, "hellaswag": 2, "mmlu": 57},
    )
    blocks = {}
    for benchmark, tasks in import_data.TASKS.items():
        for index, task in enumerate(tasks):
            values = np.array([[0, 1, index % 2]], dtype=float)
            if benchmark != "mmlu":
                values = np.concatenate((values, 1 - values), axis=0)
            blocks[task] = {"correctness": values}
    # This genuinely nonbinary block must not be rounded, thresholded or loaded.
    blocks["harness_truthfulqa_mc_0"] = {"correctness": np.array([[np.nan, 0.3, 0.7]])}
    return {
        "models": ["details_author__A", "details_author__B", "details_author__C"],
        "data": dict(reversed(list(blocks.items()))),
    }


def payload_hashes(payload):
    return {
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "git_blob": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
    }


def pin_payload(monkeypatch, payload):
    hashes = payload_hashes(payload)
    monkeypatch.setattr(
        import_data,
        "verify_source",
        lambda stream: original_verify_source(stream, **hashes),
    )


original_verify_source = import_data.verify_source


@pytest.mark.parametrize("protocol", [4, 5])
def test_numpy_only_pickle_round_trip(protocol):
    original = {"models": ["model/A"], "array": np.array([[0.0, 1.0]]), "scalar": np.float64(0.5)}
    payload = pickle.dumps(original, protocol=protocol)
    actual = import_data.NumpyOnlyUnpickler(io.BytesIO(payload)).load()
    assert actual["models"] == original["models"]
    np.testing.assert_array_equal(actual["array"], original["array"])
    assert actual["scalar"] == original["scalar"]


def test_unsafe_pickle_global_rejected():
    # A GLOBAL/REDUCE invocation would evaluate this string in an ordinary loader.
    payload = b"cbuiltins\neval\n(V1 + 1\ntR."
    with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle global"):
        import_data.NumpyOnlyUnpickler(io.BytesIO(payload)).load()


def test_persistent_pickle_reference_rejected():
    with pytest.raises(pickle.UnpicklingError):
        import_data.NumpyOnlyUnpickler(io.BytesIO(b"Pexternal-resource\n.")).load()


def test_checksums_and_stream_rewind():
    payload = b"a known array artifact"
    stream = io.BytesIO(payload)
    import_data.verify_source(stream, **payload_hashes(payload))
    assert stream.tell() == 0
    assert stream.read() == payload


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("size", 999, "size mismatch"),
        ("sha256", "0" * 64, "SHA256 mismatch"),
        ("git_blob", "0" * 40, "Git blob hash mismatch"),
    ],
)
def test_checksum_failures(field, replacement, message):
    payload = b"a known array artifact"
    hashes = payload_hashes(payload)
    hashes[field] = replacement
    with pytest.raises(ValueError, match=message):
        import_data.verify_source(io.BytesIO(payload), **hashes)


def test_checksum_checked_before_pickle(tmp_path, monkeypatch):
    path = tmp_path / "unsafe.pickle"
    path.write_bytes(b"cbuiltins\neval\n(V1 + 1\ntR.")

    def never_load(*args, **kwargs):
        pytest.fail("Unpickler must not be constructed before checksum validation")

    monkeypatch.setattr(import_data, "NumpyOnlyUnpickler", never_load)
    with pytest.raises(ValueError, match="size mismatch"):
        import_data.load_source(path)


def test_pinned_unsafe_pickle_still_rejected(tmp_path, monkeypatch):
    payload = b"cbuiltins\neval\n(V1 + 1\ntR."
    pin_payload(monkeypatch, payload)
    path = tmp_path / "unsafe.pickle"
    path.write_bytes(payload)
    with pytest.raises(pickle.UnpicklingError, match="Forbidden pickle global"):
        import_data.load_source(path)


def test_cache_hit_never_downloads(tmp_path, monkeypatch):
    path = tmp_path / "cached.pickle"
    path.write_bytes(b"cache")

    def no_network(*args, **kwargs):
        pytest.fail("Existing cache must be used without a network request")

    monkeypatch.setattr(import_data.urllib.request, "urlopen", no_network)
    import_data.ensure_cached(path)


def test_cache_downloads_only_pinned_artifact(tmp_path, monkeypatch, source):
    payload = pickle.dumps(source, protocol=5)
    pin_payload(monkeypatch, payload)
    monkeypatch.setattr(import_data, "SOURCE_BYTES", len(payload))
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(import_data.urllib.request, "urlopen", download)
    path = tmp_path / "cache" / "lb.pickle"
    import_data.ensure_cached(path)
    assert path.read_bytes() == payload
    assert calls == [(import_data.SOURCE_URL, 60)]
    matrices = import_data.build_matrices(import_data.load_source(path))
    assert list(matrices) == list(import_data.TASKS)


def test_bad_download_is_not_cached(tmp_path, monkeypatch):
    payload = b"invalid source"
    monkeypatch.setattr(
        import_data.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload)
    )
    path = tmp_path / "lb.pickle"
    with pytest.raises(ValueError, match="size mismatch"):
        import_data.ensure_cached(path)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_orientation_order_ids_and_subjects(source):
    matrices = import_data.build_matrices(source)
    assert list(matrices) == ["gsm8k", "winogrande", "arc", "hellaswag", "mmlu"]
    for benchmark, arrays in matrices.items():
        assert arrays["correctness"].shape[1] == 3
        assert arrays["correctness"].dtype == np.uint8
        assert arrays["models"].tolist() == source["models"]
        assert len(set(arrays["item_ids"].tolist())) == len(arrays["item_ids"])
        assert all(arrays[key].dtype.kind == "U" for key in ("models", "item_ids", "subjects"))
        if benchmark != "mmlu":
            np.testing.assert_array_equal(arrays["subjects"], arrays["item_ids"])
    mmlu = matrices["mmlu"]
    assert mmlu["subjects"].tolist() == list(import_data.MMLU_SUBJECTS)
    for index, task in enumerate(import_data.TASKS["mmlu"]):
        assert mmlu["item_ids"][index] == f"{task}:0"
        np.testing.assert_array_equal(
            mmlu["correctness"][index], source["data"][task]["correctness"][0]
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.25, -1, 2])
def test_nonfinite_and_nonbinary_rejected(source, value):
    source["data"]["harness_gsm8k_5"]["correctness"][0, 0] = value
    with pytest.raises(ValueError, match="Nonfinite|Nonbinary"):
        import_data.build_matrices(source)


@pytest.mark.parametrize("kind", ["duplicate", "empty", "wrong_count"])
def test_model_names_rejected(source, kind):
    if kind == "duplicate":
        source["models"][1] = source["models"][0]
    elif kind == "empty":
        source["models"][1] = ""
    else:
        source["models"].pop()
    with pytest.raises(ValueError, match="model labels|Model labels"):
        import_data.build_matrices(source)


def test_column_alignment_rejected(source):
    source["data"]["harness_gsm8k_5"]["models"] = list(reversed(source["models"]))
    with pytest.raises(ValueError, match="Model alignment mismatch"):
        import_data.build_matrices(source)


def test_column_alignment_accepted(source):
    source["data"]["harness_gsm8k_5"]["models"] = list(source["models"])
    import_data.build_matrices(source)


def test_transposed_matrix_rejected(source):
    block = source["data"]["harness_gsm8k_5"]
    block["correctness"] = block["correctness"].T
    with pytest.raises(ValueError, match="item x 3"):
        import_data.build_matrices(source)


def test_item_count_rejected(source):
    block = source["data"]["harness_gsm8k_5"]
    block["correctness"] = block["correctness"][:1]
    with pytest.raises(ValueError, match="Unexpected gsm8k item count"):
        import_data.build_matrices(source)


def test_missing_subject_rejected(source):
    del source["data"][import_data.TASKS["mmlu"][0]]
    with pytest.raises(ValueError, match="Missing task block"):
        import_data.build_matrices(source)


def test_object_correctness_rejected(source):
    block = source["data"]["harness_gsm8k_5"]
    block["correctness"] = block["correctness"].astype(object)
    with pytest.raises(ValueError, match="real numeric"):
        import_data.build_matrices(source)


def test_npz_and_manifest_round_trip(source, tmp_path):
    matrices = import_data.build_matrices(source)
    manifest = import_data.write_outputs(matrices, tmp_path)
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == on_disk
    assert manifest["source"]["model_ids"] == source["models"]
    assert manifest["source"]["original_hf_per_model_revisions"] is None
    assert "LICENSE METADATA" in manifest["licenses"]["reference_revision_scope"]
    assert manifest["licenses"]["arc"]["license"] == "CC-BY-SA-4.0"
    assert manifest["licenses"]["winogrande"]["license"] == "CC-BY"
    for benchmark, expected in matrices.items():
        path = tmp_path / f"{benchmark}.npz"
        assert (
            manifest["outputs"][benchmark]["sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
        with np.load(path, allow_pickle=False) as arrays:
            assert set(arrays.files) == {"correctness", "models", "item_ids", "subjects"}
            for key, values in expected.items():
                np.testing.assert_array_equal(arrays[key], values)


@pytest.mark.parametrize("benchmark", list(import_data.TASKS))
def test_committed_real_matrix_contract(benchmark):
    directory = IMPORTER_PATH.parent.parent / "data"
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{benchmark}.npz"
    assert import_data.file_sha256(path) == manifest["outputs"][benchmark]["sha256"]
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["correctness"].shape == (import_data.EXPECTED_ITEMS[benchmark], 395)
        assert arrays["correctness"].dtype == np.uint8
        assert ((arrays["correctness"] == 0) | (arrays["correctness"] == 1)).all()
        assert len(set(arrays["models"].tolist())) == 395
        assert arrays["models"].tolist() == manifest["source"]["model_ids"]
        assert len(set(arrays["item_ids"].tolist())) == import_data.EXPECTED_ITEMS[benchmark]
        for key in ("models", "item_ids", "subjects"):
            assert arrays[key].dtype.kind == "U"
        if benchmark == "mmlu":
            assert set(arrays["subjects"].tolist()) == set(import_data.MMLU_SUBJECTS)
        else:
            np.testing.assert_array_equal(arrays["subjects"], arrays["item_ids"])


def test_pinned_license_notice():
    directory = IMPORTER_PATH.parent.parent / "data"
    manifest = json.loads((directory / "manifest.json").read_text())
    notice = manifest["licenses"]["tinyBenchmarks"]
    assert (
        import_data.file_sha256(directory / notice["local_notice"])
        == (notice["local_notice_sha256"])
    )
    assert manifest["source"]["sha256"] == import_data.SOURCE_SHA256
    assert manifest["source"]["git_blob_sha1"] == import_data.SOURCE_GIT_BLOB
