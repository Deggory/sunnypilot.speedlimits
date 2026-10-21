"""Tests for nnslr_tools.model_catalog: add or replace one model in speed_models.json."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import pytest

from nnslr_tools import model_catalog
from nnslr_tools.bundle import REQUIRED_FILES
from publish_helpers import make_bundle


BASE_URL = "https://huggingface.co/cristianku/sunnypilot-speedlimits-models/resolve/main"


def _stage_model(tmp_path: Path, name: str, detector_bytes: bytes = b"detector-onnx") -> Path:
    bundle = make_bundle(tmp_path / ("bundle-" + name), detector_bytes)
    model_dir = tmp_path / "stage" / "models" / name
    model_dir.mkdir(parents=True)
    for file_name in REQUIRED_FILES:
        shutil.copy2(bundle / file_name, model_dir / file_name)
    return model_dir


def _run(catalog: Path, name: str, model_dir: Path, run_id: str = "run-1") -> int:
    return model_catalog.main(
        [
            str(catalog),
            "--name", name,
            "--run-id", run_id,
            "--model-dir", str(model_dir),
            "--base-url", BASE_URL,
        ]
    )


def _names(catalog: Path) -> list[str]:
    names = []
    for entry in json.loads(catalog.read_text())["models"]:
        names.append(entry["name"])
    return names


def test_new_catalog_is_created(tmp_path) -> None:
    model_dir = _stage_model(tmp_path, "baseline-001")
    catalog = tmp_path / "stage" / "speed_models.json"

    assert _run(catalog, "baseline-001", model_dir, run_id="baseline-001") == 0

    payload = json.loads(catalog.read_text())
    assert list(payload) == ["schema_version", "models"]
    assert payload["schema_version"] == 1
    entry = payload["models"][0]
    assert list(entry) == ["name", "display_date", "run_id", "created", "bundle", "files"]
    assert entry["name"] == "baseline-001"
    assert entry["run_id"] == "baseline-001"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", entry["created"])
    assert re.fullmatch(r"[A-Z][a-z]+ \d{1,2}, \d{4}", entry["display_date"])
    assert entry["bundle"]["schema_version"] == 1
    assert entry["bundle"]["feature_id"] == "nnslr-speed-vision"
    assert entry["bundle"]["dataset_sha256"] == "a" * 64
    assert entry["bundle"]["split_sha256"] == "b" * 64
    assert entry["bundle"]["advisory_only"] is True
    # Pretty JSON with a final newline.
    assert catalog.read_text().startswith("{\n  ")
    assert catalog.read_text().endswith("}\n")


def test_files_have_correct_size_sha256_and_url(tmp_path) -> None:
    model_dir = _stage_model(tmp_path, "baseline-001")
    catalog = tmp_path / "speed_models.json"
    _run(catalog, "baseline-001", model_dir)

    entry = json.loads(catalog.read_text())["models"][0]
    paths = []
    for item in entry["files"]:
        paths.append(item["path"])
        assert list(item) == ["path", "size", "sha256", "url"]
        local = model_dir / Path(item["path"]).name
        assert item["size"] == local.stat().st_size
        assert item["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()
        assert item["url"] == BASE_URL + "/" + item["path"]

    expected = []
    for file_name in sorted(REQUIRED_FILES):
        expected.append("models/baseline-001/" + file_name)
    assert paths == expected
    manifest = model_dir / "sha256sums.json"
    assert entry["bundle"]["digest"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_hashes_are_taken_from_the_bundle_manifest(tmp_path) -> None:
    model_dir = _stage_model(tmp_path, "baseline-001")
    manifest_path = model_dir / "sha256sums.json"
    manifest = json.loads(manifest_path.read_text())
    # A recognizable value proves the manifest is read instead of hashing the file again.
    manifest["files"]["reader.onnx"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n")
    catalog = tmp_path / "speed_models.json"
    _run(catalog, "baseline-001", model_dir)

    entry = json.loads(catalog.read_text())["models"][0]
    for item in entry["files"]:
        if item["path"].endswith("/reader.onnx"):
            assert item["sha256"] == "f" * 64
        if item["path"].endswith("/sha256sums.json"):
            assert item["sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_second_model_is_appended(tmp_path) -> None:
    catalog = tmp_path / "speed_models.json"
    _run(catalog, "first", _stage_model(tmp_path / "a", "first"))
    _run(catalog, "second", _stage_model(tmp_path / "b", "second"))

    assert _names(catalog) == ["first", "second"]


def test_same_name_is_replaced_in_place(tmp_path, capsys) -> None:
    catalog = tmp_path / "speed_models.json"
    _run(catalog, "first", _stage_model(tmp_path / "a", "first"))
    _run(catalog, "second", _stage_model(tmp_path / "b", "second"))
    _run(catalog, "third", _stage_model(tmp_path / "c", "third"))
    before = json.loads(catalog.read_text())["models"]

    new_dir = _stage_model(tmp_path / "d", "second", detector_bytes=b"detector-onnx-v2")
    _run(catalog, "second", new_dir, run_id="run-2")

    after = json.loads(catalog.read_text())["models"]
    assert _names(catalog) == ["first", "second", "third"]
    assert after[0] == before[0]
    assert after[2] == before[2]
    assert after[1]["run_id"] == "run-2"
    assert after[1]["bundle"]["digest"] != before[1]["bundle"]["digest"]
    assert "catalogo: sostituito second" in capsys.readouterr().out


def test_existing_catalog_from_another_writer_keeps_its_entries(tmp_path) -> None:
    catalog = tmp_path / "speed_models.json"
    other = {"name": "older", "run_id": "old", "files": []}
    catalog.write_text(json.dumps({"schema_version": 1, "models": [other]}))

    _run(catalog, "baseline-001", _stage_model(tmp_path, "baseline-001"))

    models = json.loads(catalog.read_text())["models"]
    assert models[0] == other
    assert models[1]["name"] == "baseline-001"


@pytest.mark.parametrize("name", ["bad/name", "..", ".", ".hidden", "-x", "a b", "", "nome$"])
def test_invalid_names_are_rejected(tmp_path, name) -> None:
    model_dir = _stage_model(tmp_path, "baseline-001")
    catalog = tmp_path / "speed_models.json"

    with pytest.raises(SystemExit) as error:
        _run(catalog, name, model_dir)

    assert error.value.code != 0
    assert not catalog.exists()


def test_model_dir_must_match_the_name(tmp_path) -> None:
    model_dir = _stage_model(tmp_path, "baseline-001")
    catalog = tmp_path / "speed_models.json"

    with pytest.raises(SystemExit, match="non si chiama"):
        _run(catalog, "other-name", model_dir)
    assert not catalog.exists()


def test_catalog_with_unknown_schema_is_refused(tmp_path) -> None:
    catalog = tmp_path / "speed_models.json"
    catalog.write_text(json.dumps({"schema_version": 99, "models": []}))

    with pytest.raises(SystemExit, match="schema_version"):
        _run(catalog, "baseline-001", _stage_model(tmp_path, "baseline-001"))
    assert json.loads(catalog.read_text())["schema_version"] == 99


def test_catalog_that_is_not_json_is_refused(tmp_path) -> None:
    catalog = tmp_path / "speed_models.json"
    catalog.write_text("<html>error page</html>")

    with pytest.raises(SystemExit, match="JSON non valido"):
        _run(catalog, "baseline-001", _stage_model(tmp_path, "baseline-001"))
    assert catalog.read_text() == "<html>error page</html>"


def test_module_imports_only_the_standard_library() -> None:
    source = Path(model_catalog.__file__).read_text(encoding="utf-8")
    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module.split(".")[0])

    assert imported
    for module_name in imported:
        assert module_name in sys.stdlib_module_names or module_name == "__future__", module_name
