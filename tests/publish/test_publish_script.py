"""Tests for scripts/publish_model.sh with fake curl and uvx on PATH: no network, no huggingface_hub."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from nnslr_tools.bundle import REQUIRED_FILES
from publish_helpers import make_bundle, write_fake_tools


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "publish_model.sh"
DEFAULT_REPO = "cristianku/sunnypilot-speedlimits-models"


class Fakes:
    def __init__(self, tmp_path: Path) -> None:
        self.data_root = tmp_path / "data root"
        self.bin_dir = tmp_path / "bin"
        self.curl_log = tmp_path / "curl.log"
        self.uvx_log = tmp_path / "uvx.log"
        write_fake_tools(self.bin_dir)

    def run(self, *args: str, curl_code: str = "404", curl_body: Path | None = None,
            hf_repo: str | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.pop("HF_REPO", None)
        if hf_repo is not None:
            env["HF_REPO"] = hf_repo
        env["PATH"] = str(self.bin_dir) + os.pathsep + os.environ["PATH"]
        env["NNSLR_DATA_ROOT"] = str(self.data_root)
        env["FAKE_CURL_LOG"] = str(self.curl_log)
        env["FAKE_CURL_CODE"] = curl_code
        env["FAKE_UVX_LOG"] = str(self.uvx_log)
        if curl_body is not None:
            env["FAKE_CURL_BODY"] = str(curl_body)
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

    def uvx_args(self) -> list[str]:
        return self.uvx_log.read_text().splitlines()


def _run_bundle(fakes: Fakes, run_id: str) -> Path:
    return make_bundle(fakes.data_root / "runs" / run_id / "bundle")


def test_help_prints_the_top_comment_without_side_effects(tmp_path) -> None:
    fakes = Fakes(tmp_path)

    for flag in ("--help", "-h"):
        proc = fakes.run(flag)
        assert proc.returncode == 0, proc.stderr
        assert "Usage: scripts/publish_model.sh" in proc.stdout
        assert "--no-upload" in proc.stdout
        assert "HF_REPO" in proc.stdout
        assert "hf auth login" in proc.stdout
        assert "set -euo pipefail" not in proc.stdout

    assert not fakes.data_root.exists()
    assert not fakes.curl_log.exists()
    assert not fakes.uvx_log.exists()


def test_no_upload_dry_run_stages_bundle_and_catalog(tmp_path) -> None:
    fakes = Fakes(tmp_path)
    bundle = _run_bundle(fakes, "baseline-001")

    proc = fakes.run("--no-upload")

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "--no-upload: niente caricato" in proc.stdout
    stage = fakes.data_root / "publish" / "baseline-001"
    model_dir = stage / "models" / "baseline-001"
    staged = []
    for path in sorted(model_dir.iterdir()):
        staged.append(path.name)
        assert path.read_bytes() == (bundle / path.name).read_bytes()
    assert staged == sorted(REQUIRED_FILES)

    catalog = json.loads((stage / "speed_models.json").read_text())
    entry = catalog["models"][0]
    assert entry["name"] == "baseline-001"
    assert entry["run_id"] == "baseline-001"
    base = f"https://huggingface.co/{DEFAULT_REPO}/resolve/main/"
    for item in entry["files"]:
        assert item["url"] == base + item["path"]

    assert f"https://huggingface.co/{DEFAULT_REPO}/resolve/main/speed_models.json" in fakes.curl_log.read_text()
    assert not fakes.uvx_log.exists()


def test_default_bundle_is_the_newest_run(tmp_path) -> None:
    fakes = Fakes(tmp_path)
    older = _run_bundle(fakes, "run-old")
    _run_bundle(fakes, "run-new")
    os.utime(older, (1_000_000_000, 1_000_000_000))

    proc = fakes.run("--no-upload")

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (fakes.data_root / "publish" / "run-new" / "models" / "run-new").is_dir()
    assert not (fakes.data_root / "publish" / "run-old").exists()


def test_upload_makes_one_commit_with_the_stage(tmp_path) -> None:
    fakes = Fakes(tmp_path)
    bundle = _run_bundle(fakes, "baseline-001")

    proc = fakes.run(str(bundle), "--name=prova")

    assert proc.returncode == 0, proc.stderr + proc.stdout
    stage = fakes.data_root / "publish" / "prova"
    assert fakes.uvx_args() == [
        "--from", "huggingface_hub", "hf", "upload",
        DEFAULT_REPO, str(stage), ".",
        "--commit-message", "Publish model prova",
    ]
    catalog = json.loads((stage / "speed_models.json").read_text())
    assert catalog["models"][0]["name"] == "prova"
    assert catalog["models"][0]["run_id"] == "baseline-001"
    assert "Pubblicato" in proc.stdout


def test_hf_repo_env_changes_catalog_and_upload_target(tmp_path) -> None:
    fakes = Fakes(tmp_path)
    _run_bundle(fakes, "baseline-001")

    proc = fakes.run("--name", "prova", hf_repo="someone/other-models")

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "https://huggingface.co/someone/other-models/resolve/main/speed_models.json" in fakes.curl_log.read_text()
    assert fakes.uvx_args()[4] == "someone/other-models"
    catalog = json.loads((fakes.data_root / "publish" / "prova" / "speed_models.json").read_text())
    assert catalog["models"][0]["files"][0]["url"].startswith(
        "https://huggingface.co/someone/other-models/resolve/main/models/prova/"
    )


def test_existing_catalog_is_downloaded_and_updated(tmp_path) -> None:
    fakes = Fakes(tmp_path)
    _run_bundle(fakes, "baseline-001")
    remote = tmp_path / "remote_speed_models.json"
    older = {"name": "older", "run_id": "old", "files": []}
    remote.write_text(json.dumps({"schema_version": 1, "models": [older]}))

    proc = fakes.run("--no-upload", curl_code="200", curl_body=remote)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    catalog = json.loads((fakes.data_root / "publish" / "baseline-001" / "speed_models.json").read_text())
    assert catalog["models"][0] == older
    assert catalog["models"][1]["name"] == "baseline-001"


def test_http_error_on_catalog_stops_before_upload(tmp_path) -> None:
    fakes = Fakes(tmp_path)
    _run_bundle(fakes, "baseline-001")

    for code in ("500", "401", "000"):
        proc = fakes.run(curl_code=code)
        assert proc.returncode != 0
        assert f"HTTP {code}" in proc.stderr

    assert not fakes.uvx_log.exists()


def test_broken_bundle_is_refused(tmp_path) -> None:
    fakes = Fakes(tmp_path)
    bundle = _run_bundle(fakes, "baseline-001")
    with (bundle / "reader.onnx").open("ab") as handle:
        handle.write(b"x")

    proc = fakes.run()

    assert proc.returncode != 0
    assert "bundle_hash_mismatch" in proc.stderr
    assert "bundle non valido" in proc.stderr
    assert not (fakes.data_root / "publish").exists()
    assert not fakes.curl_log.exists()
    assert not fakes.uvx_log.exists()


def test_invalid_name_is_refused(tmp_path) -> None:
    fakes = Fakes(tmp_path)
    _run_bundle(fakes, "baseline-001")

    for name in ("..", "bad/name", ".hidden"):
        proc = fakes.run("--no-upload", "--name=" + name)
        assert proc.returncode != 0
        assert "nome non valido" in proc.stderr

    assert not (fakes.data_root / "publish").exists()
    assert not fakes.curl_log.exists()


def test_missing_bundle_is_reported(tmp_path) -> None:
    fakes = Fakes(tmp_path)

    proc = fakes.run("--no-upload")

    assert proc.returncode != 0
    assert "runs/*/bundle" in proc.stderr
    assert not fakes.curl_log.exists()
