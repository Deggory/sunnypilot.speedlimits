"""wandb tracking of training runs, without wandb installed.

Every test uses a fake ``wandb`` module in ``sys.modules`` (or ``None`` there,
which makes ``import wandb`` raise ImportError), so the real package is never
needed and nothing ever reaches wandb.ai.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

from nnslr_tools import tracking

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


class FakeRun:
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.finished = []

    def log(self, metrics, step):
        self.logged.append((step, dict(metrics)))

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


class FakeImage:
    def __init__(self, data, caption=None, boxes=None):
        self.data = data
        self.caption = caption
        self.boxes = boxes


def fake_wandb(monkeypatch, *, init_error=None):
    module = types.ModuleType("wandb")
    module.init_calls = []
    module.runs = []

    def init(**kwargs):
        module.init_calls.append(kwargs)
        if init_error is not None:
            raise init_error
        run = FakeRun()
        module.runs.append(run)
        return run

    module.init = init
    module.Image = FakeImage
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def logged_in_environ():
    return {"WANDB_API_KEY": "test-key-not-real"}


def test_init_uses_default_project_group_and_job_type(tmp_path, monkeypatch, capsys):
    wandb = fake_wandb(monkeypatch)
    output = tmp_path / "runs" / "baseline-001" / "detector"
    output.mkdir(parents=True)

    tracker = tracking.start_training_run(
        "detector", output, {"epochs": 3}, environ=logged_in_environ(), home=tmp_path,
    )

    assert tracker.enabled
    assert len(wandb.init_calls) == 1
    kwargs = wandb.init_calls[0]
    assert kwargs["project"] == "sunnypilot.speedlimits"
    assert kwargs["group"] == "baseline-001"
    assert kwargs["job_type"] == "detector"
    assert kwargs["name"] == "baseline-001-detector"
    assert kwargs["dir"] == str(output)
    assert kwargs["config"] == {"epochs": 3}
    # Logged in: wandb decides the mode by itself.
    assert "mode" not in kwargs
    assert capsys.readouterr().err == ""


def test_project_and_run_id_come_from_environment(tmp_path, monkeypatch):
    wandb = fake_wandb(monkeypatch)
    output = tmp_path / "anywhere" / "reader"
    output.mkdir(parents=True)
    environ = logged_in_environ()
    environ["WANDB_PROJECT"] = "my-project"
    environ["NNSLR_RUN_ID"] = "night-run-7"

    tracking.start_training_run("reader", output, {}, environ=environ, home=tmp_path)

    kwargs = wandb.init_calls[0]
    assert kwargs["project"] == "my-project"
    assert kwargs["group"] == "night-run-7"
    assert kwargs["job_type"] == "reader"
    assert kwargs["name"] == "night-run-7-reader"


def test_detector_and_reader_of_one_run_share_the_group(tmp_path, monkeypatch):
    wandb = fake_wandb(monkeypatch)
    run_root = tmp_path / "runs" / "baseline-002"
    for task in ("detector", "reader"):
        (run_root / task).mkdir(parents=True)
        tracking.start_training_run(task, run_root / task, {}, environ=logged_in_environ(), home=tmp_path)

    groups = [call["group"] for call in wandb.init_calls]
    job_types = [call["job_type"] for call in wandb.init_calls]
    assert groups == ["baseline-002", "baseline-002"]
    assert job_types == ["detector", "reader"]


def test_training_run_id_layouts(tmp_path):
    assert tracking.training_run_id(tmp_path / "runs" / "b-1" / "reader", {}) == "b-1"
    assert tracking.training_run_id(tmp_path / "runs" / "reader-baseline-001", {}) == "reader-baseline-001"
    assert tracking.training_run_id(tmp_path / "out" / "x", {"NNSLR_RUN_ID": "explicit"}) == "explicit"


def test_no_login_falls_back_to_offline(tmp_path, monkeypatch, capsys):
    wandb = fake_wandb(monkeypatch)
    output = tmp_path / "out"
    output.mkdir()

    tracker = tracking.start_training_run("reader", output, {}, environ={}, home=tmp_path)

    assert tracker.enabled
    assert wandb.init_calls[0]["mode"] == "offline"
    err = capsys.readouterr().err
    assert "offline" in err
    assert err.count("\n") == 1


def test_offline_fallback_decision():
    home = Path("/nonexistent-home-for-test")
    assert tracking.wandb_mode({}, home) == "offline"
    assert tracking.wandb_mode({"WANDB_API_KEY": "k"}, home) is None
    assert tracking.wandb_mode({"WANDB_MODE": "offline"}, home) is None
    assert tracking.wandb_mode({"WANDB_MODE": "online"}, home) is None
    assert tracking.wandb_mode({"WANDB_MODE": "disabled"}, home) == "disabled"
    assert tracking.wandb_mode({"WANDB_MODE": "Disabled", "WANDB_API_KEY": "k"}, home) == "disabled"


def test_netrc_login_means_online(tmp_path, monkeypatch, capsys):
    wandb = fake_wandb(monkeypatch)
    (tmp_path / ".netrc").write_text("machine api.wandb.ai\n  login user\n  password secret\n")
    output = tmp_path / "out"
    output.mkdir()

    assert tracking.wandb_mode({}, tmp_path) is None
    tracking.start_training_run("reader", output, {}, environ={}, home=tmp_path)

    assert "mode" not in wandb.init_calls[0]
    err = capsys.readouterr().err
    assert err == ""
    assert "secret" not in err


def test_netrc_without_wandb_is_not_a_login(tmp_path):
    (tmp_path / ".netrc").write_text("machine github.com\n  login user\n  password x\n")
    assert tracking.wandb_mode({}, tmp_path) == "offline"


def test_disabled_mode_never_imports_wandb(tmp_path, monkeypatch, capsys):
    # None in sys.modules makes "import wandb" fail: a disabled run must not try.
    monkeypatch.setitem(sys.modules, "wandb", None)
    output = tmp_path / "out"
    output.mkdir()

    tracker = tracking.start_training_run(
        "detector", output, {}, environ={"WANDB_MODE": "disabled"}, home=tmp_path,
    )

    assert not tracker.enabled
    assert capsys.readouterr().err == ""
    tracker.log({"train/loss": 1.0}, step=1)
    assert tracker.image(object(), "caption") is None
    assert tracker.should_log_images(1, 10) is False
    tracker.set_summary({"best": 1.0})
    tracker.finish(exit_code=0)
    assert not (output / "wandb").exists()


def test_missing_wandb_warns_once_and_does_not_crash(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "wandb", None)
    output = tmp_path / "out"
    output.mkdir()

    tracker = tracking.start_training_run("reader", output, {}, environ={}, home=tmp_path)

    assert not tracker.enabled
    err = capsys.readouterr().err
    assert "not installed" in err
    assert err.count("\n") == 1
    tracker.log({"train/loss": 1.0}, step=1)
    tracker.finish(exit_code=1)


def test_failing_init_warns_and_disables(tmp_path, monkeypatch, capsys):
    fake_wandb(monkeypatch, init_error=RuntimeError("network down"))
    output = tmp_path / "out"
    output.mkdir()

    tracker = tracking.start_training_run(
        "reader", output, {}, environ=logged_in_environ(), home=tmp_path,
    )

    assert not tracker.enabled
    err = capsys.readouterr().err
    assert "init failed" in err
    assert "test-key-not-real" not in err


def test_non_main_process_does_not_log(tmp_path, monkeypatch):
    wandb = fake_wandb(monkeypatch)
    output = tmp_path / "out"
    output.mkdir()
    environ = logged_in_environ()
    environ["RANK"] = "1"

    tracker = tracking.start_training_run("reader", output, {}, environ=environ, home=tmp_path)

    assert not tracker.enabled
    assert wandb.init_calls == []


def test_tracker_logs_images_summary_and_finishes_once(tmp_path, monkeypatch):
    wandb = fake_wandb(monkeypatch)
    output = tmp_path / "out"
    output.mkdir()
    tracker = tracking.start_training_run("reader", output, {}, environ=logged_in_environ(), home=tmp_path)
    run = wandb.runs[0]

    tracker.log({"train/loss": 0.5}, step=3)
    image = tracker.image("pixels", "true a / pred b")
    boxed = tracker.image("pixels", "frame", boxes={"predictions": {"box_data": []}})
    tracker.set_summary({"best_validation_accuracy": 0.75})
    tracker.finish(exit_code=0)
    tracker.finish(exit_code=1)

    assert run.logged == [(3, {"train/loss": 0.5})]
    assert isinstance(image, FakeImage)
    assert image.caption == "true a / pred b"
    assert image.boxes is None
    assert boxed.boxes == {"predictions": {"box_data": []}}
    assert run.summary == {"best_validation_accuracy": 0.75}
    assert run.finished == [0]
    assert not tracker.enabled


def test_image_schedule(tmp_path, monkeypatch):
    fake_wandb(monkeypatch)
    output = tmp_path / "out"
    output.mkdir()
    tracker = tracking.start_training_run("reader", output, {}, environ=logged_in_environ(), home=tmp_path)

    chosen = []
    for epoch in range(1, 13):
        if tracker.should_log_images(epoch, 12):
            chosen.append(epoch)
    assert chosen == [1, 5, 10, 12]


def test_detector_box_data_relative_and_filtered():
    boxes = [[10, 20, 30, 40], [0, 0, 100, 50], [50, 25, 200, 80]]
    predicted = tracking.detector_box_data(boxes, 100, 50, scores=[0.9, 0.1, 0.3])
    reviewed = tracking.detector_box_data(boxes[:1], 100, 50)

    assert len(predicted) == 2
    assert predicted[0]["position"] == {"minX": 0.1, "maxX": 0.3, "minY": 0.4, "maxY": 0.8}
    assert predicted[0]["class_id"] == 1
    assert predicted[0]["scores"] == {"score": 0.9}
    # Clipped to the image.
    assert predicted[1]["position"]["maxX"] == 1.0
    assert predicted[1]["position"]["maxY"] == 1.0
    assert reviewed[0]["box_caption"] == "reviewed"
    assert "scores" not in reviewed[0]
    assert tracking.detector_box_data(boxes, 0, 50) == []


def test_detector_box_data_keeps_at_most_max_boxes():
    boxes = []
    scores = []
    for index in range(tracking.DETECTOR_IMAGE_MAX_BOXES + 5):
        boxes.append([index, index, index + 1, index + 1])
        scores.append(0.9)
    predicted = tracking.detector_box_data(boxes, 100, 100, scores=scores)
    assert len(predicted) == tracking.DETECTOR_IMAGE_MAX_BOXES


def test_cli_and_training_import_without_wandb_or_huggingface_hub():
    probe = (
        "import sys\n"
        "import nnslr_tools.cli\n"
        "import nnslr_tools.training\n"
        "import nnslr_tools.tracking\n"
        "hits = []\n"
        "for name in sys.modules:\n"
        "    for banned in ('wandb', 'huggingface_hub'):\n"
        "        if name == banned or name.startswith(banned + '.'):\n"
        "            hits.append(name)\n"
        "if hits:\n"
        "    print('HITS:' + ','.join(sorted(hits)))\n"
        "    sys.exit(1)\n"
        "print('ISOLATION_OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ISOLATION_OK" in proc.stdout
