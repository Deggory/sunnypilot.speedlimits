"""wandb run tracking for NNSLR training jobs.

Copied from the way openpilot.distill tracks its supervised training
(supervised/train.py + helpers/train_supervised.sh):

- the project comes from WANDB_PROJECT, default "sunnypilot.speedlimits";
- the run files go next to the training output (<output>/wandb/);
- without WANDB_MODE, WANDB_API_KEY or a login in ~/.netrc the run goes
  offline, so a fresh machine never stops at a login prompt;
- only the main process logs (RANK unset or 0).

Detector and reader of the same pipeline run share one wandb group (the run
id), with job_type "detector" / "reader".

This module is stdlib-only at import time: wandb is imported only when a real
training run starts. WANDB_MODE=disabled skips wandb completely, and a missing
wandb package prints one warning and training continues without tracking.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

DEFAULT_PROJECT = "sunnypilot.speedlimits"

# Sample images are logged on the first epoch, every IMAGE_EVERY_EPOCHS epochs
# and on the last epoch.
IMAGE_EVERY_EPOCHS = 5
DETECTOR_IMAGE_SAMPLES = 4
READER_IMAGE_SAMPLES = 16
# Full frames are shrunk to this longest side before upload; boxes are logged
# in relative coordinates, so they stay aligned with the shrunk image.
IMAGE_MAX_SIDE = 960
# Same default score threshold as evaluation.evaluate_detector.
DETECTOR_IMAGE_MIN_SCORE = 0.25
DETECTOR_IMAGE_MAX_BOXES = 10
# Detector class index 1 is the only foreground class ("speed_sign").
DETECTOR_CLASS_LABELS = {1: "speed_sign"}


def _warn(message: str) -> None:
    # stderr: stdout of "nnslr train" is the JSON result.
    print(f"wandb: {message}", file=sys.stderr)


def _netrc_has_wandb_login(netrc_path: Path) -> bool:
    """True when "wandb login" saved a key for api.wandb.ai in this netrc."""
    try:
        text = netrc_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "api.wandb.ai" in text


def wandb_mode(environ: dict[str, str], home: Path) -> str | None:
    """Decide the wandb mode like distill's helpers/train_supervised.sh.

    Returns:
      "disabled" when WANDB_MODE=disabled: wandb is not even imported;
      "offline"  when there is no login at all (no WANDB_MODE, no
                 WANDB_API_KEY, no api.wandb.ai in ~/.netrc);
      None       in every other case: wandb reads WANDB_MODE / the key itself.
    """
    mode = environ.get("WANDB_MODE", "").strip()
    if mode != "":
        if mode.lower() == "disabled":
            return "disabled"
        return None
    if environ.get("WANDB_API_KEY", "").strip() != "":
        return None
    if _netrc_has_wandb_login(home / ".netrc"):
        return None
    return "offline"


def training_run_id(output_dir: Path, environ: dict[str, str]) -> str:
    """Return the pipeline run id used as wandb group.

    NNSLR_RUN_ID (exported by scripts/run_v100_baseline.sh) wins. Otherwise the
    layout <data-root>/runs/<run id>/<task> gives the run id from the parent
    folder, and any other output folder is its own run id.
    """
    explicit = environ.get("NNSLR_RUN_ID", "").strip()
    if explicit != "":
        return explicit
    output_dir = Path(output_dir).absolute()
    if output_dir.parent.parent.name == "runs":
        return output_dir.parent.name
    return output_dir.name


def detector_box_data(
    boxes_xyxy: list[list[float]],
    width: int,
    height: int,
    scores: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Convert pixel xyxy boxes to wandb box_data in relative coordinates.

    With scores (predictions) only boxes at or above DETECTOR_IMAGE_MIN_SCORE
    are kept, at most DETECTOR_IMAGE_MAX_BOXES. Without scores (reviewed
    ground truth) every box is kept.
    """
    result = []
    if width <= 0 or height <= 0:
        return result
    for index in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[index]
        item = {
            "position": {
                "minX": max(0.0, min(1.0, float(x1) / width)),
                "maxX": max(0.0, min(1.0, float(x2) / width)),
                "minY": max(0.0, min(1.0, float(y1) / height)),
                "maxY": max(0.0, min(1.0, float(y2) / height)),
            },
            "class_id": 1,
        }
        if scores is not None:
            score = float(scores[index])
            if score < DETECTOR_IMAGE_MIN_SCORE:
                continue
            item["scores"] = {"score": score}
            item["box_caption"] = f"{score:.2f}"
        else:
            item["box_caption"] = "reviewed"
        result.append(item)
        if scores is not None and len(result) >= DETECTOR_IMAGE_MAX_BOXES:
            break
    return result


class TrainingTracker:
    """Thin wrapper around one wandb run. Every method is a no-op when disabled."""

    def __init__(self, wandb_module: Any = None, run: Any = None) -> None:
        self.wandb = wandb_module
        self.run = run

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self.run is None:
            return
        self.run.log(metrics, step=step)

    def should_log_images(self, epoch_number: int, total_epochs: int) -> bool:
        """True on the first epoch, every IMAGE_EVERY_EPOCHS and on the last one."""
        if self.run is None:
            return False
        if epoch_number == 1:
            return True
        if epoch_number == total_epochs:
            return True
        if epoch_number % IMAGE_EVERY_EPOCHS == 0:
            return True
        return False

    def image(self, image: Any, caption: str, boxes: dict[str, Any] | None = None) -> Any:
        """Build a wandb.Image (PIL image in), or None when disabled."""
        if self.run is None:
            return None
        if boxes is None:
            return self.wandb.Image(image, caption=caption)
        return self.wandb.Image(image, caption=caption, boxes=boxes)

    def set_summary(self, values: dict[str, Any]) -> None:
        if self.run is None:
            return
        for key in values:
            self.run.summary[key] = values[key]

    def finish(self, exit_code: int = 0) -> None:
        """Close the run (exit_code 1 marks it failed). Safe to call twice."""
        if self.run is None:
            return
        run = self.run
        self.run = None
        run.finish(exit_code=exit_code)


def start_training_run(
    task: str,
    output_dir: Path,
    config: dict[str, Any],
    *,
    environ: dict[str, str] | None = None,
    home: Path | None = None,
) -> TrainingTracker:
    """Start the wandb run of one training job, or return a disabled tracker.

    output_dir must already exist: wandb writes its files in <output_dir>/wandb.
    Training never fails because of wandb: a missing package or a failing
    wandb.init prints one warning and returns a disabled tracker.
    """
    if environ is None:
        environ = dict(os.environ)
    if home is None:
        home = Path.home()

    # Only the main process logs, like distill (torchrun sets RANK).
    rank = environ.get("RANK", "0").strip()
    if rank != "" and rank != "0":
        return TrainingTracker()

    mode = wandb_mode(environ, home)
    if mode == "disabled":
        return TrainingTracker()

    try:
        import wandb
    except ImportError:
        _warn(
            "not installed, training continues without wandb logging "
            "(GPU env: pip install -e '.[training]')"
        )
        return TrainingTracker()
    except Exception as exc:
        _warn(f"import failed ({type(exc).__name__}), training continues without wandb logging")
        return TrainingTracker()

    if mode == "offline":
        _warn(
            "not logged in (no WANDB_API_KEY, no api.wandb.ai in ~/.netrc), running offline; "
            "the run stays in " + str(Path(output_dir) / "wandb")
            + ", upload it later with: wandb sync <that folder>/offline-run-*"
        )

    project = environ.get("WANDB_PROJECT", "").strip()
    if project == "":
        project = DEFAULT_PROJECT
    run_id = training_run_id(output_dir, environ)

    init_arguments = {
        "project": project,
        "name": f"{run_id}-{task}",
        "group": run_id,
        "job_type": task,
        "dir": str(output_dir),
        "config": config,
    }
    if mode == "offline":
        init_arguments["mode"] = "offline"

    try:
        run = wandb.init(**init_arguments)
    except Exception as exc:
        _warn(f"init failed ({type(exc).__name__}: {exc}), training continues without wandb logging")
        return TrainingTracker()
    return TrainingTracker(wandb, run)
