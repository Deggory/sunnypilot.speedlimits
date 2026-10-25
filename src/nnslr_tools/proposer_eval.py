"""The proposers of Prepare measured side by side (nnslr evaluate-proposer, the Proposers view of the Evaluate tab).

Each proposer (preannotate.PROPOSERS: the Vietnamese YOLO, our cutter and classifier, Qwen3-VL) runs exactly as in
Prepare, with its own worker, on the reviewed frames of one partition (test by default: the roads no training saw).
Its proposals are matched one to one to the boxes reviewers kept (IoU >= 0.5, nnslr evaluate's greedy matching):

- speed signs (training.DETECTOR_FAMILIES): how many it found, and found AND read right (the same family and the
  same number), overall, by width of the sign and on the frames a person reviewed;
- every sign (speed signs and the other traffic signs): how many it found;
- false proposals per frame: the proposals that match no kept box;
- seconds per frame, the model's loading included.

Painted numbers on the road (road_marking) are left out on both sides: only the Vietnamese worker looks for them.
Nothing is written into the reviews or the routes: the proposals go to a temporary folder, the report to
<data root>/evaluations/proposers/<proposer>-<partition>.json (the latest, shown by the site) and a copy named
by the time.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from nnslr_tools.chain import match_boxes
from nnslr_tools.preannotate import PROPOSERS, proposer_setup, run_proposer_worker
from nnslr_tools.training import DETECTOR_FAMILIES, load_reviewed_dataset, reviewed_frames

IOU_THRESHOLD = 0.5
WIDTHS = (("lt12", "< 12 px", 0.0, 12.0), ("12-20", "12-20 px", 12.0, 20.0), ("20-40", "20-40 px", 20.0, 40.0),
          ("ge40", "≥ 40 px", 40.0, float("inf")))
# Left out on both sides (see the module).
SKIPPED_FAMILIES = {"road_marking", "road_marking_candidate", "not_a_sign"}


def reports_dir(root: Path) -> Path:
    return Path(root) / "evaluations" / "proposers"


def _counts() -> dict[str, int]:
    return {"truth": 0, "found": 0, "read_right": 0}


def _shares(counts: dict[str, int]) -> dict[str, Any]:
    truth = counts["truth"]
    return {**counts, "found_share": counts["found"] / truth if truth else None,
            "read_right_share": counts["read_right"] / truth if truth else None}


def read_right(proposal: dict[str, Any], truth: dict[str, Any]) -> bool:
    """The same family and the same number (both without a number for the families that have none)."""
    return proposal.get("family") == truth.get("sign_family") and proposal.get("value_kph") == truth.get("value_kph")


def tally(frames: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The counts of the module docstring, from the reviewed frames and the proposals of the same frames."""
    speed, every, human = _counts(), {"truth": 0, "found": 0}, _counts()
    widths = {key: _counts() for key, *_ in WIDTHS}
    proposals = false = 0
    for frame, row in zip(frames, rows):
        truth = [kept for kept in frame["kept"] if isinstance(kept.get("bbox_xyxy"), list)
                 and kept.get("sign_family") not in SKIPPED_FAMILIES]
        proposed = [item for item in row["detections"] if item.get("family") not in SKIPPED_FAMILIES]
        proposals += len(proposed)
        pairs = match_boxes([{"box": item["bbox_xyxy"], "score": item.get("score") or 0.0} for item in proposed],
                            [{"box": kept["bbox_xyxy"]} for kept in truth], IOU_THRESHOLD)
        false += len(proposed) - len(pairs)
        found = {index: proposed[position] for position, index in pairs.items()}
        person = frame.get("label_basis") == "human_review"
        for index, kept in enumerate(truth):
            every["truth"] += 1
            every["found"] += index in found
            if kept.get("sign_family") not in DETECTOR_FAMILIES:
                continue
            right = index in found and read_right(found[index], kept)
            width = float(kept["bbox_xyxy"][2]) - float(kept["bbox_xyxy"][0])
            bucket = next(key for key, _, low, high in WIDTHS if low <= width < high)
            for counts in (speed, widths[bucket], *([human] if person else [])):
                counts["truth"] += 1
                counts["found"] += index in found
                counts["read_right"] += right
    return {"speed_signs": _shares(speed), "every_sign": {**every, "found_share": every["found"] / every["truth"]
                                                          if every["truth"] else None},
            "speed_signs_person": _shares(human),
            "by_width": [{"key": key, "label": label, **_shares(widths[key])} for key, label, *_ in WIDTHS],
            "proposals": proposals, "false_proposals": false,
            "proposals_per_frame": proposals / len(frames) if frames else None,
            "false_per_frame": false / len(frames) if frames else None}


def evaluate_proposer(proposer: str, data_root: Path, dataset_path: Path, split_path: Path, *,
                      partition: str = "test", dataset_kind: str = "gold", limit: int | None = None) -> dict[str, Any]:
    """Runs a proposer on the reviewed frames of a partition and writes its report (see the module)."""
    if proposer not in PROPOSERS:
        raise ValueError("unknown proposer: " + ", ".join(PROPOSERS))
    root = Path(data_root)
    loaded = load_reviewed_dataset(dataset_path, split_path, root, dataset_kind=dataset_kind)
    frames = sorted((frame for frame in reviewed_frames(loaded["rows"], loaded["assignments"])
                     if frame["split"] == partition),
                    key=lambda frame: (frame["route_id"], frame["segment_index"], frame["output_index"]))
    if limit:
        frames = frames[:limit]
    if not frames:
        raise ValueError(f"no reviewed frame in {partition}")
    work = [{"frame_key": f"{frame['segment_index']}/{frame['output_index']}", "route_id": frame["route_id"],
             "segment_index": frame["segment_index"], "output_index": frame["output_index"],
             "image_path": frame["image_path"], "frame_sha256": frame["image_sha256"]} for frame in frames]
    setup = proposer_setup(proposer, root)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="proposer-") as folder:
        rows = run_proposer_worker(setup, root, work, Path(folder), "evaluate-" + proposer, "several")
    seconds = time.perf_counter() - started
    report = {"schema_version": 1, "task": "proposer", "proposer": proposer, "label": PROPOSERS[proposer],
              "partition": partition, "model": {key: value for key, value in setup["model"].items() if key != "path"},
              "confidence": setup["confidence"], "dataset_sha256": loaded["dataset_sha256"],
              "split_sha256": loaded["split_sha256"], "frames": len(frames),
              "frames_person": sum(frame.get("label_basis") == "human_review" for frame in frames),
              "limited": bool(limit), "iou_threshold": IOU_THRESHOLD, "seconds": round(seconds, 1),
              "seconds_per_frame": round(seconds / len(frames), 3),
              "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **tally(frames, rows)}
    folder = reports_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for name in (f"{proposer}-{partition}.json", f"{stamp}-{proposer}-{partition}.json"):
        temporary = folder / ("." + name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, folder / name)
    return report


def latest_reports(root: Path) -> list[dict[str, Any]]:
    """The latest report of each proposer and partition, for the site."""
    found = []
    for path in sorted(reports_dir(root).glob("*-*.json")):
        if path.name[0].isdigit():
            continue
        try:
            found.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return found
