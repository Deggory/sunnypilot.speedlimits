"""The signs of the Mapillary Traffic Sign Dataset (MTSD) as extra training crops of the classifier.

MTSD (https://www.mapillary.com/dataset/trafficsign, CC BY-NC-SA 4.0) is about 52,000 street photos from all over the
world with every traffic sign boxed and named (``regulatory--maximum-speed-limit-50--g1``). Its signs that are ours
are cut once (``nnslr mapillary-crops``, helpers/mapillary.sh) into small crops with a margin around the sign, read
straight from the zips of the download (on the server they sit on the TrueNAS share /data/mapillary):

    <data root>/external/mapillary/crops/<object key>.jpg
    <data root>/external/mapillary/crops.jsonl    a header line, then one line per crop: path, box, our label

The classifier's training takes them when asked (``nnslr train --task reader --classes v2 --mapillary
<crops.jsonl>``, scripts/run_classifier.sh with NNSLR_CLASSIFIER_MAPILLARY=1, the Mapillary switch of Train ›
Classifier): in train only, never in validation or test, so every measure stays on our own drives. Only the labels
of MAPPED_LABELS and SPEED_LABEL come in: a speed sign of the European round design, its end, and the kinds of
other signs our classifier already has as classes (a kind we fold into other_sign gets no Mapillary crops: they
would make a class of photos that are not ours). At most PER_LABEL crops of each label, so that Mapillary helps the
rare classes without burying the common ones.

MTSD's photos are not the comma's road camera: other lenses, mostly daylight, often closer. The crops are scaled so
that the sign is at most MAX_SIGN_SIDE pixels, the size of a near sign in a comma frame.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
import zipfile
import zlib

from nnslr_tools.annotations import atomic_write

CROPS_KIND = "mapillary_crops"
CROPS_VERSION = 1
# Where helpers/mapillary.sh writes the crops, under the data root.
CROPS_DIR = Path("external") / "mapillary" / "crops"
CROPS_FILE = Path("external") / "mapillary" / "crops.jsonl"
# The label basis of a Mapillary crop in a training plan (a label of MTSD's annotators, not of our reviews).
LABEL_BASIS = "mapillary"
# At most this many crops of one label in a training plan (the first by a hash of the object key: always the same).
PER_LABEL = 300
# A sign smaller than this (its longer side, in MTSD pixels) is left out: too few pixels to read.
MIN_SIGN_SIDE = 16
# A crop is scaled down until its sign is at most this many pixels on its longer side.
MAX_SIGN_SIDE = 96
# The margin cut around the sign, as a share of its side: room for the box jitter of the training (at most 1.2 x
# the side and 0.1 of it off centre, nnslr_tools.training.CLASSIFIER_BOX_JITTER).
MARGIN = 0.5
# MTSD's flags of a sign that make it a poor example: a sign MTSD itself could not name (ambiguous), a fake one
# (dummy: a sign on an advert), cut by the photo's edge, on the back of something (exterior), inside another sign.
SKIPPED_PROPERTIES = ("ambiguous", "dummy", "out-of-frame", "exterior", "included")

# The European round speed sign (design g1) of MTSD; its other designs are American and Canadian rectangles, in
# miles per hour some of them.
SPEED_LABEL = re.compile(r"^regulatory--maximum-speed-limit-(\d+)--g1$")
SPEED_VALUES = frozenset({10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130})
# MTSD's labels that are signs of ours, each checked on its crops: the European design only.
MAPPED_LABELS = {
    "regulatory--end-of-maximum-speed-limit-30--g2": "cancellation",
    "regulatory--yield--g1": "other_sign:give_way",
    "regulatory--stop--g1": "other_sign:stop",
    "regulatory--no-entry--g1": "other_sign:no_entry",
    "regulatory--priority-road--g4": "other_sign:main_road",
    "regulatory--end-of-priority-road--g1": "other_sign:main_road_end",
    "regulatory--roundabout--g1": "other_sign:roundabout",
    "regulatory--height-limit--g1": "other_sign:height_limit",
    "regulatory--weight-limit--g1": "other_sign:weight_limit",
    "regulatory--go-straight--g1": "other_sign:ahead_only",
    "information--pedestrians-crossing--g1": "other_sign:pedestrian_crossing",
    "information--motorway--g1": "other_sign:motorway_start",
    "information--end-of-motorway--g1": "other_sign:motorway_end",
    "information--limited-access-road--g1": "other_sign:expressway_start",
    "information--end-of-limited-access-road--g1": "other_sign:expressway_end",
    "information--end-of-built-up-area--g1": "other_sign:town_exit",
    "warning--other-danger--g1": "other_sign:danger_other",
    "warning--roadworks--g1": "other_sign:roadworks",
}


def our_label(mtsd_label: str) -> str | None:
    """Our classifier label of an MTSD label, or None when it is not one of ours (or not its European design)."""
    match = SPEED_LABEL.match(mtsd_label)
    if match:
        value = int(match.group(1))
        return f"maximum_speed:{value}" if value in SPEED_VALUES else None
    return MAPPED_LABELS.get(mtsd_label)


def _region(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """The pixels cut around a sign: its box and MARGIN of its side on every side, inside the photo."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1)
    pad = MARGIN * side
    return (max(0, int(x1 - pad)), max(0, int(y1 - pad)), min(width, int(x2 + pad + 1)), min(height, int(y2 + pad + 1)))


def _usable(item: dict[str, Any], ispano: bool) -> tuple[list[float] | None, str | None]:
    """The box of an MTSD sign worth a crop, or None and why not."""
    properties = item.get("properties") or {}
    for name in SKIPPED_PROPERTIES:
        if properties.get(name):
            return None, name
    bbox = item.get("bbox") or {}
    if "cross_boundary" in bbox or ispano:
        return None, "panorama"
    try:
        box = [float(bbox["xmin"]), float(bbox["ymin"]), float(bbox["xmax"]), float(bbox["ymax"])]
    except (KeyError, TypeError, ValueError):
        return None, "no_box"
    if box[2] <= box[0] or box[3] <= box[1]:
        return None, "no_box"
    if max(box[2] - box[0], box[3] - box[1]) < MIN_SIGN_SIDE:
        return None, "too_small"
    return box, None


class _Files:
    """The ``<key>.json`` and ``<key>.jpg`` files under some folders, loose or inside the .zip archives of MTSD's
    download (read from the zip without unzipping it: on a network share a hundred thousand small files cost far
    more than reading each photo once). A zip that does not open (still being copied) is left out and named; a
    hidden file (the copy in flight, .part) and one that failed its md5 check (UNVERIFIED-*) are not looked at."""

    SUFFIXES = (".json", ".jpg")

    def __init__(self, folders: list[Path], log) -> None:
        self.where: dict[str, dict[str, tuple[Path | None, str]]] = {suffix: {} for suffix in self.SUFFIXES}
        self.broken: list[str] = []
        self._zips: dict[Path, zipfile.ZipFile] = {}
        for folder in dict.fromkeys(folders):
            for path in sorted(folder.rglob("*")):
                suffix = path.suffix.lower()
                if path.name.startswith((".", "UNVERIFIED-")) or not path.is_file():
                    continue
                if suffix in self.where:
                    self.where[suffix].setdefault(path.stem, (None, str(path)))
                elif suffix == ".zip" and path not in self._zips:
                    try:
                        archive = zipfile.ZipFile(path)
                    except (OSError, zipfile.BadZipFile) as error:
                        log(f"mapillary: {path} left out, it does not open ({error})")
                        self.broken.append(str(path))
                        continue
                    self._zips[path] = archive
                    for name in archive.namelist():
                        member = PurePosixPath(name)
                        if member.suffix.lower() in self.where and not member.name.startswith("._"):
                            self.where[member.suffix.lower()].setdefault(member.stem, (path, name))

    def keys(self, suffix: str) -> list[str]:
        return sorted(self.where[suffix])

    def has(self, key: str, suffix: str) -> bool:
        return key in self.where[suffix]

    def read(self, key: str, suffix: str) -> bytes:
        archive, name = self.where[suffix][key]
        return Path(name).read_bytes() if archive is None else self._zips[archive].read(name)

    def close(self) -> None:
        for archive in self._zips.values():
            archive.close()


def make_crops(annotations: Path, images: Path, data_root: Path, *, log=print) -> dict[str, Any]:
    """Cut the MTSD signs that are ours (our_label) from the photos (``<key>.jpg``) under ``images`` named by the
    annotations (``<key>.json``) under ``annotations`` into CROPS_DIR, and write CROPS_FILE. Both are folders
    searched with their subfolders, loose files or MTSD's .zip archives as downloaded (they may be the same
    folder). A photo whose annotation has no sign of ours is never opened; one without its photo is counted, not
    an error (the annotations come in one download, the photos in several). Run again, it resumes: a photo whose
    crops are all cut already is not read again. Returns the header written."""
    from PIL import Image

    annotations, images, data_root = Path(annotations), Path(images), Path(data_root)
    if not annotations.is_dir():
        raise ValueError(f"mapillary_missing: no annotations folder {annotations}")
    if not images.is_dir():
        raise ValueError(f"mapillary_missing: no photos folder {images}")
    out_dir = data_root / CROPS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    reused = 0
    files = _Files([annotations, images], log)
    try:
        keys = files.keys(".json")
        for number, stem in enumerate(keys, 1):
            if number % 2000 == 0:
                log(f"mapillary: {number}/{len(keys)} annotations, {len(rows)} crops")
            try:
                document = json.loads(files.read(stem, ".json"))
            except (OSError, ValueError, zipfile.BadZipFile, zlib.error):
                skipped["unreadable_annotation"] += 1
                continue
            if not isinstance(document, dict) or "objects" not in document:
                continue  # not an annotation (a README, a split list)
            wanted = []
            for item in document.get("objects") or []:
                label = our_label(str(item.get("label", "")))
                if label is None:
                    skipped["not_ours"] += 1
                    continue
                box, why = _usable(item, bool(document.get("ispano")))
                if box is None:
                    skipped[why] += 1
                    continue
                key = str(item.get("key") or hashlib.sha1(f"{stem}{box}".encode()).hexdigest()[:22])
                wanted.append((item, label, box, key))
            if not wanted:
                continue
            if not files.has(stem, ".jpg"):
                skipped["photo_not_downloaded"] += len(wanted)
                continue
            width, height = document.get("width"), document.get("height")
            image = None
            if not (isinstance(width, int) and isinstance(height, int)
                    and all((out_dir / (key + ".jpg")).is_file() for *_, key in wanted)):
                try:
                    with Image.open(io.BytesIO(files.read(stem, ".jpg"))) as source:
                        image = source.convert("RGB")
                except (OSError, zipfile.BadZipFile, zlib.error):
                    skipped["photo_unreadable"] += len(wanted)
                    continue
                width, height = image.width, image.height
            else:
                reused += len(wanted)
            for item, label, box, key in wanted:
                left, top, right, bottom = _region(box, width, height)
                scale = min(1.0, MAX_SIGN_SIDE / max(box[2] - box[0], box[3] - box[1]))
                name = CROPS_DIR / (key + ".jpg")
                if image is not None:
                    crop = image.crop((left, top, right, bottom))
                    if scale < 1.0:
                        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                                           Image.LANCZOS)
                    crop.save(data_root / name, quality=92)
                rows.append({
                    "image_path": name.as_posix(),
                    "bbox_xyxy": [round((box[0] - left) * scale, 2), round((box[1] - top) * scale, 2),
                                  round((box[2] - left) * scale, 2), round((box[3] - top) * scale, 2)],
                    "label": label, "mtsd_label": item.get("label"), "mtsd_image": stem, "mtsd_object": key,
                    "occluded": bool((item.get("properties") or {}).get("occluded")),
                })
        broken = list(files.broken)
    finally:
        files.close()
    if reused:
        log(f"mapillary: {reused} crops already cut, kept")
    header = {"kind": CROPS_KIND, "version": CROPS_VERSION, "annotations": str(annotations), "images": str(images),
              "crops": len(rows), "by_label": dict(sorted(Counter(row["label"] for row in rows).items())),
              "skipped": dict(sorted(skipped.items())), "license": "CC BY-NC-SA 4.0, Mapillary Traffic Sign Dataset"}
    if broken:
        header["zips_left_out"] = broken
    lines = [json.dumps(header, sort_keys=True)] + [json.dumps(row, sort_keys=True) for row in rows]
    atomic_write(data_root / CROPS_FILE, ("\n".join(lines) + "\n").encode("utf-8"))
    return header


def read_crops(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The header and the rows of a crops file of make_crops."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"mapillary_invalid: empty file {path}")
    header = json.loads(lines[0])
    if header.get("kind") != CROPS_KIND or header.get("version") != CROPS_VERSION:
        raise ValueError(f"mapillary_invalid: {path} is not a crops file of nnslr mapillary-crops")
    return header, [json.loads(line) for line in lines[1:]]


def crops_state(data_root: Path) -> dict[str, Any]:
    """Whether the crops are ready, for the panel: {'available', 'crops', 'labels', 'reason'}."""
    path = Path(data_root) / CROPS_FILE
    if not path.is_file():
        return {"available": False, "crops": 0, "labels": 0, "reason": "Mapillary is not prepared (helpers/mapillary.sh)"}
    try:
        header, _ = read_crops(path)
    except (OSError, ValueError):
        return {"available": False, "crops": 0, "labels": 0, "reason": "the Mapillary crops file is not readable"}
    crops = int(header.get("crops") or 0)
    return {"available": crops > 0, "crops": crops, "labels": len(header.get("by_label") or {}),
            "reason": "" if crops else "no Mapillary sign of ours was found"}


def _order(row: dict[str, Any]) -> str:
    return hashlib.sha1(str(row.get("mtsd_object")).encode()).hexdigest()


def training_records(path: Path, classes: set[str], other_kinds: set[str]) -> tuple[list[dict[str, Any]], dict]:
    """The train records of the crops file ``path`` for a classifier plan: the speed signs and their end always
    (a value we have never seen becomes a class: the comma reads any value), an other sign only when its kind is
    one of ``other_kinds`` (our kinds that are classes). At most PER_LABEL of each label. Returns the records and
    a summary for the plan (file, sha256, crops by label, what was left out)."""
    header, rows = read_crops(path)
    by_label: dict[str, list[dict[str, Any]]] = {}
    left_out: Counter = Counter()
    for row in rows:
        label = row.get("label")
        if not isinstance(label, str):
            left_out["no_label"] += 1
            continue
        if label.startswith("other_sign:") and label not in other_kinds:
            left_out["kind_not_a_class_of_ours"] += 1
            continue
        by_label.setdefault(label, []).append(row)
    records = []
    for label, items in sorted(by_label.items()):
        items.sort(key=_order)
        left_out["over_the_per_label_limit"] += max(0, len(items) - PER_LABEL)
        for row in items[:PER_LABEL]:
            family, _, value = label.partition(":")
            records.append({
                "annotation_id": None, "split": "train", "image_path": row["image_path"], "image_sha256": None,
                "bbox_xyxy": [float(v) for v in row["bbox_xyxy"]], "label": label,
                "sign_family": family, "sign_type": value if family == "other_sign" else None,
                "value_state": "read" if value.isdigit() else None,
                "value_kph": int(value) if value.isdigit() else None,
                "route_id": None, "segment_index": None, "decoded_frame_index": None,
                "label_basis": LABEL_BASIS, "source": LABEL_BASIS, "mtsd_object": row.get("mtsd_object"),
            })
    from nnslr_tools.training import _sha256
    summary = {"file": str(path), "sha256": _sha256(Path(path)), "crops": len(records),
               "new_classes": sorted({r["label"] for r in records} - classes),
               "by_label": dict(sorted(Counter(r["label"] for r in records).items())),
               "left_out": dict(sorted(left_out.items())), "per_label": PER_LABEL}
    return records, summary
