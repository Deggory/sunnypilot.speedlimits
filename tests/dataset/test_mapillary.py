"""The signs of the Mapillary Traffic Sign Dataset in the classifier's train: which MTSD labels are ours, the crops
cut from a tiny MTSD-like folder, the cap per label, and the plan of a v2 classifier with them."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest
from PIL import Image

from nnslr_tools import mapillary
from nnslr_tools.mapillary import (CROPS_FILE, crops_state, make_crops, our_label, read_crops,
                                   training_records)
from nnslr_tools.training import TrainConfig, prepare_reader_training, resume_arguments
from test_classifier import _classifier_fixture


def test_the_mtsd_labels_that_are_ours():
    assert our_label("regulatory--maximum-speed-limit-50--g1") == "maximum_speed:50"
    assert our_label("regulatory--maximum-speed-limit-120--g1") == "maximum_speed:120"
    # A US design (mph) and a value no road of ours has are not ours.
    assert our_label("regulatory--maximum-speed-limit-45--g3") is None
    assert our_label("regulatory--maximum-speed-limit-5--g1") is None
    assert our_label("regulatory--yield--g1") == "other_sign:give_way"
    assert our_label("information--parking--g1") is None


def _sign(key, label, box, **properties):
    return {"key": key, "label": label, "properties": properties,
            "bbox": dict(zip(("xmin", "ymin", "xmax", "ymax"), box))}


def _mtsd(folder: Path) -> tuple[Path, Path]:
    """Three photos: two speed signs and a give way (one occluded, one too small, one ambiguous), one photo
    without a sign of ours, one annotation whose photo is not downloaded."""
    annotations, images = folder / "annotations", folder / "images"
    annotations.mkdir(parents=True)
    images.mkdir()
    documents = {
        "p1": [_sign("s50", "regulatory--maximum-speed-limit-50--g1", [100, 40, 300, 240]),
               _sign("tiny", "regulatory--maximum-speed-limit-80--g1", [10, 10, 20, 20]),
               _sign("yield", "regulatory--yield--g1", [320, 20, 360, 60], occluded=True)],
        "p2": [_sign("s80", "regulatory--maximum-speed-limit-80--g1", [0, 0, 50, 50]),
               _sign("unsure", "regulatory--maximum-speed-limit-30--g1", [60, 0, 110, 50], ambiguous=True),
               _sign("parking", "information--parking--g1", [200, 0, 250, 50])],
        "p3": [_sign("park", "information--parking--g1", [0, 0, 40, 40])],
        "p4": [_sign("away", "regulatory--maximum-speed-limit-60--g1", [0, 0, 40, 40])],
    }
    for key, objects in documents.items():
        (annotations / f"{key}.json").write_text(json.dumps({"width": 400, "height": 300, "ispano": False,
                                                             "objects": objects}))
        if key != "p4":
            Image.new("RGB", (400, 300), (90, 120, 160)).save(images / f"{key}.jpg")
    return annotations, images


def test_the_crops_of_a_tiny_mtsd(tmp_path):
    annotations, images = _mtsd(tmp_path / "mtsd")
    root = tmp_path / "data"
    assert crops_state(root) == {"available": False, "crops": 0, "labels": 0,
                                 "reason": "Mapillary is not prepared (helpers/mapillary.sh)"}
    header = make_crops(annotations, images, root, log=lambda *_: None)
    assert header["by_label"] == {"maximum_speed:50": 1, "maximum_speed:80": 1, "other_sign:give_way": 1}
    assert header["skipped"] == {"ambiguous": 1, "not_ours": 2, "photo_not_downloaded": 1, "too_small": 1}
    _, rows = read_crops(root / CROPS_FILE)
    by_object = {row["mtsd_object"]: row for row in rows}
    # The 200 px sign is cut with half its side around it and scaled down to MAX_SIGN_SIDE.
    big = by_object["s50"]
    with Image.open(root / big["image_path"]) as crop:
        assert crop.size == (192, 144)  # 300 px of the photo, its whole height
    assert big["bbox_xyxy"] == [48.0, 19.2, 144.0, 115.2] and not big["occluded"]
    # A sign on the photo's border keeps its pixels: the region stops at the border.
    assert by_object["s80"]["bbox_xyxy"] == [0.0, 0.0, 50.0, 50.0] and by_object["yield"]["occluded"]
    assert crops_state(root) == {"available": True, "crops": 3, "labels": 3, "reason": ""}
    with pytest.raises(ValueError, match="no photos folder"):
        make_crops(annotations, tmp_path / "nowhere", root)


def test_the_crops_straight_from_the_zips_and_a_resume(tmp_path):
    """MTSD as downloaded: the annotations in one zip, the photos in another, a third still being copied."""
    annotations, images = _mtsd(tmp_path / "mtsd")
    share = tmp_path / "share" / "zips"
    share.mkdir(parents=True)
    with zipfile.ZipFile(share / "mtsd_v2_fully_annotated_annotation.zip", "w") as archive:
        for path in annotations.iterdir():
            archive.write(path, f"mtsd_v2_fully_annotated/annotations/{path.name}")
        archive.writestr("mtsd_v2_fully_annotated/splits/train.txt", "p1\n")
    with zipfile.ZipFile(share / "mtsd_fully_annotated_images.train.0.zip", "w") as archive:
        for path in images.iterdir():
            archive.write(path, f"images/{path.name}")
            archive.writestr(f"__MACOSX/images/._{path.name}", b"junk")
    (share / "mtsd_fully_annotated_images.train.1.zip").write_bytes(b"PK\x03\x04 half copied")
    root = tmp_path / "data"
    lines = []
    header = make_crops(share.parent, share.parent, root, log=lines.append)
    assert header["by_label"] == {"maximum_speed:50": 1, "maximum_speed:80": 1, "other_sign:give_way": 1}
    assert header["skipped"] == {"ambiguous": 1, "not_ours": 2, "photo_not_downloaded": 1, "too_small": 1}
    assert header["zips_left_out"] == [str(share / "mtsd_fully_annotated_images.train.1.zip")]
    first = (root / CROPS_FILE).read_text()
    # Run again with photos that do not decode any more: none is read, the crops already cut are kept.
    with zipfile.ZipFile(share / "mtsd_fully_annotated_images.train.0.zip", "w") as archive:
        for path in images.iterdir():
            archive.writestr(f"images/{path.name}", b"not a photo any more")
    lines.clear()
    assert make_crops(share.parent, share.parent, root, log=lines.append)["crops"] == 3
    assert (root / CROPS_FILE).read_text() == first and "mapillary: 3 crops already cut, kept" in lines


def _crops_file(root: Path, rows: list[dict]) -> Path:
    path = root / CROPS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"kind": "mapillary_crops", "version": 1, "crops": len(rows), "by_label": {}}
    path.write_text("\n".join(json.dumps(item) for item in [header] + rows) + "\n")
    return path


def test_the_train_records_keep_our_kinds_and_a_cap_per_label(tmp_path, monkeypatch):
    monkeypatch.setattr(mapillary, "PER_LABEL", 2)
    rows = ([{"image_path": f"c/{n}.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "maximum_speed:120", "mtsd_object": f"a{n}"}
             for n in range(3)]
            + [{"image_path": "c/g.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "other_sign:give_way", "mtsd_object": "g"},
               {"image_path": "c/s.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "other_sign:stop", "mtsd_object": "s"}])
    path = _crops_file(tmp_path, rows)
    records, summary = training_records(path, {"maximum_speed:50", "other_sign:give_way"}, {"other_sign:give_way"})
    assert summary["by_label"] == {"maximum_speed:120": 2, "other_sign:give_way": 1}
    assert summary["left_out"] == {"kind_not_a_class_of_ours": 1, "over_the_per_label_limit": 1}
    assert summary["new_classes"] == ["maximum_speed:120"] and summary["per_label"] == 2
    speed = next(r for r in records if r["label"] == "maximum_speed:120")
    assert (speed["split"], speed["sign_family"], speed["value_kph"], speed["label_basis"]) == (
        "train", "maximum_speed", 120, "mapillary")
    give_way = next(r for r in records if r["label"] == "other_sign:give_way")
    assert (give_way["sign_type"], give_way["value_kph"]) == ("give_way", None)
    # The same file gives the same records: the cap keeps a fixed choice.
    assert training_records(path, set(), {"other_sign:give_way"})[0] == records


def test_the_classifier_plan_with_mapillary(tmp_path, capsys):
    dataset = _classifier_fixture(tmp_path, capsys)
    latest = tmp_path / "splits" / "latest.json"
    plain = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2")
    Image.new("RGB", (40, 40)).save(tmp_path / "s.jpg")
    path = _crops_file(tmp_path, [
        {"image_path": "s.jpg", "bbox_xyxy": [10, 10, 30, 30], "label": "maximum_speed:120", "mtsd_object": "m"},
        {"image_path": "s.jpg", "bbox_xyxy": [10, 10, 30, 30], "label": "other_sign:roundabout", "mtsd_object": "r"}])
    plan = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", mapillary_path=path)
    assert plan["classes"] == sorted(plain["classes"] + ["maximum_speed:120"])
    assert plan["mapillary"]["by_label"] == {"maximum_speed:120": 1}
    assert plan["mapillary"]["left_out"] == {"kind_not_a_class_of_ours": 1, "over_the_per_label_limit": 0}
    assert plan["mapillary_sha256"] == plan["mapillary"]["sha256"] and plain["mapillary_sha256"] is None
    assert any(item.startswith("mapillary_crops_in_train") for item in plan["limitations"])
    # Validation and test are our drives only.
    assert {r["split"] for r in plan["records"] if r["label_basis"] == "mapillary"} == {"train"}
    assert plan["counts_by_split"]["validation"] == plain["counts_by_split"]["validation"]
    with pytest.raises(ValueError, match="--classes v2"):
        prepare_reader_training(dataset, latest, tmp_path, mapillary_path=path)
    (tmp_path / "s.jpg").unlink()
    with pytest.raises(ValueError, match="missing_image: s.jpg"):
        prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", mapillary_path=path)


def test_a_classifier_with_mapillary_resumes_with_it():
    config = TrainConfig(class_scheme="v2", augment=True, mapillary="/data/external/mapillary/crops.jsonl").__dict__
    flags = resume_arguments({"task": "reader", "config": config})
    assert flags[flags.index("--mapillary") + 1] == "/data/external/mapillary/crops.jsonl"
    assert "--mapillary" not in resume_arguments({"task": "reader", "config": TrainConfig(class_scheme="v2").__dict__})
