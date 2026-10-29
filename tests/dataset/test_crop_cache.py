"""The crop cache of a reader/classifier training (nnslr_tools.crop_cache): a crop read from it is the crop the
training used to cut from the frame, jittered or not, and a cache is cut once and found again."""
from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from nnslr_tools import crop_cache
from nnslr_tools.crop_cache import EXACT, WINDOW, build_crop_cache
from nnslr_tools.training import _sample_seed, crop_rectangle, jitter_box

SIZE = 32


def _frame(path, width=200, height=120, seed=0):
    """A noisy frame, so that a crop one pixel off differs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def _records(boxes, image="frames/a.png"):
    return [{"image_path": image, "image_sha256": None, "bbox_xyxy": box} for box in boxes]


def _reference(root, record, box, scheme="v2"):
    """The crop of the training before the cache: the frame cut, then resized to the input size."""
    with Image.open(root / record["image_path"]) as source:
        image = source.convert("RGB")
    if scheme == "v2":
        region = crop_rectangle(box, image.width, image.height)
    else:
        region = tuple(int(value) for value in box)
    return np.asarray(image.crop(region).resize((SIZE, SIZE), Image.BILINEAR), dtype=int)


def _read(cache, index, box):
    return np.asarray(cache.crop(index, box).resize((SIZE, SIZE), Image.BILINEAR), dtype=int)


# A box in the middle, on each border, partly outside the frame, one pixel wide, and a near sign (larger than
# MAX_BOX_SIDE input sizes: its window is kept smaller).
BOXES = [[60.2, 40.7, 80.9, 61.3], [0.0, 0.0, 12.5, 9.0], [185.4, 100.1, 200.0, 120.0], [-5.0, 50.0, 10.0, 70.0],
         [100.0, 30.0, 101.0, 31.0], [20.0, 5.0, 150.0, 115.0]]


@pytest.mark.parametrize("workers", [0, 2])
def test_a_cached_crop_is_the_crop_of_the_frame(tmp_path, workers):
    _frame(tmp_path / "frames" / "a.png")
    records = _records(BOXES) * 2
    kinds = [WINDOW] * len(BOXES) + [EXACT] * len(BOXES)
    cache, summary = build_crop_cache(tmp_path, records, kinds, size=SIZE, scheme="v2", content={"dataset": "x"},
                                      workers=workers)
    assert summary["built"] and summary["crops"] == 12 and summary["windows"] == 6 and summary["frames"] == 1
    # Validation and the crops without jitter: exactly the old crop.
    for index in range(len(BOXES), len(records)):
        assert (np.asarray(cache.crop(index, None), dtype=int) == _reference(tmp_path, records[index],
                                                                             records[index]["bbox_xyxy"])).all()
    # The jittered crops: the same pixels for a window at the frame's resolution, very close for the near sign.
    for epoch in range(25):
        for index, box in enumerate(BOXES):
            moved = jitter_box(box, random.Random(_sample_seed("box", 0, epoch, index)))
            difference = np.abs(_read(cache, index, moved) - _reference(tmp_path, records[index], moved))
            if index == len(BOXES) - 1:
                assert difference.mean() < 40, (epoch, difference.mean())
            else:
                assert difference.max() == 0, (epoch, index, box, moved)
    # The near sign's window is smaller than the frame's pixels; the others keep them.
    assert cache.index[len(BOXES) - 1, 1] < 120 and cache.index[0, 2] == cache.index[0, 5] - cache.index[0, 3]


def test_the_v1_reader_keeps_its_integer_box(tmp_path):
    _frame(tmp_path / "frames" / "a.png")
    records = _records([[10.7, 10.2, 40.9, 30.5], [190.0, 100.0, 215.0, 130.0]])
    cache, _ = build_crop_cache(tmp_path, records, [EXACT, EXACT], size=SIZE, scheme="v1", content={})
    for index, record in enumerate(records):
        assert (np.asarray(cache.crop(index), dtype=int) == _reference(tmp_path, record, record["bbox_xyxy"],
                                                                         scheme="v1")).all()


def test_a_cache_is_cut_once_and_a_change_makes_another(tmp_path):
    _frame(tmp_path / "frames" / "a.png")
    _frame(tmp_path / "frames" / "b.png", seed=1)
    records = _records(BOXES[:2]) + _records(BOXES[:1], image="frames/b.png")
    kinds = [WINDOW, EXACT, WINDOW]
    first, summary = build_crop_cache(tmp_path, records, kinds, size=SIZE, scheme="v2", content={"dataset": "x"})
    assert summary["built"] and summary["frames"] == 2
    again, summary = build_crop_cache(tmp_path, records, kinds, size=SIZE, scheme="v2", content={"dataset": "x"})
    assert not summary["built"] and again.folder == first.folder
    assert (tmp_path / crop_cache.CACHE_FOLDER).is_dir() and len(list((tmp_path / crop_cache.CACHE_FOLDER).iterdir())) == 1
    # Another dataset, box, storage or input size is another cache.
    for change in ({"content": {"dataset": "y"}}, {"kinds": [EXACT, EXACT, WINDOW]}, {"size": SIZE + 1},
                   {"records": _records(BOXES[1:3]) + records[2:]}):
        arguments = {"records": records, "kinds": kinds, "size": SIZE, "content": {"dataset": "x"}, **change}
        other, summary = build_crop_cache(tmp_path, arguments.pop("records"), arguments.pop("kinds"), scheme="v2",
                                          **arguments)
        assert summary["built"] and other.folder != first.folder
    with pytest.raises(ValueError, match="one storage kind per record"):
        build_crop_cache(tmp_path, records, kinds[:2], size=SIZE, scheme="v2", content={})


def test_the_window_holds_the_largest_jitter():
    # Every jittered crop of a box is inside its window, whatever the draw.
    width, height = 1928, 1208
    for box in ([500.0, 300.0, 540.0, 344.0], [3.0, 3.0, 9.0, 8.0], [1900.0, 1180.0, 1928.0, 1208.0]):
        left, top, right, bottom = crop_cache._window(box, width, height)
        for seed in range(2000):
            x1, y1, x2, y2 = crop_rectangle(jitter_box(box, random.Random(seed)), width, height)
            assert left <= x1 and top <= y1 and x2 <= right and y2 <= bottom, (box, seed)
