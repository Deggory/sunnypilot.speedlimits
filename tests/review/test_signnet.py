"""SignNet's targets and decoding (nnslr_tools.signnet): a box encoded into the heat map comes back out as itself."""
import math

import pytest

torch = pytest.importorskip('torch')

from nnslr_tools.signnet import (
    BAND_HEIGHT,
    BAND_WIDTH,
    band_boxes,
    band_size,
    build_signnet,
    decode,
    encode,
    loss,
)


def test_band_boxes_keep_the_signs_of_the_band():
    # A frame twice as wide: scaled to the band; a sign centred below the band is left out.
    boxes = band_boxes([[200, 100, 240, 140], [400, 900, 440, 960]], width=BAND_WIDTH * 2)
    assert boxes == [[100.0, 50.0, 120.0, 70.0]]


@pytest.mark.parametrize('stride', (4, 8))
def test_encode_then_decode_gives_the_box_back(stride):
    box = [601.0, 133.0, 627.0, 161.0]
    heat, shape, mask = encode([box], stride)
    assert heat.shape == (1, BAND_HEIGHT // stride, BAND_WIDTH // stride) and mask.sum() == 1
    assert float(heat.max()) == 1.0
    found = decode(heat[None], shape[None], stride)[0]
    assert found[0]['score'] == pytest.approx(1.0)
    assert found[0]['box'] == pytest.approx(box, abs=1e-3)
    assert math.isclose(len([f for f in found if f['score'] > 0.5]), 1)


def test_network_shapes_and_loss_at_the_target():
    model = build_signnet(stride=8).eval()
    heat, shape = model(torch.rand(2, 3, BAND_HEIGHT, BAND_WIDTH))
    assert heat.shape == (2, 1, BAND_HEIGHT // 8, BAND_WIDTH // 8) and shape.shape == (2, 4, BAND_HEIGHT // 8, BAND_WIDTH // 8)
    target_heat, target_shape, mask = encode([[100.0, 100.0, 130.0, 130.0]], 8)
    perfect, _ = loss(target_heat[None].clamp(1e-4, 1 - 1e-4), target_shape[None], target_heat[None], target_shape[None],
                      mask[None])
    random_guess, _ = loss(heat[:1], shape[:1], target_heat[None], target_shape[None], mask[None])
    assert float(perfect) < float(random_guess)


def test_half_band():
    # Half the resolution: half the band both ways; boxes scaled to it, and the targets the size of its maps.
    assert band_size(BAND_WIDTH // 2) == (BAND_WIDTH // 2, BAND_HEIGHT // 2)
    assert band_size(896)[1] % 16 == 0
    assert band_boxes([[200, 100, 240, 140]], width=BAND_WIDTH, band_width=BAND_WIDTH // 2) == [[100.0, 50.0, 120.0, 70.0]]
    heat, _, _ = encode([[100.0, 50.0, 120.0, 70.0]], 4, BAND_WIDTH // 2)
    assert heat.shape == (1, BAND_HEIGHT // 8, BAND_WIDTH // 8)
    model = build_signnet(stride=4).eval()
    out, _ = model(torch.rand(1, 3, BAND_HEIGHT // 2, BAND_WIDTH // 2))
    assert out.shape == heat[None].shape
