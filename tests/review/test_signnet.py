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
    heat, shape, mask, weight = encode([box], stride)
    assert heat.shape == (1, BAND_HEIGHT // stride, BAND_WIDTH // stride) and mask.sum() == 1
    assert weight.shape == mask.shape and float(weight.min()) == 1.0
    assert float(heat.max()) == 1.0
    found = decode(heat[None], shape[None], stride)[0]
    assert found[0]['score'] == pytest.approx(1.0)
    assert found[0]['box'] == pytest.approx(box, abs=1e-3)
    assert math.isclose(len([f for f in found if f['score'] > 0.5]), 1)


def test_network_shapes_and_loss_at_the_target():
    model = build_signnet(stride=8).eval()
    heat, shape = model(torch.rand(2, 3, BAND_HEIGHT, BAND_WIDTH))
    assert heat.shape == (2, 1, BAND_HEIGHT // 8, BAND_WIDTH // 8) and shape.shape == (2, 4, BAND_HEIGHT // 8, BAND_WIDTH // 8)
    target_heat, target_shape, mask, _ = encode([[100.0, 100.0, 130.0, 130.0]], 8)
    perfect, _ = loss(target_heat[None].clamp(1e-4, 1 - 1e-4), target_shape[None], target_heat[None], target_shape[None],
                      mask[None])
    random_guess, _ = loss(heat[:1], shape[:1], target_heat[None], target_shape[None], mask[None])
    assert float(perfect) < float(random_guess)


def test_half_band():
    # Half the resolution: half the band both ways; boxes scaled to it, and the targets the size of its maps.
    assert band_size(BAND_WIDTH // 2) == (BAND_WIDTH // 2, BAND_HEIGHT // 2)
    assert band_size(896)[1] % 16 == 0
    assert band_boxes([[200, 100, 240, 140]], width=BAND_WIDTH, band_width=BAND_WIDTH // 2) == [[100.0, 50.0, 120.0, 70.0]]
    heat, *_ = encode([[100.0, 50.0, 120.0, 70.0]], 4, BAND_WIDTH // 2)
    assert heat.shape == (1, BAND_HEIGHT // 8, BAND_WIDTH // 8)
    model = build_signnet(stride=4).eval()
    out, _ = model(torch.rand(1, 3, BAND_HEIGHT // 2, BAND_WIDTH // 2))
    assert out.shape == heat[None].shape


def test_ignore_boxes_are_neither_signs_nor_background():
    from nnslr_tools.signnet import _in_ignore
    # A switched-off display at 100..130: no centre, weight 0 on its cells and one cell around (stride 8: 11..17).
    heat, shape, mask, weight = encode([], 8, ignore=[[100.0, 100.0, 130.0, 130.0]])
    assert float(heat.max()) == 0.0 and mask.sum() == 0
    assert weight[14, 14] == 0 and weight[11, 11] == 0 and weight[17, 17] == 0
    assert weight[18, 18] == 1 and weight[10, 10] == 1 and weight[0, 0] == 1
    # A confident peak on the panel costs nothing with the weight, and does without it.
    pred = torch.full((1, 1) + tuple(weight.shape), 1e-4)
    pred[0, 0, 14, 14] = 0.9
    zero_shape = torch.zeros(1, 4, *weight.shape)
    judged, _ = loss(pred, zero_shape, heat[None], shape[None], mask[None], weight[None])
    unjudged, _ = loss(pred, zero_shape, heat[None], shape[None], mask[None])
    assert float(judged) == pytest.approx(0.0, abs=1e-3) and float(unjudged) > 1.0
    # A sign centred on an ignored cell still counts as a sign.
    heat, shape, mask, weight = encode([[110.0, 110.0, 120.0, 120.0]], 8, ignore=[[100.0, 100.0, 130.0, 130.0]])
    assert mask.sum() == 1 and float(heat.max()) == 1.0
    missed, _ = loss(torch.full((1, 1) + tuple(weight.shape), 1e-4), zero_shape, heat[None], shape[None], mask[None],
                     weight[None])
    assert float(missed) > 1.0
    # When measuring, a box found on the panel is dropped: its centre inside, or half of it over the panel.
    panel = [[100.0, 100.0, 130.0, 130.0]]
    assert _in_ignore([105.0, 105.0, 125.0, 125.0], panel) and _in_ignore([95.0, 95.0, 128.0, 128.0], panel)
    assert not _in_ignore([140.0, 100.0, 170.0, 130.0], panel) and not _in_ignore([120.0, 120.0, 180.0, 180.0], panel)


def test_fuse_kernel_one_gives_the_same_maps():
    model = build_signnet(stride=4, fuse_kernel=1).eval()
    heat, shape = model(torch.rand(1, 3, BAND_HEIGHT // 2, BAND_WIDTH // 2))
    assert heat.shape == (1, 1, BAND_HEIGHT // 8, BAND_WIDTH // 8) and shape.shape[1] == 4
    assert model.fuse[0].kernel_size == (1, 1)
    with pytest.raises(ValueError):
        build_signnet(fuse_kernel=2)


def test_export_gives_one_output_equal_to_the_two(tmp_path):
    ort = pytest.importorskip('onnxruntime')
    import numpy as np

    from nnslr_tools.signnet import decode_output, export_signnet_onnx
    model = build_signnet((8, 8, 8, 8), 8, 1)
    torch.save({'task': 'signnet', 'state_dict': model.state_dict(), 'widths': [8, 8, 8, 8], 'stride': 8,
                'fuse_kernel': 1, 'band': [BAND_WIDTH // 2, BAND_HEIGHT // 2], 'epoch': 1}, tmp_path / 'signnet-best.pt')
    report = export_signnet_onnx(tmp_path / 'signnet-best.pt', tmp_path / 'signnet.onnx')
    assert report['verification']['passed'] and (tmp_path / 'signnet.onnx.contract.json').exists()
    session = ort.InferenceSession(str(tmp_path / 'signnet.onnx'), providers=['CPUExecutionProvider'])
    assert [o.name for o in session.get_outputs()] == ['out']
    images = np.random.rand(1, 3, BAND_HEIGHT // 2, BAND_WIDTH // 2).astype(np.float32)
    out = session.run(['out'], {'images': images})[0]
    assert out.shape == (1, 5, BAND_HEIGHT // 16, BAND_WIDTH // 16)
    with torch.no_grad():
        heat, shape = model.eval()(torch.from_numpy(images))
    assert np.allclose(out, torch.cat([heat, shape], 1).numpy(), atol=1e-4)
    first = decode_output(torch.from_numpy(out), 8, min_score=0.0)[0][0]
    same = decode(heat, shape, 8, min_score=0.0)[0][0]
    assert first['score'] == pytest.approx(same['score'], abs=1e-4) and first['box'] == pytest.approx(same['box'], abs=0.05)
