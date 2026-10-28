"""SignNet: a sign detector made for the comma's own GPU (the Adreno), to cut the signs out of the road frame in the car.

The cutter in use (SSDlite MobileNetV3 at 1344 x 760) takes 540 ms a frame on the Adreno with openpilot's tinygrad
flags, too slow; on the Chestnut it would stall the driving model (docs/NN_VISION_SPEED_LIMIT_REMAINING_WORK.md).
SignNet is the other way round: few layers of plain 3 x 3 convolutions (tinygrad pays for every layer), one output,
and only the top band of the frame (BAND_HEIGHT rows: 97 % of the reviewed signs end above row 420 of 760) at full
resolution, so the small far signs are not lost to a smaller picture. 141 ms on the Adreno untrained (27 Oct 2026).
``band_width`` trains it on a smaller band (672: half the resolution) to see how far the picture can shrink.

The output is a heat map at ``stride`` (a sign centred in this cell, sure this much) and, per cell, the sign's log
width and height in pixels and the offset of its centre in the cell (CenterNet): no default boxes, no non-maximum
suppression but the local maxima of the heat map. The classifier reads each box from the full frame, as with the
cutter.

``nnslr train-signnet`` trains it on the cutter's plan (prepare_detector_training: the same frames, boxes, families
and split) and measures it on validation and test with the cutter's own scoring (evaluation.score_frame,
detector_size_metrics), so the two compare one to one. Needs the GPU environment (.venv-gpu).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

BAND_WIDTH = 1344
BAND_HEIGHT = 416
DEFAULT_WIDTHS = (16, 32, 48, 64)
DEFAULT_STRIDE = 8
# A centre's heat spreads over the cells around it with this share of the sign's size as its Gaussian sigma.
SIGMA_SHARE = 1 / 6
MAX_DETECTIONS = 50
SCORE_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)
# Boxes per frame the classifier after it can afford on the Adreno (about 5 ms a box).
BOX_BUDGETS = (3, 5, 10)


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ValueError("training_dependencies_missing: run it with .venv-gpu") from exc
    return torch, nn


def band_size(band_width: int = BAND_WIDTH) -> tuple[int, int]:
    """The band at ``band_width`` pixels across: the same share of the frame as BAND_HEIGHT, its height rounded up to
    a multiple of 16 (the deepest stride)."""
    return band_width, math.ceil(BAND_HEIGHT * band_width / BAND_WIDTH / 16) * 16


def build_signnet(widths=DEFAULT_WIDTHS, stride: int = DEFAULT_STRIDE, fuse_kernel: int = 3):
    """The network: strides 2, 4, 8, 16; the deepest map brought back to ``stride`` (4 or 8) and joined with the
    map of that stride by a convolution of ``fuse_kernel`` (3, or 1: on the Adreno the 3 x 3 at stride 4 is 40 % of
    the time); out: heat (1, sigmoid) and shape (log width, log height, offset x, offset y)."""
    torch, nn = _torch()
    if stride not in (4, 8):
        raise ValueError("signnet stride must be 4 or 8")
    if fuse_kernel not in (1, 3):
        raise ValueError("signnet fuse kernel must be 1 or 3")

    def block(cin, cout, step, kernel=3):
        return nn.Sequential(nn.Conv2d(cin, cout, kernel, step, kernel // 2, bias=False), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True))

    class SignNet(nn.Module):
        def __init__(self):
            super().__init__()
            a, b, c, d = widths
            self.stride = stride
            self.s2 = block(3, a, 2)
            self.s4 = nn.Sequential(block(a, b, 2), block(b, b, 1))
            self.s8 = nn.Sequential(block(b, c, 2), block(c, c, 1))
            self.s16 = nn.Sequential(block(c, d, 2), block(d, d, 1))
            self.fuse = block(d + (b if stride == 4 else c), c, 1, fuse_kernel)
            self.out = nn.Conv2d(c, 5, 1)
            # Heat starts near 0.01 everywhere (CenterNet's prior): the focal loss starts from few false alarms.
            nn.init.constant_(self.out.bias[:1], -4.6)

        def forward(self, images):
            x4 = self.s4(self.s2(images))
            x8 = self.s8(x4)
            x16 = self.s16(x8)
            skip = x4 if self.stride == 4 else x8
            up = nn.functional.interpolate(x16, size=skip.shape[-2:], mode='nearest')
            y = self.out(self.fuse(torch.cat([up, skip], 1)))
            return torch.sigmoid(y[:, :1]), y[:, 1:]

    return SignNet()


def band_boxes(boxes: list[list[float]], width: int, band_width: int = BAND_WIDTH) -> list[list[float]]:
    """Reviewed boxes in the band's pixels (the frame scaled to band_width): those whose centre is in the band."""
    scale = band_width / float(width)
    band_height = band_size(band_width)[1]
    kept = []
    for x1, y1, x2, y2 in boxes:
        box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        if (box[1] + box[3]) / 2 < band_height:
            kept.append([box[0], box[1], box[2], min(box[3], float(band_height))])
    return kept


def encode(boxes: list[list[float]], stride: int, band_width: int = BAND_WIDTH):
    """Targets of one band: heat (1, H, W), shape (4, H, W), mask (H, W) of the cells that hold a centre."""
    torch, _ = _torch()
    band_w, band_h = band_size(band_width)
    height, width = band_h // stride, band_w // stride
    heat = torch.zeros(1, height, width)
    shape = torch.zeros(4, height, width)
    mask = torch.zeros(height, width)
    ys = torch.arange(height).view(-1, 1).float()
    xs = torch.arange(width).view(1, -1).float()
    for x1, y1, x2, y2 in boxes:
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        cx, cy = (x1 + x2) / 2 / stride, (y1 + y2) / 2 / stride
        ix, iy = min(int(cx), width - 1), min(int(cy), height - 1)
        sigma = max(max(w, h) / stride * SIGMA_SHARE, 0.5)
        gauss = torch.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma * sigma))
        heat[0] = torch.maximum(heat[0], gauss)
        shape[:, iy, ix] = torch.tensor([math.log(w), math.log(h), cx - ix, cy - iy])
        mask[iy, ix] = 1.0
    return heat, shape, mask


def loss(pred_heat, pred_shape, heat, shape, mask):
    """CenterNet's focal loss on the heat map, plus L1 on the shape at the centres; both per sign."""
    torch, _ = _torch()
    pred = pred_heat.clamp(1e-4, 1 - 1e-4)
    positive = heat.eq(1).float()
    negative = 1 - positive
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * positive
    neg_loss = torch.log(1 - pred) * pred ** 2 * (1 - heat) ** 4 * negative
    count = positive.sum().clamp(min=1)
    focal = -(pos_loss.sum() + neg_loss.sum()) / count
    centre = mask.unsqueeze(1)
    shape_loss = (torch.abs(pred_shape - shape) * centre).sum() / (centre.sum() * 4).clamp(min=1)
    return focal + shape_loss, {'focal': float(focal.detach()), 'shape': float(shape_loss.detach())}


def decode(heat, shape, stride: int, *, limit: int = MAX_DETECTIONS, min_score: float = 0.01) -> list[list[dict]]:
    """Boxes of a batch, in band pixels: the local maxima of the heat map (3 x 3), at most ``limit`` a frame."""
    _, nn = _torch()
    peaks = heat * (nn.functional.max_pool2d(heat, 3, 1, 1) == heat).float()
    batch, _, height, width = heat.shape
    scores, indexes = peaks.view(batch, -1).topk(min(limit, height * width))
    found = []
    for b in range(batch):
        frame = []
        for score, index in zip(scores[b].tolist(), indexes[b].tolist()):
            if score < min_score:
                break
            iy, ix = divmod(index, width)
            log_w, log_h, ox, oy = shape[b, :, iy, ix].tolist()
            w, h = math.exp(min(log_w, 8.0)), math.exp(min(log_h, 8.0))
            cx, cy = (ix + ox) * stride, (iy + oy) * stride
            frame.append({'box': [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 'score': score})
        found.append(frame)
    return found


def _band_image(path: Path, band_width: int = BAND_WIDTH):
    """The frame's top band as a uint8 tensor (3, band height, band_width), and the frame's width."""
    torch, _ = _torch()
    from PIL import Image
    band_w, band_h = band_size(band_width)
    with Image.open(path) as image:
        image = image.convert('RGB')
        width, height = image.size
        if width != band_w:
            image = image.resize((band_w, round(height * band_w / width)), Image.BILINEAR)
        band = image.crop((0, 0, band_w, band_h))
        import numpy as np
        array = np.asarray(band, dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1), width


def _dataset(frames: list[dict], data_root: Path, stride: int, *, train: bool, seed: int, band_width: int = BAND_WIDTH):
    _torch()
    from torch.utils.data import Dataset

    class Bands(Dataset):
        def __len__(self):
            return len(frames)

        def __getitem__(self, index):
            frame = frames[index]
            image, width = _band_image(data_root / frame['image_path'], band_width)
            boxes = band_boxes(frame['boxes'], width, band_width)
            image = image.float() / 255.0
            if train:
                # Python's random is seeded apart in every DataLoader worker (base seed + worker id).
                rng = random
                if rng.random() < 0.5:
                    image = image.flip(-1)
                    boxes = [[band_width - x2, y1, band_width - x1, y2] for x1, y1, x2, y2 in boxes]
                image = (image * rng.uniform(0.7, 1.3) + rng.uniform(-0.08, 0.08)).clamp(0, 1)
            heat, shape, mask = encode(boxes, stride, band_width)
            return image, heat, shape, mask, index

    return Bands()


def measure(model, frames: list[dict], data_root: Path, stride: int, device, *, batch: int = 16,
            workers: int = 8, band_width: int = BAND_WIDTH) -> dict[str, Any]:
    """The cutter's measures on ``frames``: per score threshold (detector_size_metrics, reviewed boxes of the whole
    frame, those below the band count as missed) and per box budget (the best N boxes of each frame)."""
    torch, _ = _torch()
    from torch.utils.data import DataLoader

    from nnslr_tools.evaluation import detector_size_metrics, score_frame
    loader = DataLoader(_dataset(frames, data_root, stride, train=False, seed=0, band_width=band_width),
                        batch_size=batch, shuffle=False, num_workers=workers)
    predictions: dict[int, list[dict]] = {}
    model.eval()
    with torch.no_grad():
        for image, _, _, _, indexes in loader:
            heat, shape = model(image.to(device))
            for index, found in zip(indexes.tolist(), decode(heat.float().cpu(), shape.float().cpu(), stride)):
                scale = float(frames[index]['width']) / band_width
                predictions[index] = [{'box': [v * scale for v in p['box']], 'score': p['score']} for p in found]

    def scored(select) -> dict[str, Any]:
        items = []
        for index, frame in enumerate(frames):
            chosen = select(predictions.get(index, []))
            item = score_frame(frame['boxes'], chosen, score_threshold=0.0, iou_threshold=0.5)
            item['truth_families'] = list(frame.get('box_families') or [])
            item['label_basis'] = frame.get('label_basis')
            items.append(item)
        return detector_size_metrics(items)

    return {
        'by_score_threshold': {str(t): scored(lambda found, t=t: [p for p in found if p['score'] >= t])
                               for t in SCORE_THRESHOLDS},
        'by_box_budget': {str(n): scored(lambda found, n=n: found[:n]) for n in BOX_BUDGETS},
    }


def train_signnet(plan: dict[str, Any], data_root: Path, run_dir: Path, *, widths=DEFAULT_WIDTHS,
                  stride: int = DEFAULT_STRIDE, epochs: int = 40, batch: int = 16, learning_rate: float = 3e-3,
                  workers: int = 8, seed: int = 1, band_width: int = BAND_WIDTH, fuse_kernel: int = 3,
                  log=print) -> dict[str, Any]:
    """Train on the plan's train frames, keep the epoch with the best validation recall at 5 boxes a frame, then
    measure it on validation and test. Writes signnet-best.pt and result.json in run_dir."""
    torch, _ = _torch()
    from torch.utils.data import DataLoader
    random.seed(seed)
    torch.manual_seed(seed)
    data_root, run_dir = Path(data_root), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    by_split: dict[str, list[dict]] = {}
    for frame in plan['frames']:
        by_split.setdefault(frame['split'], []).append(frame)
    train_frames, validation, test = by_split.get('train', []), by_split.get('validation', []), by_split.get('test', [])
    model = build_signnet(widths, stride, fuse_kernel).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(_dataset(train_frames, data_root, stride, train=True, seed=seed, band_width=band_width),
                        batch_size=batch, shuffle=True, num_workers=workers, drop_last=True, persistent_workers=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=epochs * len(loader))
    best, best_epoch, history = -1.0, -1, []
    for epoch in range(1, epochs + 1):
        model.train()
        started, total, parts = time.monotonic(), 0.0, {'focal': 0.0, 'shape': 0.0}
        for image, heat, shape, mask, _ in loader:
            pred_heat, pred_shape = model(image.to(device))
            value, detail = loss(pred_heat, pred_shape, heat.to(device), shape.to(device), mask.to(device))
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
            scheduler.step()
            total += float(value)
            for key in parts:
                parts[key] += detail[key]
        steps = max(len(loader), 1)
        checked = measure(model, validation, data_root, stride, device, workers=workers,
                          band_width=band_width)['by_box_budget']['5']
        history.append({'epoch': epoch, 'loss': total / steps, 'focal': parts['focal'] / steps,
                        'shape': parts['shape'] / steps, 'validation_recall_at_5': checked['recall_iou'],
                        'validation_false_positives_per_image_at_5': checked['false_positives_per_image'],
                        'seconds': round(time.monotonic() - started, 1)})
        log(f"signnet epoch {epoch}/{epochs} loss {total / steps:.4f} validation recall@5 {checked['recall_iou']:.3f}")
        if checked['recall_iou'] > best:
            best, best_epoch = checked['recall_iou'], epoch
            torch.save({'task': 'signnet', 'state_dict': model.state_dict(), 'widths': list(widths), 'stride': stride,
                        'fuse_kernel': fuse_kernel, 'band': list(band_size(band_width)), 'epoch': epoch},
                       run_dir / 'signnet-best.pt')
    checkpoint = torch.load(run_dir / 'signnet-best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    result = {
        'task': 'signnet', 'widths': list(widths), 'stride': stride, 'fuse_kernel': fuse_kernel,
        'band': list(band_size(band_width)),
        'parameters': sum(p.numel() for p in model.parameters()), 'epochs': epochs, 'best_epoch': best_epoch,
        'dataset_sha256': plan.get('dataset_sha256'), 'split_sha256': plan.get('split_sha256'),
        'families': plan.get('families'), 'counts_by_split': plan.get('counts_by_split'),
        'boxes_by_split': plan.get('boxes_by_split'), 'history': history,
        'validation': measure(model, validation, data_root, stride, device, workers=workers, band_width=band_width),
        'test': measure(model, test, data_root, stride, device, workers=workers, band_width=band_width),
    }
    (run_dir / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return result


def decode_output(out, stride: int, **kw) -> list[list[dict]]:
    """``decode`` on the one tensor the exported ONNX gives: heat in channel 0, the shape in channels 1 to 4."""
    return decode(out[:, :1], out[:, 1:], stride, **kw)


def load_signnet(checkpoint_path: Path):
    """The network of a signnet-best.pt in eval mode, and the checkpoint."""
    torch, _ = _torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('task') != 'signnet':
        raise ValueError('checkpoint_task_mismatch: expected signnet')
    model = build_signnet(tuple(checkpoint['widths']), checkpoint['stride'], checkpoint.get('fuse_kernel', 3))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval(), checkpoint


def export_signnet_onnx(checkpoint_path: Path, output_path: Path, *, verify: bool = True) -> dict[str, Any]:
    """SignNet as ONNX for the car. Input ``images`` (1, 3, band height, band width), float 0..1, RGB: the frame
    scaled to the band's width, its top rows. One output ``out`` (1, 5, H, W): heat (sigmoid), log width, log height,
    offset x, offset y (as two outputs the Adreno takes twice the time: 40 ms against 21 at 896 x 288). Checked
    against PyTorch with onnxruntime; writes ``<output>.contract.json`` beside it."""
    torch, nn = _torch()
    from nnslr_tools.exporting import _export_onnx, sha256_file
    model, checkpoint = load_signnet(checkpoint_path)
    band_w, band_h = checkpoint['band']
    stride = checkpoint['stride']

    class OneOutput(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, images):
            heat, shape = self.net(images)
            return torch.cat([heat, shape], 1)

    wrapped = OneOutput(model).eval()
    example = torch.rand(1, 3, band_h, band_w)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _export_onnx(wrapped, example, output_path, input_names=['images'], output_names=['out'])
    try:
        import onnx
    except ImportError as exc:
        raise ValueError('onnx_dependencies_missing: install onnx') from exc
    onnx.checker.check_model(onnx.load(str(output_path)))
    verification: dict[str, Any] = {'enabled': verify, 'passed': None}
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ValueError('onnxruntime_missing') from exc
        import numpy as np
        with torch.no_grad():
            expected = wrapped(example).numpy()
        session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
        actual = session.run(['out'], {'images': example.numpy()})[0]
        max_abs_error = float(np.max(np.abs(expected - actual)))
        passed = bool(np.allclose(expected, actual, rtol=1e-4, atol=1e-5))
        verification = {'enabled': True, 'passed': passed, 'max_abs_error': max_abs_error, 'rtol': 1e-4, 'atol': 1e-5,
                        'provider': 'CPUExecutionProvider'}
        if not passed:
            raise ValueError(f'onnx_parity_failed: max_abs_error={max_abs_error}')
    contract = {
        'task': 'signnet', 'widths': list(checkpoint['widths']), 'stride': stride,
        'fuse_kernel': checkpoint.get('fuse_kernel', 3), 'epoch': checkpoint.get('epoch'),
        'input': {'name': 'images', 'shape': [1, 3, band_h, band_w], 'dtype': 'float32', 'range': [0.0, 1.0],
                  'layout': 'RGB; the road frame scaled to the band width, its top rows (the band)'},
        'output': {'name': 'out', 'shape': [1, 5, band_h // stride, band_w // stride],
                   'channels': ['heat', 'log_width', 'log_height', 'offset_x', 'offset_y'],
                   'decode': 'a sign at each 3x3 local maximum of heat above the score wanted; its centre '
                             '((x + offset_x) * stride, (y + offset_y) * stride) and size (exp(log_width), '
                             'exp(log_height)), in band pixels; the frame scale is frame width / band width'},
        'onnx_sha256': sha256_file(output_path), 'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': sha256_file(Path(checkpoint_path)), 'verification': verification,
    }
    contract_path = output_path.with_suffix(output_path.suffix + '.contract.json')
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return {'valid': True, 'task': 'signnet', 'onnx': str(output_path), 'onnx_sha256': contract['onnx_sha256'],
            'contract': str(contract_path), 'verification': verification}
