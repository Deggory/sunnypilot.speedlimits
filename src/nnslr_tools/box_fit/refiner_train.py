"""Training of the box refiner (:mod:`.refiner`) on the boxes people confirmed, on the GPU (PyTorch of .venv-gpu, the
card CUDA_VISIBLE_DEVICES gives; without one it stops, never the CPU), written as ``refiner.onnx`` into the folder of a new
version of Fit box (:mod:`.versions`, ``--out``), then the measure of that version against the version in use on the routes
left out. Nothing in use is replaced: a person chooses the version (#94).

The boxes: every sign box of the frames a person confirmed in the review (``annotations/review.sqlite``, a reviewer
that is not ``ai/...``), not ``not_a_sign``, at least 4 px a side, on a frame whose picture is on the server; they
need no import into the dataset. Unlike the rankers' training (:mod:`.train`), a box a person kept as the model drew
it counts too: the person looked at it and kept it.

Each box has its source (``box_source`` of the review, #80): what last set its position, ``hand`` (the reviewer drew,
moved, resized or squared it), ``fit`` (Fit box), ``carried``, ``ai_fit``... A box saved without one is ``model`` when it
is still where its proposal was, else ``unknown`` (a review saved before #80). A box placed by hand is drawn
``HAND_WEIGHT`` times as often in training. The measure on the routes left out is also given by source and for the
boxes Fit box did not place (``fair``): a box Fit box placed and the reviewer accepted favours the refiner that drew it.

The routes are split by a hash of their name: about a quarter left out, the same every time. Each box of the others
has its neighbourhood cut once from its frame (``REGION`` times its longer side, at most ``MAX_REGION`` pixels); each
epoch draws new rough boxes around it (loose, shifted, tight: as a reviewer draws them and as a pass of the refiner
leaves them), crops them as :func:`.refiner.crop` does and augments the light, the colours, the blur, the noise and
the JPEG. The network also answers how right the rough box already is (its IoU with the person's box, 0..1), which
Fit box uses to leave a good box to the rankers (``refiner.KEEP``). The loss: smooth L1 on the sides plus GIoU, plus
L1 on that IoU. The epoch with the best mean IoU after ``STEPS`` passes on the routes left out is kept.

Then the measure, on the full frames of the routes left out with the rough boxes of the rankers' benchmark
(:func:`.train.make_cases`), each box fitted four ways: the rough box itself, the new version's rankers alone
(:func:`nnslr_tools.box_fit.fit_image` without the refiner), the new version as the review page runs it (under
``refiner.GATE`` pixels its refiner, then its rankers) and the version in use as the review page runs it (its own
rankers and refiner), on the same boxes. Each version is run from its ``measure/`` folder when it has one, the rankers
trained without these routes (:func:`.versions.measure_folder`); the new version's refiner is copied there first. The
summary says which of the two were clean (``clean``). ``--measure-only`` trains nothing and measures the version in
``--out`` (its rankers, and the refiner it holds, if any) the same way. ``--in-use ID`` names the version in use the
script read at its start, so that both measures and its last line speak of the same one. The summary is printed as JSON
on the last line.

usage: .venv-gpu/bin/python -m nnslr_tools.box_fit.refiner_train --out DIR [--epochs 40] [--in-use ID]
       [--dry-run | --measure-only]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from nnslr_tools.box_fit import refiner, versions
from nnslr_tools.box_fit.versions import held_out

REGION, MAX_REGION = 6.0, 384     # the neighbourhood cut around each box: its longer side times REGION, at most px
PER_BOX = 4                       # rough boxes of each box per epoch
BATCH = 128
LR = 2e-3
SEED = 0
MIN_BOXES = 200
HAND_WEIGHT = 3                   # a box a person placed by hand, drawn this many times as often in training
# The sources whose position Fit box may have made: they are left out of the "fair" measure.
FIT_SOURCES = ('fit', 'unknown')


def confirmed_boxes(review_db: Path, data_root: Path) -> list[dict]:
    """The boxes of the frames people confirmed in the review (see the module's description), in the database's
    order: a box a person kept as the model drew it counts too."""
    import sqlite3
    out = []
    db = sqlite3.connect(f'file:{review_db}?mode=ro', uri=True)
    try:
        rows = db.execute('SELECT route_id, frame_key, payload, reviewer FROM frame_reviews WHERE confirmed = 1 '
                          'ORDER BY route_id, frame_key').fetchall()
    finally:
        db.close()
    for route, frame_key, payload, reviewer in rows:
        if (reviewer or '').startswith('ai/'):
            continue
        frame = json.loads(payload)
        image = frame.get('image_path')
        if not image or not (data_root / image).exists():
            continue
        proposed = {p.get('detection_id'): p.get('bbox_xyxy') for p in frame.get('source_proposals') or []
                    if isinstance(p, dict)}
        for n, det in enumerate(frame.get('detections') or []):
            b = det.get('bbox_xyxy')
            family = det.get('family')
            if not isinstance(b, list) or len(b) != 4 or family in (None, 'not_a_sign') \
                    or b[2] - b[0] < 4 or b[3] - b[1] < 4:
                continue
            side = min(b[2] - b[0], b[3] - b[1])
            out.append({'annotation_id': f'{route}/{frame_key}/{n}', 'image_path': str(data_root / image),
                        'route': route, 'family': family,
                        'size': 'small' if side < 20 else ('mid' if side < 50 else 'big'),
                        'width': int(frame.get('width') or 1344), 'height': int(frame.get('height') or 760),
                        'truth': [float(v) for v in b], 'source': box_source(det, proposed)})
    return out


def box_source(det: dict, proposed: dict) -> str:
    """What last set the position of a confirmed box (see the module's description)."""
    source = det.get('box_source')
    if isinstance(source, str) and source:
        return source
    if det.get('detection_id') in proposed and proposed[det.get('detection_id')] == det.get('bbox_xyxy'):
        return 'model'
    return 'unknown'


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return 0.0 if u <= 0 else i / u


def cut_region(item):
    """The neighbourhood of a box (uint8 BGR) and its frame (x0, y0, scale); None when the image cannot be read."""
    import cv2
    img = cv2.imread(item['image_path'], cv2.IMREAD_COLOR)
    if img is None:
        return None
    t = item['truth']
    cx, cy = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
    side = REGION * max(t[2] - t[0], t[3] - t[1])
    k = min(1.0, MAX_REGION / side)
    n = int(math.ceil(side * k))
    x0, y0 = cx - side / 2, cy - side / 2
    m = np.float32([[k, 0, -x0 * k], [0, k, -y0 * k]])
    region = cv2.warpAffine(img, m, (n, n), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return region, (x0, y0, k)


def sample_rough(t, rng: random.Random) -> list[float]:
    """A rough box of ``t`` as a reviewer draws it or a pass of the refiner leaves it."""
    x1, y1, x2, y2 = t
    w, h = x2 - x1, y2 - y1
    r = rng.random()
    if r < 0.3:       # loose, sometimes looser than the benchmark
        return [x1 - rng.uniform(0, .6) * w, y1 - rng.uniform(0, .6) * h,
                x2 + rng.uniform(0, .6) * w, y2 + rng.uniform(0, .6) * h]
    if r < 0.65:      # shifted
        dx, dy = rng.uniform(-.35, .35) * w, rng.uniform(-.35, .35) * h
        return [x1 + dx + rng.uniform(-.2, .2) * w, y1 + dy + rng.uniform(-.2, .2) * h,
                x2 + dx + rng.uniform(-.2, .2) * w, y2 + dy + rng.uniform(-.2, .2) * h]
    return [x1 + rng.uniform(-.1, .1) * w, y1 + rng.uniform(-.1, .1) * h,       # tight, a later pass
            x2 + rng.uniform(-.1, .1) * w, y2 + rng.uniform(-.1, .1) * h]


def make_sample(region, frame, truth, rough, rng=None):
    """The S x S crop of ``rough`` taken from a cut region (as :func:`.refiner.crop` takes it from the frame), and the
    truth in crop units; augmented when ``rng`` is given."""
    import cv2
    x0r, y0r, k = frame
    cx0, cy0, side = refiner.crop_frame(rough)
    s = refiner.S / side
    a = s / k         # region pixel = (image - x0r) * k; crop pixel = (image - cx0) * s
    m = np.float32([[a, 0, (x0r - cx0) * s], [0, a, (y0r - cy0) * s]])
    crop = cv2.warpAffine(region, m, (refiner.S, refiner.S), flags=cv2.INTER_AREA if a < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    target = refiner.to_crop(truth, (cx0, cy0, side))
    if rng is not None:
        crop, target = augment(crop, target, rng)
    return crop, target


def augment(crop, target, rng: random.Random):
    import cv2
    img = crop.astype(np.float32)
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
        target = [1 - target[2], target[1], 1 - target[0], target[3]]
    img = img * rng.uniform(0.5, 1.4) + rng.uniform(-30, 30)                     # contrast, brightness (night)
    img = img * np.array([rng.uniform(.85, 1.15) for _ in range(3)], np.float32)  # colour cast
    if rng.random() < 0.3:
        g = img.mean(axis=2, keepdims=True)
        img = g + (img - g) * rng.uniform(0.4, 1.0)                               # washed-out colours
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.6))
    if rng.random() < 0.4:
        img = img + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 10), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 90)])
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return img, target


def network():
    import torch
    import torch.nn as nn

    class Refiner(nn.Module):
        def __init__(self):
            super().__init__()

            def block(i, o, s):
                return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
            self.body = nn.Sequential(block(3, 24, 1), block(24, 32, 2), block(32, 32, 1), block(32, 64, 2),
                                      block(64, 64, 1), block(64, 96, 2), block(96, 96, 1), block(96, 128, 2))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 16, 256), nn.ReLU(inplace=True),
                                      nn.Linear(256, 5))

        def forward(self, x):            # x: N x 3 x S x S, BGR / 255; out: the box in crop units, the rough box's IoU
            out = self.head(self.body(x - 0.45))
            return torch.cat([out[:, :4] + out.new_tensor([0.25, 0.25, 0.75, 0.75]), torch.sigmoid(out[:, 4:])], 1)
    return Refiner()


def giou_loss(p, t):
    import torch
    px1, py1, px2, py2 = p.unbind(1)
    tx1, ty1, tx2, ty2 = t.unbind(1)
    iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = iw * ih
    ap = (px2 - px1).clamp(min=1e-4) * (py2 - py1).clamp(min=1e-4)
    union = ap + (tx2 - tx1) * (ty2 - ty1) - inter
    cw = (torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(min=1e-4)
    ch = (torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(min=1e-4)
    c = cw * ch + 1e-6
    return (1 - inter / union + (c - union) / c).mean()


def _predict(model, crops):
    import torch
    with torch.no_grad():
        x = torch.from_numpy(np.stack(crops).transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        return model(x.to(next(model.parameters()).device)).cpu().numpy()


def validate(model, val) -> float:
    """Mean IoU after STEPS passes on the cut regions of the boxes left out (the epoch's score)."""
    model.eval()
    rough = [c['rough'] for _, _, c in val]
    for _ in range(refiner.STEPS):
        crops, frames = [], []
        for (region, frame, c), r in zip(val, rough):
            crops.append(make_sample(region, frame, c['truth'], r)[0])
            frames.append(refiner.crop_frame(r))
        preds = np.concatenate([_predict(model, crops[i:i + 512]) for i in range(0, len(crops), 512)])
        rough = [refiner.to_image(p, f) for p, f in zip(preds, frames)]
    model.train()
    return float(np.mean([iou(b, c['truth']) for (_, _, c), b in zip(val, rough)]))


# The two versions a worker of compare fits with: {'new': folder, 'in_use': folder or None}.
_FOLDERS: dict[str, Path | None] = {}
# What compare measures, each the mean IoU with the person's box and the share at 0.8 or more: the rough box, the new
# version's rankers alone, the new version as the review page runs it, the version in use as the review page runs it.
MEASURES = ('rough', 'rankers', 'version', 'in_use')


def _compare_one(case):
    import cv2
    from nnslr_tools import box_fit
    cv2.setNumThreads(1)
    try:
        img = box_fit.read_image(case['image_path'])
    except OSError:
        return None
    rough = box_fit.clip_box(case['rough'], img.shape[1], img.shape[0])
    if rough is None:
        return None
    new, in_use = _FOLDERS['new'], _FOLDERS.get('in_use')
    rankers = box_fit.fit_image(img, rough, use_refiner=False, folder=new) or rough
    # Each version as the review page runs it: under refiner.GATE its refiner (when it has one), then its rankers.
    version = box_fit.fit_image(img, rough, folder=new) or rough
    row = {'size': case['size'], 'kind': case['kind'], 'family': case['family'], 'source': case.get('source', 'unknown'),
           'rough': iou(rough, case['truth']), 'rankers': iou(rankers, case['truth']), 'version': iou(version, case['truth'])}
    if in_use is not None:
        row['in_use'] = iou(box_fit.fit_image(img, rough, folder=in_use) or rough, case['truth'])
    return row


def _use_folders(new: str, in_use: str | None) -> None:
    """A worker of compare: the folders of the new version and of the version in use."""
    _FOLDERS.update(new=Path(new), in_use=Path(in_use) if in_use else None)


def compare(cases, workers: int, new: Path, in_use: Path | None) -> dict:
    """MEASURES (mean IoU and share >= 0.8) of the version in ``new`` and of the one in ``in_use`` (None: not measured)
    on the same rough boxes, in all, by size, by kind and by source, and on the boxes Fit box did not place (fair)."""
    # Spawned, not forked: after the training the parent holds the threads of CUDA and torch, and a worker forked from it
    # can wait forever on a lock one of them held (11 Nov 2026, #65: the measure stood still for over an hour).
    with get_context('spawn').Pool(workers, initializer=_use_folders,
                                   initargs=(str(new), str(in_use) if in_use is not None else None)) as pool:
        rows = [r for r in pool.map(_compare_one, cases, chunksize=16) if r]
    groups = collections.defaultdict(list)
    for r in rows:
        for g in ('all', 'size:' + r['size'], 'kind:' + r['kind'], 'source:' + r['source']):
            groups[g].append(r)
        if r['source'] not in FIT_SOURCES:
            groups['fair'].append(r)
    keys = [k for k in MEASURES if k != 'in_use' or in_use is not None]
    return {g: {'cases': len(rs), **{k: round(float(np.mean([r[k] for r in rs])), 4) for k in keys},
                **{k + '>=0.8': round(float(np.mean([r[k] >= 0.8 for r in rs])), 4) for k in keys if k != 'rough'}}
            for g, rs in sorted(groups.items())}


def _version_in_use(data_root: Path, in_use_id: str | None) -> dict:
    """The version in use: the one named by ``in_use_id`` (the script read it at its start) while its folder is here,
    else the one in use now."""
    folder = versions.folder_of(data_root, in_use_id) if in_use_id else None
    if folder is not None:
        return {'id': in_use_id, 'folder': folder, 'problem': None}
    used = versions.in_use(data_root)
    return {'id': used['id'], 'folder': used['folder'], 'problem': used['problem']}


def measure_version(data_root: Path, out_dir: Path, cases, workers: int, summary: dict, log=print,
                    in_use_id: str | None = None) -> dict:
    """The version in ``out_dir`` against the version in use, on the same rough boxes left out: summary's 'in_use' (its
    id, and why the chosen one is not used, if so), 'left_out' (compare), 'clean' (which of the two ran from rankers
    trained without these routes) and 'better_than_in_use'."""
    used = _version_in_use(data_root, in_use_id)
    summary['in_use'] = {'id': used['id'], 'problem': used['problem']}
    measure_dir = out_dir / versions.MEASURE
    if (out_dir / refiner.MODEL.name).is_file() and measure_dir.is_dir():
        tmp = measure_dir / f'.{refiner.MODEL.name}.tmp'
        tmp.write_bytes((out_dir / refiner.MODEL.name).read_bytes())
        os.replace(tmp, measure_dir / refiner.MODEL.name)
    new, new_clean = versions.measure_folder(out_dir)
    old, old_clean = versions.measure_folder(used['folder'])
    summary['clean'] = {'version': new_clean, 'in_use': old_clean}
    log(f'measuring this version and the version in use ({used["id"]}) on the full frames left out'
        + ('' if new_clean and old_clean else ': ' + ' and '.join(
            name for name, clean in (('this version', new_clean), ('the version in use', old_clean)) if not clean)
           + ' trained on these routes too'))
    measure = compare(cases, workers, new, old)
    summary['left_out'] = measure
    summary['better_than_in_use'] = measure['all']['version'] > measure['all']['in_use']
    log(verdict(measure, used['id']))
    return summary


def run(data_root: Path, out_dir: Path, *, epochs: int, workers: int, dry_run: bool = False, log=print,
        device: str = 'cuda', measure_only: bool = False, in_use_id: str | None = None) -> dict:
    """Train the refiner into ``out_dir`` (a new version's folder, which already holds its rankers) and measure that
    version against the version in use; with ``measure_only`` only the measure."""
    from nnslr_tools.box_fit import train as rankers_train
    items = confirmed_boxes(data_root / 'annotations' / 'review.sqlite', data_root)
    train_items = [it for it in items if not held_out(it['route'])]
    val_items = [it for it in items if held_out(it['route'])]
    summary = {'boxes': len(items), 'train_boxes': len(train_items), 'held_out_boxes': len(val_items),
               'routes': len({it['route'] for it in items}), 'held_out_routes': len({it['route'] for it in val_items}),
               'by_size': dict(sorted(collections.Counter(it['size'] for it in items).items())),
               'by_source': dict(sorted(collections.Counter(it['source'] for it in items).items())),
               'hand_weight': HAND_WEIGHT,
               'min_boxes': MIN_BOXES, 'epochs': epochs, 'out': str(out_dir)}
    if dry_run:
        return {'dry_run': True, **summary}
    if not all((out_dir / name).is_file() for name in ('colour.npz', 'edges.npz')):
        raise ValueError(f'{out_dir} has no colour.npz and edges.npz: the rankers of the version are trained first')
    if measure_only:
        if not val_items:
            raise ValueError('no confirmed box on the routes left out: nothing to measure on')
        summary['refiner'] = 'carried' if (out_dir / refiner.MODEL.name).is_file() else 'none'
        started = time.time()
        measure_version(data_root, out_dir, rankers_train.make_cases(val_items), workers, summary, log, in_use_id)
        summary['minutes'] = round((time.time() - started) / 60, 1)
        return summary
    import torch
    import torch.nn.functional as F
    if len(train_items) < MIN_BOXES or not val_items:
        raise ValueError(f'only {len(train_items)} boxes to train on: too few for the refiner')
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, workers))
    started = time.time()
    with get_context('fork').Pool(workers) as pool:
        train_regions = pool.map(cut_region, train_items, chunksize=32)
        val_regions = pool.map(cut_region, val_items, chunksize=32)
    # A box placed by hand is drawn HAND_WEIGHT times as often: the same box, new rough boxes each time.
    regions = [(r[0], r[1], it['truth']) for r, it in zip(train_regions, train_items) if r is not None
               for _ in range(HAND_WEIGHT if it['source'] == 'hand' else 1)]
    cases = rankers_train.make_cases(val_items)
    region_of = {it['annotation_id']: r for it, r in zip(val_items, val_regions)}
    val = [(region_of[c['annotation_id']][0], region_of[c['annotation_id']][1], c) for c in cases
           if region_of[c['annotation_id']] is not None]
    log(f'{len(regions)} boxes to train on (by hand {HAND_WEIGHT} times), {len(val)} rough boxes left out, '
        f'cut in {time.time() - started:.0f} s')

    class Boxes(torch.utils.data.Dataset):
        def __init__(self, epoch):
            self.epoch = epoch

        def __len__(self):
            return len(regions) * PER_BOX

        def __getitem__(self, i):
            rng = random.Random(f'{SEED}:{self.epoch}:{i}')
            region, frame, truth = regions[i % len(regions)]
            rough = sample_rough(truth, rng)
            crop, target = make_sample(region, frame, truth, rough, rng)
            return (torch.from_numpy(crop.transpose(2, 0, 1).astype(np.float32) / 255.0),
                    torch.tensor([*target, iou(rough, truth)], dtype=torch.float32))

    model = network().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = epochs * math.ceil(len(regions) * PER_BOX / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total_steps, pct_start=0.1)
    best, best_state, scores = -1.0, None, []
    for epoch in range(epochs):
        loader = torch.utils.data.DataLoader(Boxes(epoch), batch_size=BATCH, shuffle=True, num_workers=min(12, workers),
                                             pin_memory=True,
                                             multiprocessing_context='fork',
                                             generator=torch.Generator().manual_seed(SEED + epoch))
        t, total, n = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            p = model(x)
            box, right = p[:, :4], y[:, :4]
            loss = F.smooth_l1_loss(box, right, beta=0.02) * 10 + giou_loss(box, right) + F.l1_loss(p[:, 4], y[:, 4]) * 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(x), n + len(x)
        score = validate(model, val)
        scores.append(round(score, 4))
        log(f'epoch {epoch + 1}/{epochs} loss {total / n:.4f} {time.time() - t:.0f} s, mean IoU left out {score:.4f}')
        if score > best:
            best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    summary['epoch_scores'] = scores
    summary['best_epoch'] = scores.index(max(scores)) + 1

    left_out = [c for c in cases if region_of[c['annotation_id']] is not None]
    # The refiner goes into the new version's folder in one step: the version is the rankers and this refiner together.
    tmp = out_dir / f'.{refiner.MODEL.name}.tmp'
    export(model, tmp)
    os.replace(tmp, out_dir / refiner.MODEL.name)
    summary['refiner'] = 'trained'
    measure_version(data_root, out_dir, left_out, workers, summary, log, in_use_id)
    summary['minutes'] = round((time.time() - started) / 60, 1)
    return summary


def verdict(measure: dict, in_use_id: str) -> str:
    """The outcome in one sentence for the log: the new version's mean IoU on the boxes left out as the review page runs
    it, against the version in use on the same boxes and its own rankers alone. Nothing is replaced either way."""
    new, used, rankers = measure['all']['version'], measure['all']['in_use'], measure['all']['rankers']
    word = 'better than' if new > used else 'as good as' if new == used else 'worse than'
    return (f'This version: mean IoU {new:.4f} on the {measure["all"]["cases"]} boxes left out as the review page runs it, '
            f'{word} the version in use ({in_use_id}) {used:.4f} on the same boxes; its rankers alone {rankers:.4f}. '
            'Nothing replaced: the version in use is chosen in Models › Review helpers › Fit box.')


def export(model, path: Path) -> None:
    """The network as one ONNX file (weights inside), input "crop" N x 3 x S x S, output "box" N x 5 (the box in crop units, the rough box's IoU)."""
    import onnx
    import torch
    torch.onnx.export(model, torch.zeros(1, 3, refiner.S, refiner.S), str(path), opset_version=17,
                      input_names=['crop'], output_names=['box'], dynamic_axes={'crop': {0: 'n'}, 'box': {0: 'n'}},
                      dynamo=False)
    onnx.save(onnx.load(str(path)), str(path), save_as_external_data=False)


def main(argv=None) -> int:
    from nnslr_tools.preannotate import resolve_data_root
    parser = argparse.ArgumentParser(prog='python -m nnslr_tools.box_fit.refiner_train',
                                     description='train the box refiner of Fit box on the boxes people confirmed')
    parser.add_argument('--data-root', help='default: NNSLR_DATA_ROOT or <repo>/data')
    parser.add_argument('--out', required=True,
                        help='the folder of the new version, which holds its rankers (colour.npz, edges.npz)')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--workers', type=int, default=max(1, min(12, (os.cpu_count() or 2) // 2)))
    parser.add_argument('--device', default='cuda',
                        help='the card (default cuda, the one CUDA_VISIBLE_DEVICES gives); never the CPU')
    parser.add_argument('--dry-run', action='store_true', help='only count the boxes and the routes left out')
    parser.add_argument('--measure-only', action='store_true',
                        help='train nothing: measure the version in --out against the version in use')
    parser.add_argument('--in-use', help='the version in use the training started with (box_fit:...); default: the one '
                                         'in use now')
    args = parser.parse_args(argv)
    if not args.dry_run and not args.measure_only:
        import torch
        if not args.device.startswith('cuda') or not torch.cuda.is_available():
            print(f'No CUDA card for {args.device!r} (CUDA_VISIBLE_DEVICES, nvidia-smi): the refiner trains on the GPU '
                  'only. Nothing changed.', file=sys.stderr)
            return 1
    summary = run(resolve_data_root(args.data_root), Path(args.out), epochs=args.epochs, workers=args.workers,
                  dry_run=args.dry_run, device=args.device, measure_only=args.measure_only, in_use_id=args.in_use,
                  log=lambda line: print(line, file=sys.stderr, flush=True))
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
