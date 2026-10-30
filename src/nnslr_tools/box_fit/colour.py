"""The colour fit: candidate boxes from colour masks, grabCut and edge snapping around the rough box, scored by a
small tree model (:mod:`.gbdt`), the best ones averaged.

The crop around the rough box (``PAD`` of its longer side on each side) is scaled to ``WORK`` pixels. Candidates:

- the connected components of Lab colour masks (red, blue, yellow, bright, dark, far from the colour of the ring
  around the box), with the union of the ones whose centre is inside the rough box (``_u``);
- grabCut started from the rough box shrunk, as it is and grown (``gcs``, ``gc``, ``gcx``), and the masks opened to
  cut poles and thin bridges (``...O``);
- the rough box (and a grown one) with each side snapped inward to the first strong colour edge (``snap``,
  ``snapx``), and the colour candidates with each side grown out to the last strong edge (``...G``: dim outer rings);
- the rough box itself, always the last.

:func:`features` describes each candidate (geometry against the rough box, mask fill, colour contrast with the ring
around it, edge strength on its border, symmetry, solidity, how many other sources agree...), the model predicts its
IoU with the sign, and :func:`fit` returns None when the rough box itself scores best.
"""
from __future__ import annotations

import cv2
import numpy as np

from nnslr_tools.box_fit import gbdt

# Human boxes sit about 1 px and 2 % outside the sign's edge: every candidate (not the rough box) is grown so.
EXP_A, EXP_F = 0.8, 0.02
# The crop around the rough box, as a share of its longer side; its longer side scaled to WORK pixels.
PAD = 0.45
WORK = 80
GRABCUT_SEED = 0
# grabCut from the rough box grown by these shares of its sides (negative: shrunk).
GC_RECTS = [('gcs', -0.08), ('gc', 0.0), ('gcx', 0.15)]
# The masks also opened (poles and thin bridges cut).
OPEN_SRC = ['gc', 'gcx', 'bgdist', 'bright']
# The rough box grown by a share of its sides, then each side snapped inward to the first edge at least `keep` of the
# strongest.
SNAPS = [(0.0, 0.5, 'snap'), (0.15, 0.5, 'snapx')]
# The candidates also grown out to the last strong edge.
GROW_SRC = ('gc', 'gcx', 'gcs', 'red', 'blue', 'bright', 'bgdist', 'gcO', 'gcxO')
SOURCES = (['red', 'blue', 'yellow', 'bright', 'dark', 'bgdist'] + [g for g, _ in GC_RECTS] + [s + 'O' for s in OPEN_SRC]
           + [n for _, _, n in SNAPS] + ['rough'])
# Softmax over the candidates within DELTA of the best score, at temperature TAU: the answer is their mean box.
TAU, DELTA = 0.02, 0.1
K3 = np.ones((3, 3), np.uint8)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return 0.0 if u <= 0 else i / u


# ---------------------------------------------------------------- candidates

def _crop(img, rough):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = rough
    w, h = max(2.0, x2 - x1), max(2.0, y2 - y1)
    p = PAD * max(w, h) + 3
    cx1, cy1 = int(max(0, np.floor(x1 - p))), int(max(0, np.floor(y1 - p)))
    cx2, cy2 = int(min(W, np.ceil(x2 + p))), int(min(H, np.ceil(y2 + p)))
    crop = img[cy1:cy2, cx1:cx2]
    s = WORK / max(crop.shape[:2])
    work = cv2.resize(crop, (max(8, round(crop.shape[1] * s)), max(8, round(crop.shape[0] * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    sx = work.shape[1] / crop.shape[1]
    sy = work.shape[0] / crop.shape[0]
    return work, (cx1, cy1, sx, sy)


def _components(mask, rb, src, out, min_frac=0.02):
    """The components of a binary mask that overlap the rough box ``rb``, as candidates (work pixels)."""
    n, _, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return
    rx1, ry1, rx2, ry2 = rb
    ra = (rx2 - rx1) * (ry2 - ry1)
    keep = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_frac * ra:
            continue
        ix = max(0, min(x + w, rx2) - max(x, rx1))
        iy = max(0, min(y + h, ry2) - max(y, ry1))
        if ix * iy < 0.15 * w * h:
            continue
        keep.append(i)
        out.append((src, [x, y, x + w, y + h], a))
    # The union of the components whose centre is inside the rough box.
    inside = [i for i in keep if rx1 <= cen[i][0] <= rx2 and ry1 <= cen[i][1] <= ry2]
    if len(inside) >= 2:
        xs1 = min(st[i][0] for i in inside)
        ys1 = min(st[i][1] for i in inside)
        xs2 = max(st[i][0] + st[i][2] for i in inside)
        ys2 = max(st[i][1] + st[i][3] for i in inside)
        out.append((src + '_u', [xs1, ys1, xs2, ys2], sum(st[i][4] for i in inside)))


def _masks(work, rb):
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, A, B = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    h, w = L.shape
    rx1, ry1, rx2, ry2 = [int(round(v)) for v in rb]
    ring = np.ones((h, w), bool)
    ring[max(0, ry1):ry2, max(0, rx1):rx2] = False
    if ring.sum() < 10:
        ring[:] = True
    Lb = np.median(L[ring])
    inside = (slice(max(0, ry1), ry2), slice(max(0, rx1), rx2))

    def threshold(ch, fixed, lo):
        v = ch[inside]
        return max(lo, min(fixed, 0.5 * float(np.percentile(v, 98)))) if v.size else fixed

    m = {}
    m['red'] = (A > threshold(A, 12, 5)) & (A > np.abs(B) * 0.5 + 3)
    m['blue'] = (-B > threshold(-B, 12, 6)) & (A < 10)
    m['yellow'] = (B > threshold(B, 25, 10)) & (A > -10)
    # Well above the lightness of the background: LED displays and white plates.
    inner = L[inside]
    hi = np.percentile(inner, 90) if inner.size else 255
    m['bright'] = L > max(Lb + 25, 0.5 * (Lb + hi))
    m['dark'] = L < Lb - 30
    # Far from the colour of the ring around the box.
    bgc = np.median(lab[ring], axis=0)
    d = np.sqrt(((lab - bgc) ** 2).sum(-1)).astype(np.float32)
    d8 = np.clip(d * 2, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m['bgdist'] = d8 > max(t, 30)
    return {k: cv2.morphologyEx(v.astype(np.uint8), cv2.MORPH_CLOSE, K3) for k, v in m.items()}, d


def _grabcut(work, rect, iters=3):
    h, w = work.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in rect]
    x1, y1 = max(1, x1), max(1, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    m = np.zeros((h, w), np.uint8)
    bg = np.zeros((1, 65))
    fg = np.zeros((1, 65))
    # grabCut starts its colour models with k-means on OpenCV's random generator (one per thread): seeded, the same
    # box always gets the same fit, whatever ran before on this thread.
    cv2.setRNGSeed(GRABCUT_SEED)
    try:
        cv2.grabCut(work, m, (x1, y1, x2 - x1, y2 - y1), bg, fg, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    return ((m == 1) | (m == 3)).astype(np.uint8)


def _colour_gradient(work):
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    gx = np.zeros(work.shape[:2], np.float32)
    gy = np.zeros_like(gx)
    for c, weight in ((0, 1.0), (1, 1.5), (2, 1.5)):
        gx += weight * np.abs(cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3))
        gy += weight * np.abs(cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3))
    return gx, gy


def _profile(gx, gy, b, side, ds):
    """Mean perpendicular gradient along the middle 60 % of a side, at offsets ds (positive = outward)."""
    h, w = gx.shape
    x1, y1, x2, y2 = b
    out = []
    if side in (0, 2):
        a, z = int(y1 + 0.2 * (y2 - y1)), int(np.ceil(y2 - 0.2 * (y2 - y1)))
        a, z = max(0, a), min(h, max(z, a + 1))
        for d in ds:
            x = int(round(x1 - d)) if side == 0 else int(round(x2 - 1 + d))
            out.append(gx[a:z, x].mean() if 0 <= x < w else -1.0)
    else:
        a, z = int(x1 + 0.2 * (x2 - x1)), int(np.ceil(x2 - 0.2 * (x2 - x1)))
        a, z = max(0, a), min(w, max(z, a + 1))
        for d in ds:
            y = int(round(y1 - d)) if side == 1 else int(round(y2 - 1 + d))
            out.append(gy[y, a:z].mean() if 0 <= y < h else -1.0)
    return np.array(out)


def _grow(gx, gy, b, frac=0.35, keep=0.6):
    """Each side moved outward to the outermost strong edge within frac of the side (dim outer rings)."""
    nb = list(b)
    for side in range(4):
        L = (b[2] - b[0]) if side in (0, 2) else (b[3] - b[1])
        ds = np.arange(0, int(frac * L) + 2 + 1)
        p = _profile(gx, gy, b, side, ds)
        if p.max() <= 0:
            continue
        d = ds[np.nonzero(p >= keep * p.max())[0][-1]]
        nb[side] = b[side] - d if side < 2 else b[side] + d
    return nb


def _snap(gx, gy, b, frac=0.4, keep=0.5):
    """Each side moved inward to the first strong edge within frac of the side (rough box -> sign edge)."""
    nb = list(b)
    for side in range(4):
        L = (b[2] - b[0]) if side in (0, 2) else (b[3] - b[1])
        ds = -np.arange(0, int(frac * L) + 1 + 1)
        p = _profile(gx, gy, b, side, ds)
        if p.max() <= 0:
            continue
        d = ds[np.nonzero(p >= keep * p.max())[0][0]]
        nb[side] = b[side] - d if side < 2 else b[side] + d
    if nb[2] - nb[0] < 2 or nb[3] - nb[1] < 2:
        return None
    return nb


def _edge_candidates(work, rb, raw):
    gx, gy = _colour_gradient(work)
    rw, rh = rb[2] - rb[0], rb[3] - rb[1]
    new = []
    for f, keep, name in SNAPS:
        s = _snap(gx, gy, [rb[0] - f * rw, rb[1] - f * rh, rb[2] + f * rw, rb[3] + f * rh], keep=keep)
        if s is not None:
            new.append((name, s, 0))
    for src, b, a in raw:
        if src in GROW_SRC:
            g = _grow(gx, gy, b)
            if g != list(b):
                new.append((src + 'G', g, a))
    raw.extend(new)


def candidates(img, rough):
    """(work crop, rough box in it, masks, colour distance, [(source, box, area)], to_frame) around ``rough``."""
    work, (ox, oy, sx, sy) = _crop(img, rough)
    rb = [(rough[0] - ox) * sx, (rough[1] - oy) * sy, (rough[2] - ox) * sx, (rough[3] - oy) * sy]
    masks, dist = _masks(work, rb)
    raw = []
    for k, mask in masks.items():
        _components(mask, rb, k, raw)
    rw, rh = rb[2] - rb[0], rb[3] - rb[1]
    for name, f in GC_RECTS:
        g = _grabcut(work, [rb[0] - f * rw, rb[1] - f * rh, rb[2] + f * rw, rb[3] + f * rh])
        if g is not None:
            masks[name] = g
            _components(g, rb, name, raw)
    k = max(3, int(round(min(rw, rh) / 6)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for name in OPEN_SRC:
        if name in masks:
            o = cv2.morphologyEx(masks[name], cv2.MORPH_OPEN, kernel)
            masks[name + 'O'] = o
            _components(o, rb, name + 'O', raw)
    _edge_candidates(work, rb, raw)
    raw.append(('rough', rb, 0))

    def to_frame(b, expand=True):
        f = [b[0] / sx + ox, b[1] / sy + oy, b[2] / sx + ox, b[3] / sy + oy]
        if expand:
            ex = EXP_A + EXP_F * (f[2] - f[0])
            ey = EXP_A + EXP_F * (f[3] - f[1])
            f = [f[0] - ex, f[1] - ey, f[2] + ex, f[3] + ey]
        return f

    return work, rb, masks, dist, raw, to_frame


# ---------------------------------------------------------------- features

def features(work, rb, masks, dist, raw, rough_frame):
    """One row of 50 features per candidate of :func:`candidates`."""
    h, w = dist.shape
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy)
    gmn = gm / (gm.mean() + 1e-3)
    dn = dist / (dist.mean() + 1e-3)
    rw, rh = rb[2] - rb[0], rb[3] - rb[1]
    rcx, rcy = (rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2
    rshort = np.log(max(2.0, min(rough_frame[2] - rough_frame[0], rough_frame[3] - rough_frame[1])))
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = []
    for src, b, area in raw:
        grown = src.endswith('G')
        base = src.removesuffix('G').removesuffix('_u')
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, max(x2, x1 + 1)), min(h, max(y2, y1 + 1))
        bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
        M = masks.get(base)
        fill = float(M[y1:y2, x1:x2].mean()) if M is not None else 0.0
        mtot = float(M.sum()) if M is not None else 0.0
        mout = (mtot - M[y1:y2, x1:x2].sum()) / max(1.0, bw * bh) if M is not None else 0.0
        # Colour distance inside and on a ring around the box.
        mg = max(1, int(round(0.15 * min(bw, bh))) + 1)
        X1, Y1, X2, Y2 = max(0, x1 - mg), max(0, y1 - mg), min(w, x2 + mg), min(h, y2 + mg)
        inn = dn[y1:y2, x1:x2]
        outer = dn[Y1:Y2, X1:X2]
        ring_sum = outer.sum() - inn.sum()
        ring_n = outer.size - inn.size
        cin = float(inn.mean())
        cring = float(ring_sum / ring_n) if ring_n > 0 else cin
        # Gradient on a thin band at the box edge, and inside.
        e1 = max(1, mg // 2)
        gin = gmn[Y1:Y2, X1:X2]
        core = gmn[min(h, y1 + e1):max(0, y2 - e1), min(w, x1 + e1):max(0, x2 - e1)]
        band = (gin.sum() - core.sum()) / max(1, gin.size - core.size)
        gcore = float(core.mean()) if core.size else 0.0
        touch = float(x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h)
        support = 0
        for s2, b2, _ in raw:
            if s2.removesuffix('_u') != base and s2 != 'rough' and _iou(b, b2) > 0.75:
                support += 1
        onehot = [float(base == s) for s in SOURCES]
        if M is not None and y2 - y1 >= 2 and x2 - x1 >= 2:
            mb = M[y1:y2, x1:x2].astype(np.float32)
            slr = 1 - float(np.abs(mb - mb[:, ::-1]).mean())
            sud = 1 - float(np.abs(mb - mb[::-1]).mean())
            cs, _ = cv2.findContours(M[y1:y2, x1:x2].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                c = max(cs, key=cv2.contourArea)
                ca = cv2.contourArea(c)
                ha = cv2.contourArea(cv2.convexHull(c))
                sol = ca / ha if ha > 0 else 0.0
                ext = ca / max(1.0, (x2 - x1) * (y2 - y1))
            else:
                sol = ext = 0.0
        else:
            slr = sud = sol = ext = 0.0
        li = lab[y1:y2, x1:x2].reshape(-1, 3)
        lo = lab[Y1:Y2, X1:X2].reshape(-1, 3)
        mi = li.mean(0)
        mo = (lo.sum(0) - li.sum(0)) / max(1, len(lo) - len(li)) if len(lo) > len(li) else mi
        colour = [mi[0] / 255, (mi[1] - 128) / 30, (mi[2] - 128) / 30, float(li[:, 0].std()) / 50,
                  (mi[0] - mo[0]) / 50, float(np.abs(mi[1:] - mo[1:]).sum()) / 30, (x2 - x1) * (y2 - y1) / float(w * h)]
        dx = (b[0] + b[2] - 2 * rcx) / 2 / rw
        dy = (b[1] + b[3] - 2 * rcy) / 2 / rh
        out.append([1.0, _iou(b, rb), np.log(bw / rw), np.log(bh / rh), dx, dy,
                    abs(np.log(bw / rw)), abs(np.log(bh / rh)), abs(dx), abs(dy),
                    abs(np.log(bw / bh)), fill, min(mout, 3.0), cin, cin - cring, float(band), gcore, touch,
                    min(support, 6) / 3.0, rshort, float(src.endswith('_u')), float(grown),
                    np.log1p(area / (rw * rh)), slr, sud, sol, ext] + colour + onehot)
    return np.array(out, np.float32)


def model_rows(F):
    """The rows the model sees: each candidate's features and their difference from the rough box's (the last)."""
    return np.concatenate([F, F - F[-1:]], 1)


def extract(img, rough):
    """Candidates of ``rough`` in pixels of the frame (the rough box itself last, not grown) and their model rows."""
    work, rb, masks, dist, raw, to_frame = candidates(img, rough)
    F = features(work, rb, masks, dist, raw, rough)
    boxes = np.array([to_frame(b) if s != 'rough' else list(rough) for s, b, _ in raw], float)
    return boxes, model_rows(F)


def pick(boxes, X, model, width, height) -> list[float] | None:
    """The model's answer among the candidates ``boxes`` (rows ``X``, the rough box last) on a ``width`` x
    ``height`` image: the best ones averaged, or None when the rough box itself scores best."""
    p = gbdt.predict(model, X)
    j = int(np.argmax(p))
    if j == len(p) - 1:
        return None
    sel = np.nonzero(p >= p[j] - DELTA)[0]
    weights = np.exp((p[sel] - p[j]) / TAU)
    b = (weights[:, None] * boxes[sel]).sum(0) / weights.sum()
    b = [max(0.0, b[0]), max(0.0, b[1]), min(float(width), b[2]), min(float(height), b[3])]
    return b if b[2] - b[0] >= 1 and b[3] - b[1] >= 1 else None


def fit(img, rough, model) -> list[float] | None:
    """The colour fit of ``rough`` (pixels of ``img``), or None when the model ranks the rough box first."""
    boxes, X = extract(img, rough)
    return pick(boxes, X, model, img.shape[1], img.shape[0])
