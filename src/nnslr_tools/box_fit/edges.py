"""The edge fit: candidate boxes from edges, MSER regions and colour contrast, ranked by a tiny listwise MLP.

1. The rough box and ``PAD`` of its longer side around it are cut out and scaled so its longer side is ``S`` pixels.
2. Candidates (source numbers in brackets): the bounding boxes of the contours of a colour Canny (L, a, b OR-ed),
   raw and closed by 3 and 5 (0, 1, 2); MSER regions on L (3) and on chroma, a and b (4); the components of the
   colour distance to the crop's border, Otsu-thresholded, raw and opened (5, 6); the rough box shrunk by 0, 6 and
   12 % (7, 8, 9); and the image candidates with each side snapped to the nearest gradient-profile peak (10).
3. Per candidate 56 cheap features from integral images (:func:`features`): edge support of each side,
   gradient-profile value, inner/outer band colour contrast, calm outer ring, colour against the background, geometry
   against the rough box, source, agreement between candidates, containment (inner part of a larger candidate).
4. Three MLPs (56-32-1, trained listwise by :func:`train_ranker`) score them; the answer is the softmax-weighted mean
   box (temperature ``TEMP``), pulled back toward the rough box when it is already close to it (IoU above
   ``BLEND_LO``): a safety for boxes the reviewer drew well.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

S, PAD, SIG, SNAP, TEMP = 64, 0.55, 0.8, 0.10, 0.6
BLEND_LO = 0.7
NSRC = 12
# The sources of the rough box shrunk (7, 8, 9) and of the snapped candidates (10); below 7 the image's own.
ROUGH_SRC = 7
SNAPPED_SRC = 10
ENSEMBLE = 3
# One MSER per thread: OpenCV's MSER keeps its work buffers in the object and runs without the GIL, so two threads
# sharing one (two fits at once on the review server) crash the process.
_local = threading.local()


def _mser():
    mser = getattr(_local, 'mser', None)
    if mser is None:
        mser = _local.mser = cv2.MSER_create(delta=4, min_area=12, max_area=20000, max_variation=0.5)
    return mser


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return 0.0 if u <= 0 else i / u


def _prep(frame, rough):
    x1, y1, x2, y2 = rough
    w, h = x2 - x1, y2 - y1
    m = PAD * max(w, h) + 2
    H, W = frame.shape[:2]
    X1, Y1 = max(0, int(np.floor(x1 - m))), max(0, int(np.floor(y1 - m)))
    X2, Y2 = min(W, int(np.ceil(x2 + m))), min(H, int(np.ceil(y2 + m)))
    sub = frame[Y1:Y2, X1:X2]
    k = S / max(w, h, 1.0)
    nw, nh = max(4, int(round(sub.shape[1] * k))), max(4, int(round(sub.shape[0] * k)))
    kx, ky = nw / sub.shape[1], nh / sub.shape[0]
    img = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    r = [(x1 - X1) * kx, (y1 - Y1) * ky, (x2 - X1) * kx, (y2 - Y1) * ky]
    return img, (kx, ky), (X1, Y1), r


def _box_sum(I, x1, y1, x2, y2, H, W):
    """Mean of the integral image ``I`` over each box (clipped to the image), and whether the box is not empty."""
    x1 = np.clip(np.round(x1), 0, W).astype(int)
    x2 = np.clip(np.round(x2), 0, W).astype(int)
    y1 = np.clip(np.round(y1), 0, H).astype(int)
    y2 = np.clip(np.round(y2), 0, H).astype(int)
    a = np.maximum((x2 - x1) * (y2 - y1), 1)
    s = I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]
    return s / (a[:, None] if s.ndim == 2 else a), (x2 > x1) & (y2 > y1)


# ---------------------------------------------------------------- candidates

def _component_boxes(mask, src, out, min_area, max_area):
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        if min_area <= w * h <= max_area and w >= 3 and h >= 3:
            out.append((x, y, x + w, y + h, src))


def _image_candidates(img, r):
    H, W = img.shape[:2]
    rw, rh = r[2] - r[0], r[3] - r[1]
    min_area, max_area = 0.08 * rw * rh, 2.6 * rw * rh
    out = []
    lab8 = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    blur = cv2.GaussianBlur(lab8, (0, 0), 0.8)
    gray = blur[..., 0]
    edges = np.zeros((H, W), np.uint8)
    for ch in range(3):
        c = blur[..., ch]
        gx = cv2.Sobel(c, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(c, cv2.CV_32F, 0, 1)
        mg = np.sqrt(gx * gx + gy * gy)
        hi = max(np.percentile(mg, 85), 20 if ch == 0 else 12)
        edges |= cv2.Canny(c, 0.5 * hi, hi, L2gradient=True)
    for ks, src in ((1, 0), (3, 1), (5, 2)):
        e = edges if ks == 1 else cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
        contours, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if min_area <= w * h <= max_area and w >= 3 and h >= 3:
                out.append((x, y, x + w, y + h, src))
    chroma = np.sqrt((blur[..., 1].astype(np.float32) - 128) ** 2 + (blur[..., 2].astype(np.float32) - 128) ** 2)
    chroma8 = np.clip(chroma * 3, 0, 255).astype(np.uint8)
    for im, src in ((gray, 3), (chroma8, 4), (blur[..., 1], 4), (blur[..., 2], 4)):
        _, regions = _mser().detectRegions(im)
        for (x, y, w, h) in regions:
            if min_area <= w * h <= max_area and w >= 3 and h >= 3:
                out.append((x, y, x + w, y + h, src))
    # The colour distance to the crop's border (its background), Otsu-thresholded.
    labf = blur.astype(np.float32)
    border = np.concatenate([labf[:2].reshape(-1, 3), labf[-2:].reshape(-1, 3), labf[:, :2].reshape(-1, 3),
                             labf[:, -2:].reshape(-1, 3)])
    dist = np.sqrt(((labf - np.median(border, 0)) ** 2).sum(-1))
    d8 = np.clip(dist * (255.0 / max(dist.max(), 1)), 0, 255).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _component_boxes(m, 5, out, min_area, max_area)
    _component_boxes(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), 6, out, min_area, max_area)
    for s, src in ((0.0, 7), (0.06, 8), (0.12, 9)):
        out.append((r[0] + s * rw, r[1] + s * rh, r[2] - s * rw, r[3] - s * rh, src))
    keep = []
    for b in out:
        # Near-duplicates of an image candidate dropped; the rough-derived ones always kept.
        if b[4] >= ROUGH_SRC or all(_iou(b, k) < 0.97 or k[4] >= ROUGH_SRC for k in keep):
            keep.append(b)
    return keep, edges


def _maps(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), SIG)
    gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1)
    ax = np.sqrt((gx ** 2).sum(2))
    ay = np.sqrt((gy ** 2).sum(2))
    gs = float(np.percentile(np.maximum(ax, ay), 90)) + 1.0
    return lab, ax / gs, ay / gs, gs


def _snap(profile, pos, radius, offset):
    lo = int(max(0, np.floor(pos - radius)))
    hi = int(min(len(profile) - 1, np.ceil(pos + radius)))
    if hi <= lo:
        return pos
    return float(lo + int(np.argmax(profile[lo:hi + 1])) + offset)


def candidates(img, r):
    """(boxes N x 4, sources, maps, edges) in the pixels of the prepared crop ``img``."""
    found, edges = _image_candidates(img, r)
    H, W = img.shape[:2]
    lab, ax, ay, gs = _maps(img)
    out = list(found)
    for (x1, y1, x2, y2, s) in found:
        if s >= ROUGH_SRC:
            continue
        w, h = x2 - x1, y2 - y1
        ya, yb = int(max(0, y1 + .15 * h)), int(min(H, y2 - .15 * h)) + 1
        xa, xb = int(max(0, x1 + .15 * w)), int(min(W, x2 - .15 * w)) + 1
        if yb - ya < 2 or xb - xa < 2:
            continue
        px = ax[ya:yb].mean(0)
        py = ay[:, xa:xb].mean(1)
        b = (_snap(px, x1, SNAP * w, 0), _snap(py, y1, SNAP * h, 0), _snap(px, x2 - 1, SNAP * w, 1),
             _snap(py, y2 - 1, SNAP * h, 1), SNAPPED_SRC)
        if b[2] - b[0] >= 3 and b[3] - b[1] >= 3:
            out.append(b)
    rw, rh = r[2] - r[0], r[3] - r[1]
    ya, yb = int(max(0, r[1] + .1 * rh)), int(min(H, r[3] - .1 * rh)) + 1
    xa, xb = int(max(0, r[0] + .1 * rw)), int(min(W, r[2] - .1 * rw)) + 1
    maps = dict(lab=lab, ax=ax, ay=ay, gs=gs,
                px=np.convolve(ax[ya:yb].mean(0), [.25, .5, .25], 'same'),
                py=np.convolve(ay[:, xa:xb].mean(1), [.25, .5, .25], 'same'))
    G = np.array([b[:4] for b in out], np.float64)
    src = np.array([b[4] for b in out])
    return G, src, maps, edges


# ---------------------------------------------------------------- features

def features(img, r, g, src, maps, edges):
    """One row of 56 features per candidate box of ``g``."""
    H, W = img.shape[:2]
    lab, ax, ay, px, py = maps['lab'], maps['ax'], maps['ay'], maps['px'], maps['py']
    il = cv2.integral(lab)
    iax = cv2.integral(ax)
    iay = cv2.integral(ay)
    x1, y1, x2, y2 = g.T
    w = x2 - x1
    h = y2 - y1
    rw, rh = r[2] - r[0], r[3] - r[1]
    rcx, rcy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2

    def bs(I, a, b, c, d):
        return _box_sum(I, a, b, c, d, H, W)

    # The edge strength on each side, and the gradient profile of the rough box there.
    el, _ = bs(iax, x1 - 1, y1, x1 + 1, y2)
    er, _ = bs(iax, x2 - 1, y1, x2 + 1, y2)
    et, _ = bs(iay, x1, y1 - 1, x2, y1 + 1)
    eb, _ = bs(iay, x1, y2 - 1, x2, y2 + 1)
    E = np.stack([el, er, et, eb], 1)
    pv = np.stack([px[np.clip(x1.astype(int), 0, W - 1)], px[np.clip(x2.astype(int) - 1, 0, W - 1)],
                   py[np.clip(y1.astype(int), 0, H - 1)], py[np.clip(y2.astype(int) - 1, 0, H - 1)]], 1)
    pv = pv / np.r_[px.max(), px.max(), py.max(), py.max()][None] if px.max() > 0 and py.max() > 0 else pv
    # Colour contrast between a band inside and a band outside each side.
    bw = np.maximum(1.0, 0.15 * np.minimum(w, h))
    cs = []
    for (ia, oa) in (((x1, y1, x1 + bw, y2), (x1 - bw, y1, x1, y2)), ((x2 - bw, y1, x2, y2), (x2, y1, x2 + bw, y2)),
                     ((x1, y1, x2, y1 + bw), (x1, y1 - bw, x2, y1)), ((x1, y2 - bw, x2, y2), (x1, y2, x2, y2 + bw))):
        a, oka = bs(il, *ia)
        b, okb = bs(il, *oa)
        cs.append(np.where(oka & okb, np.sqrt(((a - b) ** 2).sum(1)) / 40.0, 0.0))
    Cs = np.stack(cs, 1)
    # Gradient on a ring outside the box (calm around a sign) and inside it.
    ob = np.maximum(2.0, 0.25 * np.minimum(w, h))
    gout_tot, _ = bs(iax + iay, x1 - ob, y1 - ob, x2 + ob, y2 + ob)
    gin, _ = bs(iax + iay, x1, y1, x2, y2)
    A_out = (np.clip(x2 + ob, 0, W) - np.clip(x1 - ob, 0, W)) * (np.clip(y2 + ob, 0, H) - np.clip(y1 - ob, 0, H))
    A_in = w * h
    gout = (gout_tot * A_out - gin * A_in) / np.maximum(A_out - A_in, 1)
    border = np.concatenate([lab[:2].reshape(-1, 3), lab[-2:].reshape(-1, 3), lab[:, :2].reshape(-1, 3),
                             lab[:, -2:].reshape(-1, 3)])
    mi, _ = bs(il, x1, y1, x2, y2)
    din = np.sqrt(((mi - np.median(border, 0)) ** 2).sum(1)) / 40.0
    ix = np.clip(np.minimum(x2, r[2]) - np.maximum(x1, r[0]), 0, None)
    iy = np.clip(np.minimum(y2, r[3]) - np.maximum(y1, r[1]), 0, None)
    inter = ix * iy
    iou_rough = inter / (w * h + rw * rh - inter)
    geo = np.stack([iou_rough, ((x1 + x2) / 2 - rcx) / rw, ((y1 + y2) / 2 - rcy) / rh, np.log(w / rw), np.log(h / rh),
                    np.log(w / h), np.abs(np.log(w / h)),
                    (x1 - r[0]) / rw, (y1 - r[1]) / rh, (r[2] - x2) / rw, (r[3] - y2) / rh], 1)
    whole = np.stack([np.full(len(g), maps['gs'] / 50.0), np.full(len(g), np.log(rw * rh))], 1)
    F = np.hstack([np.ones((len(g), 1)), geo, E, E.mean(1, keepdims=True), E.min(1, keepdims=True), pv,
                   pv.mean(1, keepdims=True), pv.min(1, keepdims=True), Cs, Cs.mean(1, keepdims=True),
                   Cs.min(1, keepdims=True), gout[:, None], gin[:, None], (gin - gout)[:, None], din[:, None], whole])
    onehot = np.zeros((len(g), NSRC))
    onehot[np.arange(len(g)), src] = 1
    ied = cv2.integral((edges > 0).astype(np.float32))
    edin, _ = _box_sum(ied, x1 + 1, y1 + 1, x2 - 1, y2 - 1, H, W)
    edring, _ = _box_sum(ied, x1 - bw, y1 - bw, x2 + bw, y2 + bw, H, W)
    # How many image candidates agree with each one.
    ix = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
    iy = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
    a = w * h
    inter = ix * iy
    iou = inter / (a[:, None] + a[None] - inter)
    from_image = src < ROUGH_SRC
    agree = ((iou > 0.8) & from_image[None]).sum(1)
    agree6 = ((iou > 0.6) & from_image[None]).sum(1)
    # Containment: the candidate is the inner part of a larger image candidate (the inner disc of a red rim...), or
    # holds a smaller one.
    inside = ((x1[:, None] >= x1[None] + 0.5) & (y1[:, None] >= y1[None] + 0.5) & (x2[:, None] <= x2[None] - 0.5)
              & (y2[:, None] <= y2[None] - 0.5))
    ratio = a[None] / a[:, None]
    enclosed = inside & from_image[None] & (ratio > 1.1) & (ratio < 2.5)
    encloses = inside.T & from_image[None] & (ratio < 0.9) & (ratio > 0.3)
    cxa, cya = (x1 + x2) / 2, (y1 + y2) / 2
    centred = (np.abs(cxa[:, None] - cxa[None]) / w[:, None] + np.abs(cya[:, None] - cya[None]) / h[:, None]) < 0.2
    return np.hstack([F, onehot, edin[:, None], edring[:, None], np.log1p(agree)[:, None], np.log1p(agree6)[:, None],
                      np.log1p(enclosed.sum(1))[:, None], np.log1p(encloses.sum(1))[:, None],
                      np.log1p((enclosed & centred).sum(1))[:, None], np.log1p((encloses & centred).sum(1))[:, None]])


def extract(frame, rough):
    """Candidates of ``rough`` in pixels of the frame, and their features."""
    img, k, origin, r = _prep(frame, rough)
    G, src, maps, edges = candidates(img, r)
    F = features(img, r, G, src, maps, edges)
    boxes = np.stack([G[:, 0] / k[0] + origin[0], G[:, 1] / k[1] + origin[1],
                      G[:, 2] / k[0] + origin[0], G[:, 3] / k[1] + origin[1]], 1)
    return boxes, F


# ---------------------------------------------------------------- ranker

def load(path) -> list:
    with np.load(path) as z:
        return [(z[f'mu{s}'], z[f'sd{s}'], [z[f'p{s}_{i}'] for i in range(4)]) for s in range(ENSEMBLE)]


def score(model, F):
    """The mean score of the MLPs of ``model`` for each row of ``F``."""
    out = 0.0
    for mu, sd, (W1, b1, W2, b2) in model:
        Z = (F - mu) / sd
        out = out + np.maximum(Z @ W1 + b1, 0) @ W2 + b2[0]
    return out / len(model)


def train_ranker(cases, *, seed=0, hidden=32, epochs=30, lr=2e-3, wd=1e-4, tau=0.05, ncand=200, top=30, batch=16):
    """One MLP trained listwise: for each case a softmax over its candidates, against softmax(IoU / tau).

    ``cases`` is a list of (features, IoU of each candidate with the person's box). Adam, mini-batches of ``batch``
    cases; a case with more than ``ncand`` candidates is cut to its ``top`` best and ``ncand`` random ones.
    Returns (mu, sd, [W1, b1, W2, b2])."""
    rng = np.random.default_rng(seed)
    X = np.vstack([F for F, _ in cases]).astype(np.float64)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    mu[0], sd[0] = 0, 1
    d = X.shape[1]
    ps = [rng.normal(0, 1 / np.sqrt(d), (d, hidden)), np.zeros(hidden), rng.normal(0, 1 / np.sqrt(hidden), hidden),
          np.zeros(1)]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    t = 0
    Zs = [(F.astype(np.float64) - mu) / sd for F, _ in cases]
    Ys = [y.astype(np.float64) for _, y in cases]
    for _ in range(epochs):
        order = rng.permutation(len(cases))
        for start in range(0, len(order), batch):
            grads = [np.zeros_like(p) for p in ps]
            nb = 0
            for ci in order[start:start + batch]:
                Z, y = Zs[ci], Ys[ci]
                if len(y) > ncand:
                    idx = np.argsort(-y)[:top]
                    idx = np.unique(np.r_[idx, rng.choice(len(y), ncand, replace=False)])
                    Z, y = Z[idx], y[idx]
                a = Z @ ps[0] + ps[1]
                hd = np.maximum(a, 0)
                o = hd @ ps[2] + ps[3][0]
                p = np.exp(o - o.max())
                p /= p.sum()
                target = np.exp((y - y.max()) / tau)
                target /= target.sum()
                go = p - target
                grads[2] += hd.T @ go
                grads[3] += go.sum()
                gh = np.outer(go, ps[2]) * (a > 0)
                grads[0] += Z.T @ gh
                grads[1] += gh.sum(0)
                nb += 1
            t += 1
            for i in range(4):
                g = grads[i] / nb + (wd * ps[i] if i in (0, 2) else 0)
                m[i] = .9 * m[i] + .1 * g
                v[i] = .999 * v[i] + .001 * g ** 2
                ps[i] -= lr * (m[i] / (1 - .9 ** t)) / (np.sqrt(v[i] / (1 - .999 ** t)) + 1e-8)
    return mu, sd, ps


def pick(boxes, F, model, rough, width, height) -> list[float] | None:
    """The softmax-weighted mean of the candidates ``boxes`` (features ``F``) of ``rough``, pulled back toward it
    when close to it, on a ``width`` x ``height`` image; None when nothing usable is left."""
    s = score(model, F)
    p = np.exp((s - s.max()) / TEMP)
    p /= p.sum()
    out = list((p[:, None] * boxes).sum(0))
    a = min(1.0, max(0.0, (_iou(out, rough) - BLEND_LO) / (1.0 - BLEND_LO)))
    out = [(1 - a) * o + a * q for o, q in zip(out, rough)]
    out = [min(max(out[0], 0.0), width), min(max(out[1], 0.0), height), min(max(out[2], 0.0), width),
           min(max(out[3], 0.0), height)]
    if out[2] - out[0] < 1 or out[3] - out[1] < 1:
        return None
    return out


def fit(frame, rough, model) -> list[float] | None:
    """The edge fit of ``rough`` (pixels of ``frame``, at least 3 px a side), or None when nothing usable is left."""
    boxes, F = extract(frame, rough)
    return pick(boxes, F, model, rough, frame.shape[1], frame.shape[0])
