"""Grid helpers shared by all stages (numpy only, no scipy)."""
from __future__ import annotations

import numpy as np

NEIGHBORS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NEIGHBORS4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]


def shifted(a: np.ndarray, dy: int, dx: int, fill: float | None = None) -> np.ndarray:
    """Return b with b[y, x] = a[y + dy, x + dx]; out of range uses edge values or ``fill``."""
    h, w = a.shape[:2]
    if fill is None:
        ys = np.clip(np.arange(h) + dy, 0, h - 1)
        xs = np.clip(np.arange(w) + dx, 0, w - 1)
        return a[ys][:, xs]
    out = np.full_like(a, fill)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = a[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
    return out


def _box1d(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    if r <= 0:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def box_blur(a: np.ndarray, r: int) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64)
    for axis in range(out.ndim):
        out = _box1d(out, int(r), axis)
    return out


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian approximation with three box passes."""
    if sigma <= 0.3:
        return np.asarray(a, dtype=np.float64).copy()
    w = np.sqrt(12.0 * sigma * sigma / 3.0 + 1.0)
    r = max(1, int(round((w - 1.0) / 2.0)))
    out = np.asarray(a, dtype=np.float64)
    for _ in range(3):
        out = box_blur(out, r)
    return out


def sliding_min(a: np.ndarray, r: int) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64)
    for axis in range(2):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(p, 2 * r + 1, axis=axis)
        out = win.min(axis=-1)
    return out


def sliding_max(a: np.ndarray, r: int) -> np.ndarray:
    return -sliding_min(-np.asarray(a, dtype=np.float64), r)


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    return sliding_max(mask.astype(np.float64), r) > 0.5


def smoothstep(e0: float, e1: float, x) -> np.ndarray:
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gradient(h: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (dh/dx, dh/dy) in metres per metre; x = columns, y = rows."""
    gy, gx = np.gradient(h.astype(np.float64), cell)
    return gx, gy


def slope(h: np.ndarray, cell: float) -> np.ndarray:
    """Slope as tangent (rise over run)."""
    gx, gy = gradient(h, cell)
    return np.hypot(gx, gy)


def bilinear(a: np.ndarray, x, y) -> np.ndarray:
    """Sample a[y, x] at fractional index coordinates with edge clamping."""
    h, w = a.shape
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, w - 1.0)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, h - 1.0)
    x0 = np.minimum(np.floor(x).astype(np.int64), w - 2) if w > 1 else np.zeros_like(x, dtype=np.int64)
    y0 = np.minimum(np.floor(y).astype(np.int64), h - 2) if h > 1 else np.zeros_like(y, dtype=np.int64)
    fx = x - x0
    fy = y - y0
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    a00 = a[y0, x0]
    a10 = a[y0, x1]
    a01 = a[y1, x0]
    a11 = a[y1, x1]
    return (a00 * (1 - fx) + a10 * fx) * (1 - fy) + (a01 * (1 - fx) + a11 * fx) * fy


def _cubic_weights(t: np.ndarray) -> list[np.ndarray]:
    # Catmull-Rom spline weights
    t2 = t * t
    t3 = t2 * t
    return [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]


def bicubic(a: np.ndarray, x, y) -> np.ndarray:
    h, w = a.shape
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, w - 1.0)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    wx = _cubic_weights(x - x0)
    wy = _cubic_weights(y - y0)
    out = np.zeros(np.broadcast(x, y).shape)
    for j in range(4):
        yy = np.clip(y0 + j - 1, 0, h - 1)
        row = np.zeros_like(out)
        for i in range(4):
            xx = np.clip(x0 + i - 1, 0, w - 1)
            row += wx[i] * a[yy, xx]
        out += wy[j] * row
    return out


def resample(a: np.ndarray, shape: tuple[int, int], cubic: bool = False) -> np.ndarray:
    """Resample with aligned corners (first/last samples map to each other)."""
    h, w = a.shape
    ys = np.linspace(0.0, h - 1.0, shape[0])
    xs = np.linspace(0.0, w - 1.0, shape[1])
    xx, yy = np.meshgrid(xs, ys)
    return bicubic(a, xx, yy) if cubic else bilinear(a, xx, yy)


def cells_to_cells(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample a cell-centred grid covering the map onto another cell-centred grid covering
    the same map: cell i of n sits at (i + 0.5) / n of the extent. (``resample`` maps the
    first and last samples onto each other instead, which is right for vertex grids.)"""
    h, w = a.shape
    ys = (np.arange(shape[0]) + 0.5) * h / shape[0] - 0.5
    xs = (np.arange(shape[1]) + 0.5) * w / shape[1] - 0.5
    return bilinear(a, xs[None, :], ys[:, None])


def chamfer_distance(mask: np.ndarray, max_dist: float | None = None) -> np.ndarray:
    """Approximate Euclidean distance (in cells) to the nearest True cell."""
    d = np.where(mask, 0.0, np.inf)
    limit = int(max_dist) + 2 if max_dist is not None else sum(mask.shape)
    diag = np.sqrt(2.0)
    for _ in range(limit):
        new = d.copy()
        for dy, dx in NEIGHBORS8:
            wgt = diag if dy and dx else 1.0
            np.minimum(new, shifted(d, dy, dx, fill=np.inf) + wgt, out=new)
        if np.array_equal(new, d):
            break
        d = new
    if max_dist is not None:
        d = np.minimum(d, max_dist)
    return d


def _neighbor_offsets(ndim: int, diagonal: bool) -> list[tuple[int, ...]]:
    offs = []
    for off in np.ndindex(*([3] * ndim)):
        o = tuple(v - 1 for v in off)
        if all(v == 0 for v in o):
            continue
        if not diagonal and sum(abs(v) for v in o) != 1:
            continue
        offs.append(o)
    return offs


def _shift_nd(a: np.ndarray, off: tuple[int, ...], fill) -> np.ndarray:
    out = np.full_like(a, fill)
    src = []
    dst = []
    for n, o in zip(a.shape, off):
        if o >= 0:
            src.append(slice(o, n))
            dst.append(slice(0, n - o))
        else:
            src.append(slice(0, n + o))
            dst.append(slice(-o, n))
    out[tuple(dst)] = a[tuple(src)]
    return out


def label(mask: np.ndarray, diagonal: bool = True) -> tuple[np.ndarray, int]:
    """Connected components for 2D or 3D masks (labels 1..n, 0 = background)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), 0
    big = np.iinfo(np.int64).max
    lab = np.where(mask, np.arange(mask.size, dtype=np.int64).reshape(mask.shape), big)
    offsets = _neighbor_offsets(mask.ndim, diagonal)
    while True:
        new = lab.copy()
        for off in offsets:
            np.minimum(new, _shift_nd(lab, off, big), out=new)
        new[~mask] = big
        flat = new.reshape(-1)
        valid = flat != big
        for _ in range(8):  # pointer jumping: a label is a cell index in the same component
            idx = flat[valid]
            jumped = flat[idx]
            if np.array_equal(jumped, idx):
                break
            flat[valid] = jumped
        if np.array_equal(new, lab):
            break
        lab = new
    roots = np.unique(lab[mask])
    out = np.zeros(mask.shape, dtype=np.int32)
    out[mask] = np.searchsorted(roots, lab[mask]).astype(np.int32) + 1
    return out, int(len(roots))


def border_mask(shape: tuple[int, int], width: int = 1) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[:width, :] = True
    m[-width:, :] = True
    m[:, :width] = True
    m[:, -width:] = True
    return m


def normalize01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float64)
    return (a - lo) / (hi - lo)


def entropy_of_onehot(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    p = np.clip(stack, 1e-9, 1.0)
    p = p / p.sum(axis=axis, keepdims=True)
    return -(p * np.log(p)).sum(axis=axis)
