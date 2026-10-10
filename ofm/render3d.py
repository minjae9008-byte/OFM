"""Dependency-free 3D previews: density cross-sections and a z-buffered
point-splat mesh renderer (numpy only)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .volume.density import DensityField
from .volume.mesher import Mesh


def cross_section(world, field: DensityField, x0: float, y0: float, x1: float, y1: float,
                  z0: float, z1: float, px: int = 512) -> Image.Image:
    """Vertical slice along the segment (x0, y0) -> (x1, y1): rock coloured by stratum."""
    length = float(np.hypot(x1 - x0, y1 - y0))
    pz = max(32, int(px * (z1 - z0) / max(length, 1e-6)))
    t = np.linspace(0.0, 1.0, px)
    zs = np.linspace(z1, z0, pz)
    X = (x0 + (x1 - x0) * t)[:, None, None] * np.ones((1, 1, pz))
    Y = (y0 + (y1 - y0) * t)[:, None, None] * np.ones((1, 1, pz))
    Z = np.broadcast_to(zs[None, None, :], X.shape).copy()
    D, D0 = field.sample(X, Y, Z)
    D, D0 = D[:, 0, :].T, D0[:, 0, :].T
    strata = world.strata
    fold = strata.fold_at(X[:, 0, 0], Y[:, 0, 0])
    idx = strata.layer_index(Z[:, 0, :].T, fold[None, :])
    rock = strata.color_table[idx]
    sky = np.array([170, 200, 230], dtype=np.float64)
    cave_air = np.array([25, 22, 30], dtype=np.float64)
    img = np.where((D > 0)[..., None], rock, np.where((D0 > 0)[..., None], cave_air, sky))
    sea = (Z[:, 0, :].T < world.cfg.world.sea_level) & (D <= 0) & (D0 <= 0)
    img[sea] = (50, 90, 140)
    edge = np.abs(D) < (z1 - z0) / pz * 0.9
    img[edge] = img[edge] * 0.4
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f


def _bary(n: int) -> np.ndarray:
    ws = [(i / n, j / n, 1 - (i + j) / n) for i in range(n + 1) for j in range(n + 1 - i)]
    return np.array(ws)


def render_mesh(mesh: Mesh, eye, target, width: int = 640, height: int = 400, fov_deg: float = 55.0,
                light=(0.4, 0.3, 0.85), background=(160, 190, 220), max_samples: int = 72) -> Image.Image:
    """Z-buffered splat rasteriser: each triangle is sampled densely enough
    (based on its projected size) to cover its pixels, then Lambert shaded."""
    r, u, f = _look_at(eye, target)
    eye = np.asarray(eye, float)
    p = mesh.positions
    tri = mesh.indices.astype(np.int64)
    img = np.empty((height * width, 3))
    img[:] = background
    zbuf = np.full(height * width, np.inf)
    if len(tri) == 0:
        return Image.fromarray(img.reshape(height, width, 3).astype(np.uint8))
    focal = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
    rel = p - eye
    zc = rel @ f
    sx = width / 2 + focal * (rel @ r) / np.maximum(zc, 1e-3)
    sy = height / 2 - focal * (rel @ u) / np.maximum(zc, 1e-3)

    nrm = mesh.normals.astype(np.float64)
    n_t = nrm[tri].sum(axis=1)
    n_t /= np.maximum(np.linalg.norm(n_t, axis=1, keepdims=True), 1e-9)
    centre = p[tri].mean(axis=1)
    front = (n_t * (eye - centre)).sum(axis=1) > 0
    in_front = (zc[tri] > 0.5).all(axis=1)
    on_screen = ((sx[tri].max(1) >= 0) & (sx[tri].min(1) < width) & (sy[tri].max(1) >= 0) & (sy[tri].min(1) < height))
    keep = front & in_front & on_screen
    tri, n_t = tri[keep], n_t[keep]
    if len(tri) == 0:
        return Image.fromarray(img.reshape(height, width, 3).astype(np.uint8))
    if mesh.colors is not None:
        col_t = mesh.colors[tri, :3].astype(np.float64).mean(axis=1)
    else:
        col_t = np.full((len(tri), 3), 180.0)
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    rgb_t = col_t * (0.38 + 0.62 * np.clip(n_t @ L, 0, 1))[:, None]

    ex = np.stack([sx[tri], sy[tri]], axis=-1)
    span = np.max(np.linalg.norm(ex - np.roll(ex, 1, axis=1), axis=2), axis=1)
    level = np.clip(np.ceil(span * 1.4), 1, max_samples).astype(int)
    for n in np.unique(level):
        sel = level == n
        ws = _bary(int(n))
        t = tri[sel]
        a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
        pts = (ws[:, 0, None, None] * a + ws[:, 1, None, None] * b + ws[:, 2, None, None] * c).reshape(-1, 3)
        col = np.tile(rgb_t[sel], (len(ws), 1))
        rl = pts - eye
        z = rl @ f
        px = (width / 2 + focal * (rl @ r) / z).astype(np.int64)
        py = (height / 2 - focal * (rl @ u) / z).astype(np.int64)
        ok = (px >= 0) & (px < width) & (py >= 0) & (py < height) & (z > 0.5)
        pix = py[ok] * width + px[ok]
        if pix.size == 0:
            continue
        z = z[ok]
        col = col[ok]
        order = np.lexsort((z, pix))
        pix, z, col = pix[order], z[order], col[order]
        first = np.r_[True, pix[1:] != pix[:-1]]
        pix, z, col = pix[first], z[first], col[first]
        closer = z < zbuf[pix]
        zbuf[pix[closer]] = z[closer]
        img[pix[closer]] = col[closer]
    # light fog for depth cue
    fog = np.where(np.isfinite(zbuf), 1.0 - np.exp(-np.nan_to_num(zbuf, posinf=0) / 2500.0), 0.0)[:, None]
    img = img * (1 - fog * 0.6) + np.asarray(background) * fog * 0.6
    return Image.fromarray(np.clip(img.reshape(height, width, 3), 0, 255).astype(np.uint8))


def cave_profile(world, field: DensityField, cave, px_per_m: float = 1.2, above: float = 60.0,
                 below: float = 25.0) -> Image.Image:
    """Unrolled vertical section following a cave's trunk passage."""
    segs = cave.segments[: max(1, cave.main_count or len(cave.segments))]
    pts = np.vstack([segs[:, 0:3], segs[-1:, 3:6]])
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))]
    total = float(d[-1])
    px = max(64, int(total * px_per_m))
    s = np.linspace(0.0, total, px)
    xs = np.interp(s, d, pts[:, 0])
    ys = np.interp(s, d, pts[:, 1])
    z0 = float(pts[:, 2].min()) - below
    z1 = float(field.surface(xs, ys).max()) + above
    pz = max(32, int((z1 - z0) * px_per_m))
    zs = np.linspace(z1, z0, pz)
    X = np.broadcast_to(xs[:, None, None], (px, 1, pz)).copy()
    Y = np.broadcast_to(ys[:, None, None], (px, 1, pz)).copy()
    Z = np.broadcast_to(zs[None, None, :], (px, 1, pz)).copy()
    D, D0 = field.sample(X, Y, Z)
    D, D0 = D[:, 0, :].T, D0[:, 0, :].T
    strata = world.strata
    fold = strata.fold_at(xs, ys)
    idx = strata.layer_index(Z[:, 0, :].T, fold[None, :])
    img = np.where((D > 0)[..., None], strata.color_table[idx],
                   np.where((D0 > 0)[..., None], np.array([25.0, 22.0, 30.0]), np.array([170.0, 200.0, 230.0])))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
