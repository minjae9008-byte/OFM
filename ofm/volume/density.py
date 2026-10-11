"""The volumetric density field D(x, y, z) (positive = rock, negative = air).

Starting from the heightfield as an approximate signed distance, the field
is modified by processes a heightmap cannot represent:
  * differential weathering: soft strata on cliff faces recede further than
    hard caprock, leaving overhanging ledges;
  * wave-cut notches at sea level on coastal cliffs;
  * rough rock surface displacement on cliffs;
  * karst / sea caves (capsules with flat walkable floors, ellipsoid chambers);
  * volumetric landmarks (spires, arches, buttes).
"""
from __future__ import annotations

import numpy as np

from ..grid import bilinear, dilate, gradient, label, smoothstep
from ..noise import Noise, derive_seed


class DensityField:
    def __init__(self, world):
        cfg = world.cfg
        self.world = world
        self.cc = cfg.cliffs
        self.vc = cfg.volume
        self.sea = cfg.world.sea_level
        self.hires = world.hires
        self.spacing = world.hires_spacing
        gx, gy = gradient(self.hires, self.spacing)
        self.slope_factor = np.sqrt(1.0 + gx * gx + gy * gy)
        slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
        cliff_w = smoothstep(self.cc.cliff_slope_deg - 6.0, self.cc.cliff_slope_deg + 6.0, slope_deg)
        # undercuts and rough rock are weathering of exposed faces: not on steep lake beds or
        # the sea floor (wave notches at sea level are handled separately)
        n0 = self.hires.shape[0]
        c0 = np.arange(n0) * self.spacing / world.cell - 0.5
        wet = bilinear(world.water.astype(np.float64), c0[None, :], c0[:, None])
        cliff_w = cliff_w * (1.0 - smoothstep(0.3, 0.7, wet))
        # Only coherent cliff faces get volumetric treatment, not isolated steep pixels.
        lab, n = label(cliff_w > 0.5)
        if n:
            sizes = np.bincount(lab.ravel())
            big = sizes >= self.cc.min_cliff_cells
            big[0] = False
            cliff_w = cliff_w * dilate(big[lab], 1)
        self.cliff_w = cliff_w
        n = self.hires.shape[0]
        coords = np.arange(n) * self.spacing
        X, Y = np.meshgrid(coords, coords)
        self.coast_w = np.zeros_like(self.hires)
        if world.coast_cliff is not None:
            cw = world.coast_cliff.astype(np.float64)
            self.coast_w = np.clip(bilinear(cw, X / world.cell - 0.5, Y / world.cell - 0.5) * 1.5, 0, 1)
        self.noise = Noise(derive_seed(cfg.world.seed, "density"))
        self.cave_noise = Noise(derive_seed(cfg.world.seed, "cave-walls"))
        self.caves = world.caves
        self.features = world.features
        self._cave_boxes = [c.bbox(pad=cfg.caves.wall_noise + 2.0) for c in self.caves]
        self._feat_boxes = [f.bbox() for f in self.features]

    # -- 2D lookups ---------------------------------------------------------------
    def _uv(self, X, Y):
        return np.asarray(X) / self.spacing, np.asarray(Y) / self.spacing

    def surface(self, X, Y) -> np.ndarray:
        u, v = self._uv(X, Y)
        return bilinear(self.hires, u, v)

    def base_density(self, X, Y, Z) -> np.ndarray:
        """Heightfield-only signed distance approximation."""
        u, v = self._uv(X, Y)
        h = bilinear(self.hires, u, v)
        sf = bilinear(self.slope_factor, u, v)
        return (h - Z) / sf

    # -- full field -----------------------------------------------------------------
    def sample(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """X, Y, Z are 3D grids (same shape). Returns (D, D_base)."""
        u, v = self._uv(X[..., 0], Y[..., 0])
        h2 = bilinear(self.hires, u, v)[..., None]
        sf2 = bilinear(self.slope_factor, u, v)[..., None]
        cliff2 = bilinear(self.cliff_w, u, v)[..., None]
        coast2 = bilinear(self.coast_w, u, v)[..., None]
        D0 = (h2 - Z) / sf2
        D = D0.copy()
        below = np.clip(D0, 0.0, None)

        strata = self.world.strata
        fold = strata.fold_at(X[..., 0], Y[..., 0])[..., None]
        idx = strata.layer_index(Z, fold)
        soft = 1.0 - strata.hardness_table[idx]

        if cliff2.max() > 0.01:
            r = self.cc.undercut_depth * soft ** 1.5 * cliff2 / 0.567
            D = D - r * np.exp(-below / np.maximum(r, 1e-3))
            # rough rock surface, evaluated only on near-surface cliff voxels
            m = (np.broadcast_to(cliff2, D.shape) > 0.02) & (np.abs(D0) < 24.0)
            if m.any():
                rough = self.noise.fbm3(X[m] / 9.0, Y[m] / 9.0, Z[m] / 7.0, 3) * self.cc.rock_noise
                D[m] += rough * np.broadcast_to(cliff2, D.shape)[m] * np.exp(-np.abs(D0[m]) / 8.0)
        if coast2.max() > 0.01:
            zn = (Z - self.sea - self.cc.notch_height * 0.6) / self.cc.notch_height
            nd = self.cc.notch_depth * coast2 * np.exp(-zn * zn) * (0.5 + soft) / 0.567
            D = D - nd * np.exp(-below / np.maximum(nd, 1e-3))

        lo = np.array([X.min(), Y.min(), Z.min()])
        hi = np.array([X.max(), Y.max(), Z.max()])
        aligned = _aligned(X, Y, Z)
        hits = [c for c, (blo, bhi) in zip(self.caves, self._cave_boxes)
                if not (np.any(bhi < lo) or np.any(blo > hi))]
        if hits:
            # wall roughness only where some cave passage can reach
            near = np.zeros(D.shape, dtype=bool)
            for cave in hits:
                for sl in _cave_slices(X, Y, Z, cave, aligned):
                    near[sl] = True
            wall = np.zeros(D.shape)
            wall[near] = self.cave_noise.fbm3(X[near] / 6.0, Y[near] / 6.0, Z[near] / 4.0, 3) \
                * self.world.cfg.caves.wall_noise
            for cave in hits:
                D = _carve_cave(D, X, Y, Z, cave, wall, aligned)
        for feat, (flo, fhi) in zip(self.features, self._feat_boxes):
            if np.any(fhi < lo) or np.any(flo > hi):
                continue
            s = feat.sdf(X, Y, Z)
            m = np.abs(s) < 6.0
            if m.any():
                s[m] += self.noise.fbm3(X[m] / 5.0, Y[m] / 5.0, Z[m] / 5.0, 2) * 0.8
            D = np.maximum(D, -s)
        return D, D0


def _sub_box(X, Y, Z, lo, hi, aligned=True):
    """Index box of an axis-aligned ascending grid inside [lo, hi]; a boolean mask otherwise."""
    if not aligned:
        m = ((X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1]) & (Z >= lo[2]) & (Z <= hi[2]))
        return m if m.any() else None
    xs = X[:, 0, 0]
    ys = Y[0, :, 0]
    zs = Z[0, 0, :]
    i0, i1 = np.searchsorted(xs, lo[0]), np.searchsorted(xs, hi[0], side="right")
    j0, j1 = np.searchsorted(ys, lo[1]), np.searchsorted(ys, hi[1], side="right")
    k0, k1 = np.searchsorted(zs, lo[2]), np.searchsorted(zs, hi[2], side="right")
    if i0 >= i1 or j0 >= j1 or k0 >= k1:
        return None
    return slice(i0, i1), slice(j0, j1), slice(k0, k1)


def _aligned(X, Y, Z) -> bool:
    xs, ys, zs = X[:, 0, 0], Y[0, :, 0], Z[0, 0, :]
    return bool(np.all(np.diff(xs) >= 0) and np.all(np.diff(ys) >= 0) and np.all(np.diff(zs) >= 0)
                and X.shape[1] * X.shape[2] > 0 and np.all(X[:1, :1, :1] == X[0, 0, 0])
                and (X.shape[1] == 1 or np.allclose(X[:, 1, 0], xs))
                and (Y.shape[0] == 1 or np.allclose(Y[1, :, 0], ys)))


def _cave_slices(X, Y, Z, cave, aligned):
    pad = 3.0
    out = []
    for seg in cave.segments:
        r = max(seg[6], seg[7])
        sl = _sub_box(X, Y, Z, np.minimum(seg[0:3], seg[3:6]) - r - pad,
                      np.maximum(seg[0:3], seg[3:6]) + r + pad, aligned)
        if sl is not None:
            out.append(sl)
    for ch in cave.chambers:
        sl = _sub_box(X, Y, Z, ch[0:3] - ch[3:6] - pad, ch[0:3] + ch[3:6] + pad, aligned)
        if sl is not None:
            out.append(sl)
    return out


def _carve_cave(D, X, Y, Z, cave, wall, aligned=True):
    pad = 3.0
    for seg in cave.segments:
        a = seg[0:3]
        b = seg[3:6]
        r = max(seg[6], seg[7])
        lo = np.minimum(a, b) - r - pad
        hi = np.maximum(a, b) + r + pad
        sl = _sub_box(X, Y, Z, lo, hi, aligned)
        if sl is None:
            continue
        x, y, z = X[sl], Y[sl], Z[sl]
        ba = b - a
        denom = max(float(ba @ ba), 1e-9)
        t = np.clip(((x - a[0]) * ba[0] + (y - a[1]) * ba[1] + (z - a[2]) * ba[2]) / denom, 0.0, 1.0)
        px = x - (a[0] + ba[0] * t)
        py = y - (a[1] + ba[1] * t)
        pz = z - (a[2] + ba[2] * t)
        rad = seg[6] + (seg[7] - seg[6]) * t
        s = np.sqrt(px * px + py * py + pz * pz) - rad + wall[sl]
        floor = a[2] + ba[2] * t - rad * 0.55
        s = np.maximum(s, floor - z)
        D[sl] = np.minimum(D[sl], s)
    for ch in cave.chambers:
        c = ch[0:3]
        rr = ch[3:6]
        sl = _sub_box(X, Y, Z, c - rr - pad, c + rr + pad, aligned)
        if sl is None:
            continue
        x, y, z = X[sl], Y[sl], Z[sl]
        k = np.sqrt(((x - c[0]) / rr[0]) ** 2 + ((y - c[1]) / rr[1]) ** 2 + ((z - c[2]) / rr[2]) ** 2)
        s = (k - 1.0) * rr.min() + wall[sl] * 1.3
        s = np.maximum(s, (c[2] - rr[2] * 0.6) - z)
        D[sl] = np.minimum(D[sl], s)
    return D
