"""Hydraulic (droplet) and thermal erosion, both modulated by rock hardness.

Hydraulic erosion simulates thousands of rain droplets in parallel batches.
Each droplet carries sediment, eroding where it speeds up and depositing where
it slows. Erosion is scaled by ``1 - hardness`` so soft shales are carved into
valleys while hard caprock survives as ridges.

Thermal erosion moves material downslope where the slope exceeds the angle of
repose; the repose angle grows with hardness, so hard layers keep near
vertical faces (cliffs) while soft layers relax into talus slopes.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import ErosionConfig
from ..grid import NEIGHBORS8, shifted


def _brush(radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oy, ox, w = [], [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d = np.hypot(dx, dy)
            if d <= radius:
                oy.append(dy)
                ox.append(dx)
                w.append(max(0.0, radius + 0.5 - d))
    w = np.array(w)
    return np.array(oy), np.array(ox), w / w.sum()


def hydraulic_erosion(height: np.ndarray, hardness_fn: Callable[[np.ndarray], np.ndarray],
                      cfg: ErosionConfig, rng: np.random.Generator) -> np.ndarray:
    h_m = height.astype(np.float64)
    size_y, size_x = h_m.shape
    lo, hi = float(h_m.min()), float(h_m.max())
    scale = max(hi - lo, 1.0)
    # Normalised heights; the classic droplet parameters are tuned for a [0, 1]
    # height range on a ~256 grid, so slopes are rescaled for other sizes.
    res_scale = size_x / 256.0
    hn = (h_m - lo) / scale * res_scale
    flat = hn.reshape(-1)
    n_cells = flat.size

    n_drops = int(cfg.droplets_per_cell * n_cells)
    batch = max(256, cfg.batch)
    by, bx, bw = _brush(cfg.brush_radius)
    hard = hardness_fn(lo + hn / res_scale * scale)

    for b in range(0, n_drops, batch):
        n = min(batch, n_drops - b)
        if b // batch % 4 == 0:
            hard = hardness_fn(lo + hn / res_scale * scale).reshape(-1)
        px = rng.uniform(0, size_x - 1.001, n)
        py = rng.uniform(0, size_y - 1.001, n)
        dx = np.zeros(n)
        dy = np.zeros(n)
        speed = np.ones(n)
        water = np.ones(n)
        sed = np.zeros(n)
        alive = np.ones(n, dtype=bool)

        for _ in range(cfg.max_steps):
            ix = px.astype(np.int64)
            iy = py.astype(np.int64)
            fx = px - ix
            fy = py - iy
            i00 = iy * size_x + ix
            h00 = flat[i00]
            h10 = flat[i00 + 1]
            h01 = flat[i00 + size_x]
            h11 = flat[i00 + size_x + 1]
            gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
            gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
            hcur = (h00 * (1 - fx) + h10 * fx) * (1 - fy) + (h01 * (1 - fx) + h11 * fx) * fy

            dx = dx * cfg.inertia - gx * (1 - cfg.inertia)
            dy = dy * cfg.inertia - gy * (1 - cfg.inertia)
            length = np.hypot(dx, dy)
            stuck = length < 1e-9
            if stuck.any():
                a = rng.uniform(0, 2 * np.pi, int(stuck.sum()))
                dx[stuck] = np.cos(a)
                dy[stuck] = np.sin(a)
                length[stuck] = 1.0
            dx /= length
            dy /= length
            npx = px + dx
            npy = py + dy
            alive &= (npx >= 0) & (npx < size_x - 1.001) & (npy >= 0) & (npy < size_y - 1.001)
            if not alive.any():
                break
            npx = np.where(alive, npx, px)
            npy = np.where(alive, npy, py)

            jx = npx.astype(np.int64)
            jy = npy.astype(np.int64)
            gfx = npx - jx
            gfy = npy - jy
            j00 = jy * size_x + jx
            hnew = ((flat[j00] * (1 - gfx) + flat[j00 + 1] * gfx) * (1 - gfy)
                    + (flat[j00 + size_x] * (1 - gfx) + flat[j00 + size_x + 1] * gfx) * gfy)
            dh = hnew - hcur

            cap = np.maximum(-dh * speed * water * cfg.capacity, cfg.min_capacity)
            depositing = alive & ((sed > cap) | (dh > 0))
            eroding = alive & ~depositing

            dep = np.where(dh > 0, np.minimum(dh, sed), (sed - cap) * cfg.deposit_speed)
            dep = np.where(depositing, np.maximum(dep, 0.0), 0.0)
            soft = 1.0 - hard[i00] * cfg.hardness_effect
            ero = np.minimum((cap - sed) * cfg.erode_speed, -dh) * soft
            ero = np.where(eroding, np.maximum(ero, 0.0), 0.0)
            sed = sed - dep + ero

            # Deposit bilinearly at the current node; erode with a soft brush.
            idx = [i00, i00 + 1, i00 + size_x, i00 + size_x + 1]
            wts = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]
            all_idx = [np.concatenate(idx)]
            all_w = [np.concatenate([dep * w for w in wts])]
            if ero.any():
                bix = np.clip(ix[None, :] + bx[:, None], 0, size_x - 1)
                biy = np.clip(iy[None, :] + by[:, None], 0, size_y - 1)
                all_idx.append((biy * size_x + bix).reshape(-1))
                all_w.append((-ero[None, :] * bw[:, None]).reshape(-1))
            flat += np.bincount(np.concatenate(all_idx), weights=np.concatenate(all_w), minlength=n_cells)

            speed = np.sqrt(np.maximum(speed * speed - dh * cfg.gravity, 0.0))
            water *= 1.0 - cfg.evaporate
            px, py = npx, npy

    return lo + hn / res_scale * scale


def thermal_erosion(height: np.ndarray, hardness: Callable[[np.ndarray], np.ndarray] | np.ndarray,
                    cfg: ErosionConfig, cell: float, iterations: int | None = None) -> np.ndarray:
    h = height.astype(np.float64).copy()
    iters = cfg.thermal_iterations if iterations is None else iterations
    soft = np.tan(np.radians(cfg.soft_talus_deg))
    hard_t = np.tan(np.radians(cfg.hard_talus_deg))
    dists = [np.hypot(dy, dx) * cell for dy, dx in NEIGHBORS8]
    for it in range(iters):
        hv = hardness(h) if callable(hardness) else hardness
        talus = soft + (hard_t - soft) * np.clip(hv, 0, 1) ** 1.5
        best = np.zeros_like(h)
        best_k = np.full(h.shape, -1)
        for k, (dy, dx) in enumerate(NEIGHBORS8):
            excess = (h - shifted(h, dy, dx)) - talus * dists[k]
            better = excess > best
            best = np.where(better, excess, best)
            best_k = np.where(better, k, best_k)
        move = best * 0.25
        delta = -move
        for k, (dy, dx) in enumerate(NEIGHBORS8):
            m = np.where(best_k == k, move, 0.0)
            delta += shifted(m, -dy, -dx, fill=0.0)
        h += delta
    return h
