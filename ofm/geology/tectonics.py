"""Plate-tectonic base relief.

Plates are warped Voronoi cells with random velocities. The relative velocity
across each plate boundary decides whether it is convergent (orogeny,
mountain belts) or divergent (rift valleys). A continental-shape mask then
decides where the land mass sits, and the result is shifted so the requested
land fraction ends above sea level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import OFMConfig
from ..grid import NEIGHBORS4, blur, normalize01, shifted, smoothstep
from ..noise import Noise, derive_seed


@dataclass
class TectonicResult:
    height: np.ndarray       # metres
    uplift: np.ndarray       # metres of orogenic uplift
    plate_id: np.ndarray
    convergence: np.ndarray  # signed boundary convergence, blurred


def _shape_mask(shape: str, u: np.ndarray, v: np.ndarray, noise: Noise, rng: np.random.Generator) -> np.ndarray:
    if shape == "inland":
        return np.ones_like(u)
    if shape == "continent":
        uw, vw = noise.warp2(u, v, 0.18, 2.0)
        d = np.hypot(uw - 0.5, vw - 0.5) / 0.5
        return 1.0 - smoothstep(0.55, 1.0, d)
    if shape == "coast":
        ang = rng.uniform(0, 2 * np.pi)
        dx, dy = np.cos(ang), np.sin(ang)
        uw, vw = noise.warp2(u, v, 0.22, 1.6)
        t = (uw - 0.5) * dx + (vw - 0.5) * dy
        return smoothstep(-0.35, 0.25, t)
    # archipelago: several noisy blobs
    uw, vw = noise.warp2(u, v, 0.25, 3.0)
    n = noise.fbm2(uw * 3.5 + 5.0, vw * 3.5 - 2.0, 4)
    edge = 1.0 - smoothstep(0.7, 1.0, np.hypot(u - 0.5, v - 0.5) / 0.5)
    return np.clip(smoothstep(-0.12, 0.25, n) * edge, 0.0, 1.0)


def simulate_tectonics(cfg: OFMConfig) -> TectonicResult:
    wc, tc = cfg.world, cfg.tectonics
    size = wc.size
    rng = np.random.default_rng(derive_seed(wc.seed, "tectonics"))
    noise = Noise(derive_seed(wc.seed, "tectonics-noise"))

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    u = (xx + 0.5) / size
    v = (yy + 0.5) / size

    # Plates: warped Voronoi cells, seeds relaxed once so plates have similar sizes.
    n_plates = max(2, tc.plates)
    seeds = rng.uniform(0.05, 0.95, size=(n_plates, 2))
    uw, vw = noise.warp2(u, v, tc.warp, 2.5)
    for _ in range(2):
        d2 = (uw[..., None] - seeds[:, 0]) ** 2 + (vw[..., None] - seeds[:, 1]) ** 2
        plate = np.argmin(d2, axis=-1)
        for p in range(n_plates):
            sel = plate == p
            if sel.any():
                seeds[p] = 0.5 * seeds[p] + 0.5 * np.array([u[sel].mean(), v[sel].mean()])
    d2 = (uw[..., None] - seeds[:, 0]) ** 2 + (vw[..., None] - seeds[:, 1]) ** 2
    plate = np.argmin(d2, axis=-1)

    vel = rng.normal(size=(n_plates, 2))
    vel /= np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9
    vel *= rng.uniform(0.4, 1.0, size=(n_plates, 1))
    plate_relief = rng.normal(0.0, 1.0, n_plates)

    conv = np.zeros((size, size))
    count = np.zeros((size, size))
    for dy, dx in NEIGHBORS4:
        nb = shifted(plate, dy, dx)
        diff = nb != plate
        rel = vel[plate] - vel[nb]
        c = rel[..., 0] * dx + rel[..., 1] * dy
        conv += np.where(diff, c, 0.0)
        count += diff
    conv = np.where(count > 0, conv / np.maximum(count, 1), 0.0)

    belt = max(1.5, tc.belt_width * size)
    conv_pos = blur(np.maximum(conv, 0.0), belt)
    conv_neg = blur(np.maximum(-conv, 0.0), belt * 0.6)
    orogeny = normalize01(conv_pos) ** 0.8
    rift = normalize01(conv_neg)

    # Mountain ranges: ridged multifractal following the belts.
    extent = size * wc.cell_size
    f_ridge = extent / tc.ridge_wavelength
    f_hill = extent / tc.hill_wavelength
    ridges = noise.ridged2(uw * f_ridge + 3.1, vw * f_ridge - 7.4, 6)
    massif = noise.fbm2(u * f_ridge * 0.8 + 9.0, v * f_ridge * 0.8 + 1.0, 3) * 0.5 + 0.5
    uplift = tc.mountain_height * orogeny * (0.45 + 0.55 * ridges) * (0.6 + 0.4 * massif)

    shape = _shape_mask(wc.shape, u, v, noise, rng)
    continental = blur(shape, size * 0.07)
    base = tc.ocean_depth + (tc.land_base - tc.ocean_depth) * continental
    base += blur(plate_relief[plate], size * 0.04) * 60.0

    hills = noise.fbm2(u * f_hill + 1.7, v * f_hill + 8.3, 6, gain=0.42) * tc.hill_amplitude
    hills *= 0.45 + 0.55 * continental
    plains = noise.fbm2(u * f_hill * 0.4 - 4.0, v * f_hill * 0.4 + 2.0, 3, gain=0.4) * tc.hill_amplitude * 0.8

    height = base + plains + hills + uplift * (0.35 + 0.65 * continental) - rift * tc.rift_depth

    # Sea level placement so that ~land_fraction of the map is land.
    if wc.shape == "inland":
        height = height - height.min() + wc.sea_level + 25.0
    else:
        q = np.quantile(height, 1.0 - wc.land_fraction)
        height = height - q + wc.sea_level
    # Keep the sea floor shallow enough for gameplay shores.
    height = np.where(height < wc.sea_level, wc.sea_level + (height - wc.sea_level) * 0.55, height)

    return TectonicResult(
        height=height.astype(np.float64),
        uplift=uplift.astype(np.float64),
        plate_id=plate.astype(np.int16),
        convergence=conv_pos - conv_neg,
    )


def carve_basins(height: np.ndarray, cfg: OFMConfig) -> np.ndarray:
    """Closed depressions (grabens, cirques, kettle basins), applied after erosion.

    Each basin is lowered below the lowest point of its rim, so it is a true
    closed depression; hydrology later decides whether it holds an open lake,
    a saline terminal lake or a dry playa from its water balance.
    """
    from ..grid import slope as grid_slope
    tc, wc = cfg.tectonics, cfg.world
    rng = np.random.default_rng(derive_seed(wc.seed, "basins"))
    noise = Noise(derive_seed(wc.seed, "basins-noise"))
    size = height.shape[0]
    cell = wc.cell_size
    land = height > wc.sea_level + 15.0
    area_km2 = land.sum() * cell * cell / 1e6
    n = int(round(tc.basins_per_km2 * area_km2))
    if n <= 0 or not land.any():
        return height
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    s = blur(grid_slope(height, cell), 3.0)
    margin = max(4, int(tc.basin_radius / cell * 1.5))
    inner = np.zeros_like(land)
    inner[margin:-margin, margin:-margin] = True
    cand = np.flatnonzero((land & inner).reshape(-1))
    if cand.size == 0:
        return height
    weights = 1.0 / (0.05 + s.reshape(-1)[cand]) ** 2
    picks = rng.choice(cand, size=min(n * 8, cand.size), replace=False, p=weights / weights.sum())
    placed: list[tuple[float, float]] = []
    out = height.copy()
    for c in picks:
        if len(placed) >= n:
            break
        j, i = divmod(int(c), size)
        if any(np.hypot(i - pi, j - pj) < size * 0.12 for pi, pj in placed):
            continue
        radius = rng.uniform(tc.basin_radius * 0.6, tc.basin_radius * 1.3) / cell
        depth = rng.uniform(0.5, 1.0) * tc.basin_depth
        stretch = rng.uniform(0.6, 1.0)
        ang = rng.uniform(0, np.pi)
        dx, dy = xx - i, yy - j
        ax = (dx * np.cos(ang) + dy * np.sin(ang)) / radius
        ay = (-dx * np.sin(ang) + dy * np.cos(ang)) / (radius * stretch)
        r = np.hypot(ax, ay) * (1.0 + 0.2 * noise.fbm2(xx / size * 9 + c, yy / size * 9, 3))
        rim = (r > 1.0) & (r < 1.15)
        if not rim.any():
            continue
        rim_min = float(out[rim].min())
        if rim_min <= wc.sea_level + 2.0:
            continue
        # Bowl below the rim, then a gentle cone outside it so the uphill side
        # becomes a shore slope rather than a cut wall.
        inside = rim_min - 1.0 - depth * np.clip(1.0 - r * r, 0.0, None) ** 0.8
        outside = rim_min - 1.0 + np.clip(r - 1.0, 0.0, None) * radius * cell * np.tan(np.radians(16.0))
        target = np.where(r < 1.0, inside, outside)
        blend = 1.0 - smoothstep(2.2, 3.0, r)
        out = out + blend * (np.minimum(out, target) - out)
        placed.append((i, j))
    return out
