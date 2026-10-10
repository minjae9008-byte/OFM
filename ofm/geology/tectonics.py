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
from ..grid import bilinear, NEIGHBORS4, blur, label, normalize01, shifted, smoothstep
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
        if tc.coast_roughness > 0:
            # Fractal domain warp in a band around sea level: the shoreline is displaced
            # horizontally at every scale of the fBm, carving bays, headlands and islets
            # (natural coastlines have a box-counting dimension of ~1.1-1.3; the smooth
            # continental falloff alone gives ~1.0). Inland relief is left untouched.
            f = size * wc.cell_size / max(tc.coast_wavelength, 1.0)
            disp = tc.coast_roughness / wc.cell_size
            wx = noise.fbm2(u * f + 13.7, v * f - 5.1, 7, gain=0.62)
            wy = noise.fbm2(u * f - 41.2, v * f + 22.8, 7, gain=0.62)
            warped = bilinear(height, xx + wx * disp, yy + wy * disp)
            near = np.exp(-np.abs(height - wc.sea_level) / max(tc.coast_band, 1.0))
            height = height + near * (warped - height)
    # Keep the sea floor shallow enough for gameplay shores.
    height = np.where(height < wc.sea_level, wc.sea_level + (height - wc.sea_level) * 0.55, height)

    return TectonicResult(
        height=height.astype(np.float64),
        uplift=uplift.astype(np.float64),
        plate_id=plate.astype(np.int16),
        convergence=conv_pos - conv_neg,
    )


def carve_basins(height: np.ndarray, cfg: OFMConfig) -> np.ndarray:
    """Closed depressions (tectonic sags, glacial and dammed valley basins), applied after erosion.

    Basins prefer broadly gentle ground (lowlands, plateaus, wide valley floors
    after fluvial erosion) and are elongated along the local downslope (valley)
    direction. The lake level is set just below the lower part of the
    surrounding rim; a smooth bowl (zero at the rim) deepens the floor, so the
    terrain keeps its texture and the shoreline follows the contours instead of
    a carved crater. Where the rim is lower than the lake (the outlet valley) a
    moraine/landslide dam with a 12 degree apron closes it, so every basin is a
    true closed depression. Hydrology later decides from the water balance
    whether a basin holds an open lake, a saline terminal lake or a dry playa.
    Sites that would need a deep gouge or a high dam are skipped; on worlds
    without any gentle site the least demanding one is still used (relaxed
    limits) so every map keeps at least one basin.
    """
    from ..grid import gradient
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
    # Broadly gentle ground at the basin scale.
    s = blur(grid_slope(height, cell), max(3.0, 0.5 * tc.basin_radius / cell))
    gx, gy = gradient(blur(height, 3.0), cell)
    margin = max(4, int(tc.basin_radius / cell * 1.5))
    inner = np.zeros_like(land)
    inner[margin:-margin, margin:-margin] = True
    cand = np.flatnonzero((land & inner).reshape(-1))
    if cand.size == 0:
        return height
    weights = 1.0 / (0.05 + s.reshape(-1)[cand]) ** 2
    picks = rng.choice(cand, size=min(n * 20, cand.size), replace=False, p=weights / weights.sum())
    sites = []
    for c in picks:
        j, i = divmod(int(c), size)
        sites.append((int(c), i, j,
                      rng.uniform(tc.basin_radius * 0.6, tc.basin_radius * 1.3) / cell,   # radius (cells)
                      rng.uniform(0.5, 1.0) * tc.basin_depth,                            # depth
                      rng.uniform(0.55, 0.9),                                            # stretch
                      float(np.arctan2(gy[j, i], gx[j, i])) + rng.normal(0.0, 0.35)))   # long axis

    def dig(out, site, relax):
        """Carve one basin into a window of ``out``; None if the site is unsuitable."""
        c, i, j, radius, depth, stretch, ang = site
        radius = radius / relax                       # relaxed pass: smaller basins on steep ground
        half = int(np.ceil(2.0 * radius)) + 2
        j0, j1, i0, i1 = max(0, j - half), min(size, j + half + 1), max(0, i - half), min(size, i + half + 1)
        yy, xx = np.mgrid[j0:j1, i0:i1].astype(np.float64)
        win = out[j0:j1, i0:i1]
        dx, dy = xx - i, yy - j
        ax = (dx * np.cos(ang) + dy * np.sin(ang)) / radius
        ay = (-dx * np.sin(ang) + dy * np.cos(ang)) / (radius * stretch)
        r = np.hypot(ax, ay) * (1.0 + 0.2 * noise.fbm2(xx / size * 9 + c, yy / size * 9, 3))
        ring = (r > 1.05) & (r < 1.25)
        if not ring.any():
            return None
        # Lake level just below the lower part of the rim, at most half the basin
        # depth above the present floor (valley lakes back up the valley).
        floor_z = float(out[j, i])
        level = min(float(np.quantile(win[ring], 0.3)) - 0.5, floor_z + 0.5 * depth)
        amp = max(0.0, floor_z - level) + 0.5 * depth
        if level <= wc.sea_level + 2.0 or amp > 1.5 * relax * tc.basin_depth:
            return None                  # too steep a site: the bowl would gouge the slope
        # Smooth bowl (zero at the rim): the terrain keeps its texture and the
        # shoreline follows the contour at the lake level.
        dug = win - amp * np.clip(1.0 - r * r, 0.0, 1.0)
        lab, _ = label((dug < level) & (r < 1.05), diagonal=True)
        wet = lab == lab[j - j0, i - i0]
        top = level + 1.5
        if wet.sum() < 6 or top - float(win[ring].min()) > 1.5 * relax * tc.basin_depth:
            return None                  # outlet gorge too deep to dam convincingly
        # Dam only where the rim ring is lower than the lake, with a 12 degree apron.
        apron = np.clip(np.abs(r - 1.15) - 0.1, 0.0, None) * radius * cell * np.tan(np.radians(12.0))
        new = out.copy()
        new[j0:j1, i0:i1] = np.where(~wet & (r > 0.9) & (r < 1.6), np.maximum(dug, top - apron), dug)
        return new

    placed: list[tuple[float, float]] = []
    out = height.copy()
    for relax in (1.0, 2.5):
        for site in sites:
            if len(placed) >= n:
                break
            if any(np.hypot(site[1] - pi, site[2] - pj) < size * 0.12 for pi, pj in placed):
                continue
            new = dig(out, site, relax)
            if new is not None:
                out = new
                placed.append((site[1], site[2]))
        if placed:
            break
    return out
