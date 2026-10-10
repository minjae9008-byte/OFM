"""Detail amplification: the export-resolution heightfield.

The simulation grid is upsampled with Catmull-Rom interpolation and gains
detail that depends on geology instead of uniform noise:
  * hard, steep rock gets sharp ridged detail; soil gets gentle undulation;
  * water surfaces and channels stay smooth;
  * steep slopes are terraced along the strata, so hard layers stand out as
    vertical risers with benches on top (stair-step canyon walls);
  * export-resolution landforms follow (``hires.refine``): fine droplet rills,
    scree aprons, floodplains, beaches and shore platforms, then river
    channels are kept open and the sea floor / lake beds below their water.
The volumetric layer samples this same field, so terrain and meshes agree.
"""
from __future__ import annotations

import numpy as np

from ..grid import bicubic, bilinear, gradient, smoothstep
from ..noise import Noise, derive_seed
from .hires import refine


def strata_terrace(h: np.ndarray, fold: np.ndarray, strata, weight: np.ndarray,
                   slope_tan: np.ndarray | None = None, spacing: float = 0.0) -> np.ndarray:
    bot, top = strata.layer_bounds(h, fold)
    idx = strata.layer_index(h, fold)
    thick = np.maximum(top - bot, 1e-3)
    u = np.clip((h - bot) / thick, 0.0, 1.0)
    hard = strata.hardness_table[idx]
    k = np.clip(1.0 - 0.8 * hard, 0.2, 1.0)            # riser fraction of the layer
    if slope_tan is not None and spacing > 0.0:
        # a riser spans k * thick / slope metres; keep it >= 1.5 cells so thin layers
        # on steep ground do not alias into dotted stair-steps
        k = np.maximum(k, np.clip(1.5 * spacing * slope_tan / thick, 0.0, 1.0))
    g = smoothstep(0.0, 1.0, np.clip(u / k, 0.0, 1.0))
    g = g + (u - g) * smoothstep(0.85, 1.0, k)          # thick risers fade to the plain slope
    terraced = bot + g * thick
    w = np.where(idx == strata.basement_index, 0.0, weight)
    return h + w * (terraced - h)


def amplify(world) -> tuple[np.ndarray, float]:
    cfg = world.cfg
    dc, cc = cfg.detail, cfg.cliffs
    f = dc.upscale
    n = world.size * f + 1
    spacing = world.extent / (n - 1)
    cell = world.cell
    coords = np.arange(n) * spacing
    X, Y = np.meshgrid(coords, coords)
    U = X / cell - 0.5
    V = Y / cell - 0.5

    base = bicubic(world.height, U, V)
    if f == 1:
        base = bilinear(world.height, U, V)
    fold = bilinear(world.strata.fold, U, V)
    hard = world.strata.hardness(base, fold)
    gx, gy = gradient(base, spacing)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

    water = bilinear(world.water.astype(np.float64), U, V)
    river = bilinear(world.river_mask.astype(np.float64), U, V)
    calm = np.clip(1.0 - 2.0 * np.maximum(water, river), 0.0, 1.0)

    noise = Noise(derive_seed(cfg.world.seed, "amplify"))
    rock = smoothstep(22.0, 42.0, slope_deg) * (0.35 + 0.65 * hard)
    wl = max(spacing * 6.0, 24.0)
    ridged = noise.ridged2(X / wl, Y / wl, 4) - 0.5
    soft = noise.fbm2(X / (wl * 1.7) + 50.0, Y / (wl * 1.7), 4)
    detail = ridged * rock * dc.rock_detail * cell + soft * (1.0 - rock) * dc.soil_detail * cell
    h = base + detail * calm

    steep = smoothstep(28.0, 46.0, slope_deg) * calm
    h = strata_terrace(h, fold, world.strata, steep * cc.terrace_strength, np.hypot(gx, gy), spacing)

    # export-resolution landforms: rills, scree, floodplains, beaches (detail/hires.py)
    h = refine(world, h, fold, U, V, spacing)

    if world.lake_mask is not None and world.lake_mask.any():
        # Outside lakes use the terrain itself, which meets the lake level at the shore.
        level = np.where(world.lake_mask, world.lake_level, world.height)
        level_hi = bilinear(level, U, V)
        lake_hi = bilinear(world.lake_mask.astype(np.float64), U, V) > 0.5
        h = np.where(lake_hi, np.minimum(h, level_hi - 0.4), h)
    return h, spacing
