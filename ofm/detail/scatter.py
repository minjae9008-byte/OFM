"""Ecological vegetation and rock scatter.

Instance density follows simple ecological rules instead of uniform noise:
precipitation and temperature set canopy density per biome, the tree line
follows temperature, trees avoid steep rock and water, riparian bushes
cluster along rivers, and boulders collect at the foot of cliffs (talus).
Positions come from a jittered grid with a minimum spacing per type; each
instance is then checked on the exported ground (detail slope, water).
"""
from __future__ import annotations

import numpy as np

from ..grid import bilinear, blur, chamfer_distance, gradient, slope, smoothstep
from ..noise import Noise, derive_seed
from ..physics.biome import B, BIOME_VEG
from .lakes import lake_surface, under_lake

# type -> (min spacing m, base scale)
SCATTER_TYPES = {
    "conifer": (7.0, 1.0),
    "broadleaf": (8.0, 1.0),
    "palm": (9.0, 1.0),
    "bush": (4.0, 0.6),
    "dead_tree": (14.0, 0.9),
    "cactus": (10.0, 0.7),
    "boulder": (9.0, 1.0),
    "reed": (3.0, 0.5),
}


def _density_maps(world) -> dict[str, np.ndarray]:
    t = world.temperature
    biome = world.biome
    s_deg = np.degrees(np.arctan(slope(world.height, world.cell)))
    water = world.water
    near_river = chamfer_distance(world.river_mask | world.lake_mask, max_dist=5)
    veg = BIOME_VEG[biome] * (1.0 - smoothstep(30.0, 42.0, s_deg)) * (~water)
    cold = smoothstep(9.0, 2.0, t)
    hot = smoothstep(17.0, 23.0, t)
    forest = np.isin(biome, [B["boreal_forest"], B["temperate_forest"], B["temperate_rainforest"],
                             B["tropical_seasonal_forest"], B["tropical_rainforest"]])
    tree_line = smoothstep(-1.0, 2.5, t)
    maps = {
        "conifer": veg * tree_line * (cold * 0.9 + 0.15) * np.where(forest | (biome == B["tundra"]), 1.0, 0.25),
        "broadleaf": veg * (1.0 - cold) * (1.0 - hot * 0.6) * np.where(forest, 1.0, 0.18),
        "palm": veg * hot * np.isin(biome, [B["tropical_rainforest"], B["tropical_seasonal_forest"],
                                            B["savanna"], B["beach"]]),
        "bush": np.clip(veg * 0.6 + np.isin(biome, [B["shrubland"], B["savanna"]]) * 0.5, 0, 1) * (~water),
        "dead_tree": (biome == B["wetland"]) * 0.25 + (biome == B["salt_flat"]) * 0.0
                     + np.isin(biome, [B["desert"], B["shrubland"]]) * 0.05,
        "cactus": (biome == B["desert"]) * hot * 0.5,
        "reed": ((near_river <= 1.5) & ~water & (s_deg < 10)) * 0.9 + (biome == B["wetland"]) * 0.6,
    }
    cliff_foot = blur(world.cliff_mask.astype(np.float64), 1.5) * (s_deg < 30) * (~water)
    maps["boulder"] = np.clip(cliff_foot * 1.6 + (biome == B["alpine_rock"]) * 0.3
                              + (biome == B["tundra"]) * 0.1, 0, 1)
    maps["bush"] = np.clip(maps["bush"] + ((near_river <= 2.5) & ~water) * 0.4, 0, 1)
    return maps


def scatter_vegetation(world) -> dict[str, np.ndarray]:
    """Returns {type: array (n, 5) of x, y, z, yaw, scale}."""
    cfg = world.cfg
    rng = np.random.default_rng(derive_seed(cfg.world.seed, "scatter"))
    noise = Noise(derive_seed(cfg.world.seed, "scatter-noise"))
    maps = _density_maps(world)
    extent = world.extent
    budget = cfg.detail.scatter_max
    # The density maps live on the simulation grid; every instance is checked again on the
    # exported ground: the detail pass adds terrace risers, canyon walls and carved channels
    # the 16 m grid smooths over, and an instance must not stand on them or in water.
    sp = world.hires_spacing
    gx, gy = gradient(world.hires, sp)
    slope_hi = np.degrees(np.arctan(np.hypot(gx, gy)))
    sea = cfg.world.sea_level
    lake_level = np.where(world.lake_mask, world.lake_level, -np.inf) if world.lake_mask is not None \
        else np.full((world.size, world.size), -np.inf)
    surface = lake_surface(world)          # lake sheets reach past the lake cells to the shoreline
    raw = {}
    for k_idx, (kind, (spacing, base_scale)) in enumerate(SCATTER_TYPES.items()):
        m = maps[kind]
        if m.max() <= 0.01:
            continue
        n_side = int(extent / spacing)
        gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side))
        x = (gx + rng.uniform(0.15, 0.85, gx.shape)) * spacing
        y = (gy + rng.uniform(0.15, 0.85, gy.shape)) * spacing
        x, y = x.ravel(), y.ravel()
        d = bilinear(m, x / world.cell - 0.5, y / world.cell - 0.5)
        # clumping: forests grow in patches with clearings
        clump = 0.65 + 0.7 * noise.fbm2(x / 180.0 + k_idx * 13.7, y / 180.0, 3)
        keep = rng.random(x.shape) < np.clip(d * clump, 0, 1) * 0.55
        x, y = x[keep], y[keep]
        if len(x) == 0:
            continue
        z = bilinear(world.hires, x / sp, y / sp)
        s_hi = bilinear(slope_hi, x / sp, y / sp)
        lo, hi = (40.0, 52.0) if kind == "boulder" else (32.0, 44.0)
        ci = np.clip((x / world.cell).astype(int), 0, world.size - 1)
        cj = np.clip((y / world.cell).astype(int), 0, world.size - 1)
        wet = (z < sea + 0.05) | (z < lake_level[cj, ci] + 0.05) | under_lake(world, x, y, z, surface)
        if kind != "reed":
            wet |= world.river_mask[cj, ci]
        ok = ~wet & (rng.random(len(x)) < 1.0 - smoothstep(lo, hi, s_hi))
        x, y, z = x[ok], y[ok], z[ok]
        if len(x) == 0:
            continue
        yaw = rng.uniform(0, 360, len(x))
        scale = base_scale * rng.uniform(0.75, 1.3, len(x))
        raw[kind] = np.stack([x, y, z, yaw, scale], axis=1)
    total = sum(len(v) for v in raw.values())
    if total > budget:
        frac = budget / total
        for kind in raw:
            sel = rng.random(len(raw[kind])) < frac
            raw[kind] = raw[kind][sel]
    return raw
