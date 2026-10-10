"""Biome classification (Whittaker diagram) and ground-material splat weights."""
from __future__ import annotations

import numpy as np

from ..grid import blur, chamfer_distance, slope, smoothstep
from ..noise import Noise, derive_seed

# id, key, colour, vegetation density
BIOMES = [
    (0, "ocean", (38, 74, 122), 0.0),
    (1, "lake", (52, 104, 150), 0.0),
    (2, "beach", (222, 205, 150), 0.02),
    (3, "glacier", (236, 242, 248), 0.0),
    (4, "tundra", (160, 166, 140), 0.08),
    (5, "boreal_forest", (58, 92, 66), 0.75),
    (6, "temperate_forest", (68, 120, 58), 0.85),
    (7, "temperate_rainforest", (40, 104, 62), 1.0),
    (8, "grassland", (142, 168, 86), 0.25),
    (9, "shrubland", (150, 148, 92), 0.35),
    (10, "desert", (214, 190, 132), 0.03),
    (11, "savanna", (178, 168, 92), 0.2),
    (12, "tropical_seasonal_forest", (98, 140, 52), 0.8),
    (13, "tropical_rainforest", (30, 110, 44), 1.0),
    (14, "alpine_rock", (130, 126, 120), 0.02),
    (15, "wetland", (84, 112, 82), 0.5),
    (16, "salt_flat", (232, 226, 214), 0.0),
]
BIOME_NAMES = [b[1] for b in BIOMES]
BIOME_COLORS = np.array([b[2] for b in BIOMES], dtype=np.uint8)
BIOME_VEG = np.array([b[3] for b in BIOMES])
B = {name: i for i, name in enumerate(BIOME_NAMES)}

# Ground material layers exported as splat maps (2 x RGBA)
MATERIALS = [
    ("grass", (96, 128, 60)),
    ("forest_floor", (70, 76, 46)),
    ("dirt", (122, 98, 70)),
    ("sand", (214, 196, 148)),
    ("rock", (120, 116, 110)),
    ("snow", (236, 240, 245)),
    ("mud", (82, 70, 56)),
    ("gravel", (150, 144, 134)),
]
MATERIAL_NAMES = [m[0] for m in MATERIALS]


def classify(height, temperature, precipitation, ocean, lake_mask, river_mask, playa,
             cell: float, sea_level: float, seed: int = 0) -> np.ndarray:
    size = height.shape[0]
    noise = Noise(derive_seed(seed, "biome"))
    yy, xx = np.mgrid[0:size, 0:size] / size
    jitter = noise.fbm2(xx * 10.0, yy * 10.0, 4)
    t = temperature + jitter * 1.5
    p = precipitation * (1.0 + 0.22 * noise.fbm2(xx * 8.0 + 40.0, yy * 8.0, 4))
    biome = np.full(height.shape, B["grassland"], dtype=np.int16)

    # Whittaker: temperature bands then precipitation thresholds.
    cold = t < 0.0
    cool = (t >= 0.0) & (t < 7.0)
    temperate = (t >= 7.0) & (t < 19.0)
    hot = t >= 19.0
    biome[cold] = B["tundra"]
    biome[cool & (p >= 450)] = B["boreal_forest"]
    biome[cool & (p < 450)] = B["tundra"]
    biome[temperate & (p < 280)] = B["desert"]
    biome[temperate & (p >= 280) & (p < 520)] = B["shrubland"]
    biome[temperate & (p >= 520) & (p < 800)] = B["grassland"]
    biome[temperate & (p >= 800) & (p < 1900)] = B["temperate_forest"]
    biome[temperate & (p >= 1900)] = B["temperate_rainforest"]
    biome[hot & (p < 350)] = B["desert"]
    biome[hot & (p >= 350) & (p < 900)] = B["savanna"]
    biome[hot & (p >= 900) & (p < 2000)] = B["tropical_seasonal_forest"]
    biome[hot & (p >= 2000)] = B["tropical_rainforest"]

    s = slope(height, cell)
    biome[(t < -4.0) & (p > 250)] = B["glacier"]
    biome[(s > np.tan(np.radians(46))) & ~ocean] = B["alpine_rock"]

    water = ocean | lake_mask
    near_water = chamfer_distance(water | river_mask, max_dist=4) <= 2
    flat = s < 0.06
    biome[near_water & flat & (p > 650) & (t > 0) & ~water] = B["wetland"]
    d_ocean = chamfer_distance(ocean, max_dist=4)
    beach = (d_ocean <= 2) & (height < sea_level + 6.0) & (s < 0.25) & ~ocean
    biome[beach] = B["beach"]
    biome[playa] = B["salt_flat"]
    biome[lake_mask] = B["lake"]
    biome[ocean] = B["ocean"]
    return biome


def splat_weights(height, biome, temperature, cell: float, hardness: np.ndarray,
                  river_mask: np.ndarray) -> np.ndarray:
    """Per-cell weights for MATERIALS, shape (H, W, 8), rows sum to 1."""
    n = len(MATERIALS)
    w = np.zeros(height.shape + (n,))
    s_deg = np.degrees(np.arctan(slope(height, cell)))
    g, ff, dirt, sand, rock, snow, mud, gravel = range(n)

    def add(mask, idx, val=1.0):
        w[..., idx] += np.where(mask, val, 0.0)

    add(np.isin(biome, [B["grassland"], B["savanna"], B["shrubland"], B["wetland"]]), g)
    add(np.isin(biome, [B["boreal_forest"], B["temperate_forest"], B["temperate_rainforest"],
                        B["tropical_seasonal_forest"], B["tropical_rainforest"]]), ff)
    add(np.isin(biome, [B["shrubland"], B["savanna"], B["tundra"]]), dirt, 0.5)
    add(np.isin(biome, [B["beach"], B["desert"]]), sand)
    add(np.isin(biome, [B["ocean"], B["lake"]]), sand, 0.6)
    add(np.isin(biome, [B["ocean"], B["lake"]]), mud, 0.4)
    add(biome == B["salt_flat"], gravel, 0.4)
    add(biome == B["salt_flat"], sand, 0.6)
    add(biome == B["tundra"], gravel, 0.6)
    add(biome == B["wetland"], mud, 0.8)
    add(biome == B["glacier"], snow, 2.0)
    add(river_mask, gravel, 1.2)

    rockiness = smoothstep(30.0, 48.0, s_deg) * (0.6 + 0.4 * hardness)
    w[..., rock] += rockiness * 3.0 + np.where(biome == B["alpine_rock"], 2.0, 0.0)
    snowy = smoothstep(-1.0, -5.0, temperature) * (1.0 - smoothstep(40.0, 55.0, s_deg))
    w[..., snow] += snowy * 3.0
    w[..., dirt] += smoothstep(18.0, 32.0, s_deg) * 0.6

    for k in range(n):
        w[..., k] = blur(w[..., k], 0.8)
    total = w.sum(axis=-1, keepdims=True)
    w = np.where(total > 1e-6, w / np.maximum(total, 1e-6), 0.0)
    w[..., g] += np.where(total[..., 0] <= 1e-6, 1.0, 0.0)
    return w
