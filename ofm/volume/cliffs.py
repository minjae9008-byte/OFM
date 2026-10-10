"""Cliff detection and physically motivated placement of rock landmarks.

* Sea stacks are left where waves isolate headlands of steep coastal cliffs.
* Hoodoos (spires) and buttes survive where a hard caprock protects soft rock
  in dry, steep badlands.
* Natural arches form in hard, insoluble sandstone fins.
"""
from __future__ import annotations

import numpy as np

from ..grid import chamfer_distance, slope
from ..noise import derive_seed
from .features import make_feature


def cliff_masks(world) -> tuple[np.ndarray, np.ndarray]:
    cfg = world.cfg
    s_deg = np.degrees(np.arctan(slope(world.height, world.cell)))
    land = ~world.water
    cliff = land & (s_deg > cfg.cliffs.cliff_slope_deg - 8.0)
    d_ocean = chamfer_distance(world.ocean, max_dist=4)
    coast = land & (d_ocean <= 2.5) & (s_deg > 20.0) & (world.height > cfg.world.sea_level + 4.0)
    return cliff, coast


def place_procedural_features(world) -> list:
    cfg = world.cfg
    rng = np.random.default_rng(derive_seed(cfg.world.seed, "features"))
    cell = world.cell
    size = world.size
    h = world.height
    s_deg = np.degrees(np.arctan(slope(h, cell)))
    feats = []
    taken: list[tuple[float, float]] = []

    def far(x, y, d=400.0):
        return all(np.hypot(x - a, y - b) > d for a, b in taken)

    margin = 6

    def pick(mask, n, kind, scale=1.0):
        jj, ii = np.nonzero(mask)
        ok = (ii > margin) & (ii < size - margin) & (jj > margin) & (jj < size - margin)
        jj, ii = jj[ok], ii[ok]
        for k in rng.permutation(len(ii))[: n * 30]:
            if sum(f.kind == kind for f in feats) >= n:
                break
            x, y = (ii[k] + 0.5) * cell, (jj[k] + 0.5) * cell
            if not far(x, y):
                continue
            f = make_feature(world, kind, x, y, rng, scale)
            f.id = f"feature_{len(world.features) + len(feats)}"
            feats.append(f)
            taken.append((x, y))

    # Sea stacks: shallow water right next to coastal cliffs.
    if world.coast_cliff is not None and world.coast_cliff.any():
        near_cliff = chamfer_distance(world.coast_cliff, max_dist=4) <= 2.5
        depth = cfg.world.sea_level - h
        pick(world.ocean & near_cliff & (depth > 1.0) & (depth < 10.0), 2, "sea_stack")
    # Hoodoos / buttes: steep, hard rock in dry climate.
    hard = world.hardness > 0.6
    dry = world.precipitation < np.quantile(world.precipitation[~world.water], 0.45) if (~world.water).any() else hard
    pick(~world.water & hard & dry & (s_deg > 22) & (s_deg < 40), 2, "spire")
    pick(~world.water & hard & dry & (s_deg < 12) & (h > np.quantile(h, 0.5)), 1, "butte")
    # Arches: hard, insoluble rock on steep ground.
    insoluble = world.strata.solubility(h) < 0.2
    pick(~world.water & hard & insoluble & (s_deg > 18) & (s_deg < 35), 1, "arch")
    return feats
