"""Sedimentary strata over a crystalline basement.

The layer stack is a repeating sedimentary cycle. Layer boundaries are folded
by tectonic uplift, dipped regionally and undulated with noise, so erosion of
the surface exposes different rocks in different places. Rock hardness drives
differential erosion (cliffs, mesas, overhangs) and solubility drives karst
cave formation.
"""
from __future__ import annotations

import numpy as np

from ..config import OFMConfig
from ..grid import bilinear
from ..noise import Noise, derive_seed


class Strata:
    def __init__(self, cfg: OFMConfig, uplift: np.ndarray, cell: float):
        sc = cfg.strata
        self.layers = list(sc.layers)
        self.names = [layer.name for layer in self.layers] + [sc.basement_name]
        self.hardness_table = np.array([layer.hardness for layer in self.layers] + [sc.basement_hardness])
        self.solubility_table = np.array([layer.solubility for layer in self.layers] + [0.0])
        self.color_table = np.array([layer.color for layer in self.layers] + [sc.basement_color], dtype=np.float64)
        thick = np.array([max(1.0, layer.thickness) for layer in self.layers])
        self.cum = np.concatenate([[0.0], np.cumsum(thick)])
        self.period = float(self.cum[-1])
        self.basement_level = sc.basement_level
        self.basement_index = len(self.layers)
        self.cell = cell

        size = uplift.shape[0]
        noise = Noise(derive_seed(cfg.world.seed, "strata"))
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        x_m = (xx + 0.5) * cell
        y_m = (yy + 0.5) * cell
        rng = np.random.default_rng(derive_seed(cfg.world.seed, "strata-dip"))
        ang = rng.uniform(0, 2 * np.pi)
        dip = sc.dip * (np.cos(ang) * x_m + np.sin(ang) * y_m)
        und = noise.fbm2(xx / size * 6.0, yy / size * 6.0, 4) * sc.undulation
        # fold(x, y): vertical offset of the stack at each column
        self.fold = uplift * sc.fold_factor + dip + und

    # -- queries ------------------------------------------------------------
    def fold_at(self, x_m, y_m) -> np.ndarray:
        return bilinear(self.fold, np.asarray(x_m) / self.cell - 0.5, np.asarray(y_m) / self.cell - 0.5)

    def layer_index(self, z, fold) -> np.ndarray:
        """Layer index for elevations ``z`` given the fold offsets at those columns."""
        r = np.asarray(z, dtype=np.float64) - fold
        depth = np.mod(-r, self.period)
        idx = np.searchsorted(self.cum, depth, side="right") - 1
        idx = np.clip(idx, 0, len(self.layers) - 1)
        return np.where(r < self.basement_level, self.basement_index, idx)

    def layer_at_grid(self, z: np.ndarray) -> np.ndarray:
        return self.layer_index(z, self.fold)

    def hardness(self, z, fold=None) -> np.ndarray:
        fold = self.fold if fold is None else fold
        return self.hardness_table[self.layer_index(z, fold)]

    def solubility(self, z, fold=None) -> np.ndarray:
        fold = self.fold if fold is None else fold
        return self.solubility_table[self.layer_index(z, fold)]

    def layer_bounds(self, z, fold) -> tuple[np.ndarray, np.ndarray]:
        """Bottom and top elevation of the layer containing each z (basement: tall slab)."""
        r = np.asarray(z, dtype=np.float64) - fold
        depth = np.mod(-r, self.period)
        idx = np.clip(np.searchsorted(self.cum, depth, side="right") - 1, 0, len(self.layers) - 1)
        top_r = r + (depth - self.cum[idx])        # moving up reduces depth to the layer top
        bot_r = r - (self.cum[idx + 1] - depth)
        basement = r < self.basement_level
        top_r = np.where(basement, self.basement_level, top_r)
        bot_r = np.where(basement, self.basement_level - 1e4, bot_r)
        return bot_r + fold, top_r + fold

    def describe(self) -> list[dict]:
        out = []
        for i, name in enumerate(self.names):
            out.append({
                "index": i,
                "name": name,
                "hardness": float(self.hardness_table[i]),
                "solubility": float(self.solubility_table[i]),
                "color": [int(c) for c in self.color_table[i]],
            })
        return out
