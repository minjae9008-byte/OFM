"""Stream-power landscape evolution (fluvial erosion).

CONTRACT (used by pipeline.generate_base):

    stream_power_erosion(height, uplift, strata, cfg) -> np.ndarray

* ``height``: (N, N) float64 metres after tectonics (sea level = cfg.world.sea_level).
* ``uplift``: (N, N) metres of orogenic uplift from tectonics (>= 0); scales the uplift rate.
* ``strata``: geology.strata.Strata; ``strata.hardness(z)`` gives rock hardness 0..1 at
  elevations z on the simulation grid (use it to scale erodibility).
* ``cfg``: OFMConfig; parameters in ``cfg.fluvial`` (FluvialConfig), cell size in
  ``cfg.world.cell_size``.
Returns a new array of the same shape; must be deterministic for a given input.

PLACEHOLDER: returns the input unchanged until implemented.
"""
from __future__ import annotations

import numpy as np


def stream_power_erosion(height: np.ndarray, uplift: np.ndarray, strata, cfg) -> np.ndarray:
    return np.asarray(height, dtype=np.float64).copy()
