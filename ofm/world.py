"""The World container passed between pipeline stages.

Coordinate convention (metres, right-handed, Z up):
  * x grows east (array column), y grows north (array row), z is elevation.
  * Simulation cell (row j, col i) is centred at ((i + 0.5) * cell, (j + 0.5) * cell).
  * The world covers [0, size * cell] on both horizontal axes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import OFMConfig

REGION_DIVS = 4
REGION_ROWS = "ABCD"


@dataclass
class Landmark:
    id: str
    kind: str
    x: float
    y: float
    z: float
    height: float
    salience: float = 0.0
    visibility: float = 0.0
    name: str = ""
    description: str = ""


@dataclass
class POI:
    id: str
    kind: str
    x: float
    y: float
    z: float
    importance: float
    region: str = ""
    source: str = "procedural"     # procedural | psych | ai
    name: str = ""
    description: str = ""
    props: list[str] = field(default_factory=list)


@dataclass
class World:
    cfg: OFMConfig
    size: int
    cell: float
    # terrain
    height: np.ndarray | None = None          # current heightfield (m)
    height_base: np.ndarray | None = None     # post-erosion, pre-hydrology (director edits this)
    uplift: np.ndarray | None = None
    plate_id: np.ndarray | None = None
    strata: Any = None
    hardness: np.ndarray | None = None
    # water
    ocean: np.ndarray | None = None
    lake_mask: np.ndarray | None = None
    lake_level: np.ndarray | None = None
    lakes: list[dict] = field(default_factory=list)
    playa: np.ndarray | None = None
    discharge: np.ndarray | None = None
    receiver: np.ndarray | None = None
    river_mask: np.ndarray | None = None
    rivers: list[dict] = field(default_factory=list)
    waterfalls: list[dict] = field(default_factory=list)
    water_table: np.ndarray | None = None
    # climate & surface
    temperature: np.ndarray | None = None
    precipitation: np.ndarray | None = None
    biome: np.ndarray | None = None
    splat: np.ndarray | None = None
    # 3D features
    caves: list[Any] = field(default_factory=list)
    forced_caves: list[Any] = field(default_factory=list)  # caves added by the AI director
    features: list[Any] = field(default_factory=list)   # volumetric landmark primitives
    cliff_mask: np.ndarray | None = None
    coast_cliff: np.ndarray | None = None
    # detail
    hires: np.ndarray | None = None
    hires_spacing: float = 0.0
    scatter: dict[str, np.ndarray] = field(default_factory=dict)
    pois: list[POI] = field(default_factory=list)
    landmarks: list[Landmark] = field(default_factory=list)
    # psychology
    psych: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    lore: dict[str, Any] = field(default_factory=dict)
    edits: list[dict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    # -- geometry helpers ---------------------------------------------------
    @property
    def extent(self) -> float:
        return self.size * self.cell

    def cell_center(self, i, j):
        return (np.asarray(i) + 0.5) * self.cell, (np.asarray(j) + 0.5) * self.cell

    def to_index(self, x, y):
        """World metres -> fractional cell index (col, row)."""
        return np.asarray(x) / self.cell - 0.5, np.asarray(y) / self.cell - 0.5

    def height_at(self, x, y) -> np.ndarray:
        from .grid import bilinear
        ci, cj = self.to_index(x, y)
        return bilinear(self.height, ci, cj)

    def visual_height(self) -> np.ndarray:
        """Heightfield including tops of volumetric landmark features (for visibility)."""
        h = self.height.copy()
        for feat in self.features:
            feat.stamp_top(h, self.cell)
        if self.lake_level is not None:
            h = np.where(self.lake_mask, np.maximum(h, self.lake_level), h)
        return np.maximum(h, self.cfg.world.sea_level)

    @property
    def water(self) -> np.ndarray:
        w = self.ocean.copy()
        if self.lake_mask is not None:
            w |= self.lake_mask
        return w

    # -- regions ------------------------------------------------------------
    def region_of(self, x: float, y: float) -> str:
        n = REGION_DIVS
        c = int(np.clip(x / self.extent * n, 0, n - 1))
        r = int(np.clip(y / self.extent * n, 0, n - 1))
        return f"{REGION_ROWS[r]}{c + 1}"

    def region_slices(self, region: str) -> tuple[slice, slice]:
        r = REGION_ROWS.index(region[0].upper())
        c = int(region[1:]) - 1
        if not (0 <= r < REGION_DIVS and 0 <= c < REGION_DIVS):
            raise ValueError(f"bad region {region}")
        step = self.size // REGION_DIVS
        return slice(r * step, (r + 1) * step), slice(c * step, (c + 1) * step)

    @staticmethod
    def all_regions() -> list[str]:
        return [f"{REGION_ROWS[r]}{c + 1}" for r in range(REGION_DIVS) for c in range(REGION_DIVS)]
