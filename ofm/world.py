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

REGION_ROWS = "ABCDEFGH"   # rows south -> north; columns are numbered 1.. west -> east


def region_names(divs: int = 4) -> list[str]:
    """All region ids of a divs x divs grid, e.g. A1..D4 for divs = 4."""
    return [f"{REGION_ROWS[r]}{c + 1}" for r in range(divs) for c in range(divs)]


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
    # where the play-test agents start (x, y in metres): fixed at the first evaluation so every
    # candidate of the AI director is played from the same place
    spawn: list[float] | None = None
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
        """The simulation heightfield (``height``, hydrology carved) at world positions."""
        from .grid import bilinear
        ci, cj = self.to_index(x, y)
        return bilinear(self.height, ci, cj)

    def ground_at(self, x, y) -> np.ndarray:
        """The exported terrain surface (the detail heightfield ``hires``) at world positions;
        the simulation heightfield before the detail stage has run.

        Everything placed on the terrain (caves, features, vegetation, POIs) and everything
        that evaluates what a player sees uses this surface, because the detail pass
        (terraces, rills, channels) moves the ground by metres to tens of metres."""
        if self.hires is None:
            return self.height_at(x, y)
        from .grid import bilinear
        return bilinear(self.hires, np.asarray(x) / self.hires_spacing, np.asarray(y) / self.hires_spacing)

    def ground(self) -> np.ndarray:
        """``ground_at`` sampled at the simulation cell centres, shape (size, size)."""
        if self.hires is None:
            return self.height
        c = (np.arange(self.size) + 0.5) * self.cell
        return self.ground_at(c[None, :], c[:, None])

    def visual_height(self) -> np.ndarray:
        """Exported ground including tops of volumetric landmark features (for visibility)."""
        h = self.ground().copy()
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
    @property
    def region_divs(self) -> int:
        return int(self.cfg.evaluation.region_divs)

    def region_of(self, x: float, y: float) -> str:
        n = self.region_divs
        c = int(np.clip(x / self.extent * n, 0, n - 1))
        r = int(np.clip(y / self.extent * n, 0, n - 1))
        return f"{REGION_ROWS[r]}{c + 1}"

    def region_slices(self, region: str) -> tuple[slice, slice]:
        """Array slices (rows, cols) of a region; bounds are spread evenly for any divs."""
        n = self.region_divs
        try:
            r = REGION_ROWS.index(region[0].upper())
            c = int(region[1:]) - 1
        except (ValueError, IndexError):
            raise ValueError(f"bad region {region}") from None
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError(f"bad region {region}")
        edges = np.linspace(0, self.size, n + 1).round().astype(int)
        return slice(edges[r], edges[r + 1]), slice(edges[c], edges[c + 1])

    def all_regions(self) -> list[str]:
        return region_names(self.region_divs)
