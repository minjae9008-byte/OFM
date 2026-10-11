"""Volumetric landmark primitives (signed distance functions).

These are rock formations a heightmap cannot express: free-standing spires
and sea stacks, natural arches, buttes. They are unioned into the density
field and also stamped into a "visual height" map so the perception
simulation can see them from afar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FEATURE_KINDS = ("spire", "arch", "butte", "monolith", "sea_stack")


@dataclass
class Feature:
    id: str
    kind: str
    x: float
    y: float
    z: float              # ground elevation at the base
    height: float
    radius: float
    angle: float = 0.0    # orientation (radians) for arches
    source: str = "procedural"
    name: str = ""
    description: str = ""
    extra: dict = field(default_factory=dict)
    foot: float = 25.0    # the rock reaches this far below ``z`` (see ``footing``)

    # -- geometry -----------------------------------------------------------
    @property
    def top(self) -> float:
        return self.z + self.height

    def _arch_dims(self) -> tuple[float, float]:
        major = self.height * 0.62
        return major, max(self.radius * 0.5, major * 0.3)

    def footprint(self) -> tuple[np.ndarray, np.ndarray]:
        """Ground points under the rock at its base: both legs of an arch, the disc of the
        others."""
        if self.kind == "arch":
            major, minor = self._arch_dims()
            a = (np.array([-1.0, 1.0])[:, None] * major + np.linspace(-minor, minor, 3)[None, :]).ravel()
            b = np.linspace(-1.35 * minor, 1.35 * minor, 3)
        else:
            r = self.radius * (1.1 if self.kind == "monolith" else 1.0)
            ang = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
            a = np.concatenate([[0.0], 0.5 * r * np.cos(ang), r * np.cos(ang)])
            b = np.concatenate([[0.0], 0.5 * r * np.sin(ang), r * np.sin(ang)])
        if self.kind == "arch":
            a, b = (m.ravel() for m in np.meshgrid(a, b))
        c, s = np.cos(self.angle), np.sin(self.angle)
        return self.x + a * c - b * s, self.y + a * s + b * c

    def footing(self, world) -> float:
        """Depth below ``z`` that reaches 2 m under the lowest ground beneath the footprint, so
        the downhill leg or side of a landform on a slope does not float, while the rock stops
        short of caves running deeper underneath."""
        x, y = self.footprint()
        low = float(np.min(world.ground_at(x, y)))
        least = 1.5 * self._arch_dims()[1] if self.kind == "arch" else 2.0
        return max(least, self.z - low + 2.0)

    def stands_dry(self, world) -> bool:
        """True if no part of the footprint is on a lake, the sea or a river channel: the
        rock would wall off the water (sea stacks stand in the sea by definition)."""
        if self.kind == "sea_stack":
            return True
        x, y = self.footprint()
        i = np.clip((np.asarray(x) / world.cell).astype(np.int64), 0, world.size - 1)
        j = np.clip((np.asarray(y) / world.cell).astype(np.int64), 0, world.size - 1)
        wet = world.water | world.river_mask if world.river_mask is not None else world.water
        return not bool(wet[j, i].any())

    def approach(self, world) -> tuple[float, float] | None:
        """A dry point a walker can stand on to visit the landform (its POI): under an
        arch's span, else at the foot just outside the rock on the side most level with
        the base, or for a sea stack the nearest shore. None if there is none nearby."""
        if self.kind == "arch" and not rock_at(world, np.array([self.x]), np.array([self.y]))[0] \
                and _dry(world, np.array([self.x]), np.array([self.y]))[0]:
            return self.x, self.y
        reach = self.height * 0.75 + self.radius if self.kind == "arch" else self.radius * 1.1
        ang = np.arange(16) * (np.pi / 8)
        rings = [reach + 4.0] if self.kind != "sea_stack" else reach + 4.0 + 20.0 * np.arange(10)
        for r in np.atleast_1d(rings):
            x = np.clip(self.x + r * np.cos(ang), 0.0, world.extent)
            y = np.clip(self.y + r * np.sin(ang), 0.0, world.extent)
            ok = _dry(world, x, y) & ~rock_at(world, x, y)
            if ok.any():
                score = np.where(ok, np.abs(world.ground_at(x, y) - self.z), np.inf)
                k = int(np.argmin(score))
                return float(x[k]), float(y[k])
        return None

    def bbox(self, pad: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
        r = self.radius
        if self.kind == "arch":
            r = self.height * 0.75 + self.radius
        lo = np.array([self.x - r - pad, self.y - r - pad, self.z - self.foot - pad])
        hi = np.array([self.x + r + pad, self.y + r + pad, self.top + pad])
        return lo, hi

    def sdf(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Signed distance (negative inside rock)."""
        dx = X - self.x
        dy = Y - self.y
        dz = Z - self.z
        if self.kind in ("spire", "sea_stack", "monolith", "butte"):
            t = np.clip(dz / max(self.height, 1e-3), 0.0, 1.0)
            if self.kind == "spire":
                r = self.radius * (1.0 - 0.62 * t ** 0.9)
            elif self.kind == "sea_stack":
                r = self.radius * (1.0 - 0.3 * t)
            elif self.kind == "butte":
                r = self.radius * (1.0 - 0.12 * t) - 2.5 * np.round(t * 4) / 4
            else:
                r = self.radius * (1.0 - 0.15 * t)
            if self.kind == "monolith":
                c, s = np.cos(self.angle), np.sin(self.angle)
                a = np.abs(dx * c + dy * s) - r * 0.45
                b = np.abs(-dx * s + dy * c) - r * 1.0
                side = np.maximum(a, b)
            else:
                side = np.hypot(dx, dy) - r
            cap = dz - self.height
            return np.maximum(np.maximum(side, cap), -(dz + self.foot))
        if self.kind == "arch":
            c, s = np.cos(self.angle), np.sin(self.angle)
            a = dx * c + dy * s            # along the span
            b = -dx * s + dy * c           # across (thickness)
            major, minor = self._arch_dims()
            # a half ring above the base; below it the legs go straight down to the footing
            q1 = np.hypot(a, np.maximum(dz, 0.0)) - major
            ring = np.sqrt(q1 * q1 + (b / 1.35) ** 2) - minor
            return np.maximum(ring, -(dz + self.foot))
        raise ValueError(f"unknown feature kind {self.kind}")

    def stamp_top(self, h: np.ndarray, cell: float) -> None:
        """Raise a heightfield to the top of this feature (visual height map)."""
        size = h.shape[0]
        lo, hi = self.bbox(pad=0.0)
        i0, i1 = int(max(0, lo[0] / cell - 1)), int(min(size, hi[0] / cell + 1))
        j0, j1 = int(max(0, lo[1] / cell - 1)), int(min(size, hi[1] / cell + 1))
        if i0 >= i1 or j0 >= j1:
            return
        jj, ii = np.mgrid[j0:j1, i0:i1]
        X = (ii + 0.5) * cell
        Y = (jj + 0.5) * cell
        zs = np.linspace(self.z, self.top, 24)
        best = np.full(X.shape, -np.inf)
        for z in zs:
            inside = self.sdf(X, Y, np.full(X.shape, z)) < 0
            best = np.where(inside, z, best)
        sub = h[j0:j1, i0:i1]
        h[j0:j1, i0:i1] = np.maximum(sub, best)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "x": round(self.x, 2), "y": round(self.y, 2),
            "z": round(self.z, 2), "height": round(self.height, 2), "radius": round(self.radius, 2),
            "angle": round(self.angle, 4), "source": self.source, "name": self.name,
            "description": self.description,
        }


def _dry(world, x, y) -> np.ndarray:
    i = np.clip((np.asarray(x) / world.cell).astype(np.int64), 0, world.size - 1)
    j = np.clip((np.asarray(y) / world.cell).astype(np.int64), 0, world.size - 1)
    wet = world.water | world.river_mask if world.river_mask is not None else world.water
    return ~wet[j, i] & (world.ground_at(x, y) > world.cfg.world.sea_level + 0.3)


def rock_at(world, x, y, clearance: float = 1.5) -> np.ndarray:
    """True where a rock landform fills the air ``clearance`` metres above the exported
    ground, so nobody can stand or walk there (an arch's opening stays free). The ground
    under a landform is the terrain it stands on; only this test sees the rock."""
    shape = np.shape(x)
    x = np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()
    y = np.atleast_1d(np.asarray(y, dtype=np.float64)).ravel()
    out = np.zeros(x.shape, dtype=bool)
    if world.features:
        z = np.asarray(world.ground_at(x, y), dtype=np.float64) + clearance
        for f in world.features:
            lo, hi = f.bbox(pad=0.0)
            m = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
            if m.any():
                out[m] |= f.sdf(x[m], y[m], z[m]) < 0
    return out.reshape(shape)


def make_feature(world, kind: str, x: float, y: float, rng: np.random.Generator,
                 scale: float = 1.0, source: str = "procedural") -> Feature:
    if kind not in FEATURE_KINDS:
        raise ValueError(f"unknown feature kind {kind}")
    z = float(world.ground_at(x, y))
    if kind == "sea_stack":
        z = min(z, world.cfg.world.sea_level - 2.0)
    dims = {
        "spire": (rng.uniform(45, 80), rng.uniform(9, 14)),
        "arch": (rng.uniform(28, 46), rng.uniform(6, 9)),
        "butte": (rng.uniform(30, 55), rng.uniform(40, 70)),
        "monolith": (rng.uniform(25, 40), rng.uniform(7, 11)),
        "sea_stack": (rng.uniform(30, 55), rng.uniform(10, 16)),
    }[kind]
    height, radius = dims[0] * scale, dims[1] * scale
    if kind == "sea_stack":
        height += world.cfg.world.sea_level - z
    f = Feature(
        id=f"feature_{len(world.features)}", kind=kind, x=float(x), y=float(y), z=z,
        height=float(height), radius=float(radius), angle=float(rng.uniform(0, np.pi)), source=source,
    )
    f.foot = f.footing(world)
    return f
