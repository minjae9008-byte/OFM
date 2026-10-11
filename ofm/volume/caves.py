"""Karst cave formation as a dissolution-agent simulation.

Caves form where slightly acidic water dissolves soluble rock (limestone,
dolomite). Water enters at sinkholes on exposed karst, descends steeply
through the unsaturated (vadose) zone, then travels near the water table
(phreatic zone) towards a spring on a valley side, which becomes the second
entrance. Agents follow that path while preferring soluble strata and
pre-existing fractures (3D noise), branch occasionally and dissolve
chambers where the rock is most soluble. Sea caves are added where waves
attack coastal cliffs.

The output is a set of capsule segments and ellipsoid chambers carved into
the volumetric density field later on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import OFMConfig
from ..grid import bilinear, blur, chamfer_distance, slope
from ..noise import Noise, derive_seed


@dataclass
class CaveSystem:
    id: str
    kind: str                                  # karst | sea | fissure
    segments: np.ndarray                       # (n, 8) ax ay az bx by bz ra rb
    chambers: np.ndarray                       # (m, 6) cx cy cz rx ry rz
    entrances: list[dict] = field(default_factory=list)
    name: str = ""
    description: str = ""
    main_count: int = 0                        # segments [0, main_count) form the trunk passage

    @property
    def length(self) -> float:
        if len(self.segments) == 0:
            return 0.0
        d = self.segments[:, 3:6] - self.segments[:, 0:3]
        return float(np.linalg.norm(d, axis=1).sum())

    def bbox(self, pad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        parts_lo, parts_hi = [], []
        if len(self.segments):
            r = np.maximum(self.segments[:, 6], self.segments[:, 7])[:, None]
            parts_lo.append(np.minimum(self.segments[:, 0:3], self.segments[:, 3:6]) - r)
            parts_hi.append(np.maximum(self.segments[:, 0:3], self.segments[:, 3:6]) + r)
        if len(self.chambers):
            parts_lo.append(self.chambers[:, 0:3] - self.chambers[:, 3:6])
            parts_hi.append(self.chambers[:, 0:3] + self.chambers[:, 3:6])
        lo = np.min(np.vstack(parts_lo), axis=0) - pad
        hi = np.max(np.vstack(parts_hi), axis=0) + pad
        return lo, hi

    def to_dict(self) -> dict:
        lo, hi = self.bbox()
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "length_m": round(self.length, 1),
            "depth_range": [round(float(lo[2]), 1), round(float(hi[2]), 1)],
            "entrances": self.entrances,
            "chambers": [[round(float(v), 2) for v in c] for c in self.chambers],
            "segments": [[round(float(v), 2) for v in s] for s in self.segments],
        }


class _Terrain:
    """Small sampling helper around the world arrays."""

    def __init__(self, world):
        self.w = world
        self.cell = world.cell
        self.extent = world.extent
        self.noise = Noise(derive_seed(world.cfg.world.seed, "fractures"))

    def h(self, x, y):
        # the exported ground: mouths open at the visible surface and the passage cover is
        # measured under the detail terrain and the carved river channels
        return self.w.ground_at(x, y)

    def wt(self, x, y):
        return bilinear(self.w.water_table, np.asarray(x) / self.cell - 0.5, np.asarray(y) / self.cell - 0.5)

    def solubility(self, x, y, z):
        fold = self.w.strata.fold_at(x, y)
        return self.w.strata.solubility(z, fold)

    def fracture(self, x, y, z):
        # Joint sets: anisotropic noise, stretched horizontally like bedding-parallel joints.
        return self.noise.perlin3(np.asarray(x) / 70.0, np.asarray(y) / 70.0, np.asarray(z) / 25.0)

    def inside(self, x, y, margin=8.0):
        return (margin <= x <= self.extent - margin) and (margin <= y <= self.extent - margin)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _fit_chamber(t: _Terrain, cc, ch) -> list | None:
    """Shrink a chamber until its roof stays under the ground across its whole footprint;
    None if even a small one would break out. Cover is only known at its centre, while its
    flanks reach under lower ground (a valley side, a lake bed)."""
    cx, cy, cz, rx, ry, rz = (float(v) for v in ch)
    ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    R, A = np.meshgrid(np.array([0.0, 0.5, 0.8, 0.95]), ang)
    margin = 1.0 + 1.3 * cc.wall_noise              # the wall noise widens the carved shape
    for _ in range(5):
        top = cz + rz * np.sqrt(1.0 - R * R)
        if (t.h(cx + rx * R * np.cos(A), cy + ry * R * np.sin(A)) - top).min() >= margin:
            return [cx, cy, cz, rx, ry, rz]
        rx, ry, rz = rx * 0.75, ry * 0.75, rz * 0.75
    return None


def _walk(t: _Terrain, cc, rng, start, target, max_len, vadose_first=True, allow_exit=True):
    """Run one dissolution agent. Returns (segments, chambers, reached_target, branch_requests)."""
    pos = np.array(start, dtype=np.float64)
    segs, chambers, branches = [], [], []
    to = np.asarray(target, dtype=np.float64) - pos
    direction = _unit(np.array([to[0], to[1], -0.8 * np.hypot(to[0], to[1])]))
    phase = "vadose" if vadose_first else "phreatic"
    travelled = 0.0
    step = cc.step
    reached = False
    n_steps = int(max_len / step) + 1
    for k in range(n_steps):
        to = np.asarray(target, dtype=np.float64) - pos
        dist_h = float(np.hypot(to[0], to[1]))
        if allow_exit and dist_h < step * 1.5 and abs(to[2]) < step * 3:
            r = cc.tunnel_radius
            segs.append([*pos, *target, r, r * 1.15])
            reached = True
            break
        horiz = to[:2] / max(dist_h, 1e-6)
        wt_here = float(t.wt(pos[0], pos[1]))
        if phase == "vadose" and pos[2] <= wt_here + 4.0:
            phase = "phreatic"
        if dist_h < 90.0 and allow_exit:
            desired = _unit(to)
        elif phase == "vadose":
            # descend at least as steeply as the ground ahead falls, so the shaft stays covered
            ahead = pos[:2] + horiz * step
            ground_drop = (float(t.h(ahead[0], ahead[1])) - float(t.h(pos[0], pos[1]))) / step
            desired = _unit(np.array([horiz[0], horiz[1], min(-0.85, ground_drop - 0.5)]))
        else:
            z_goal = max(float(target[2]), wt_here) if allow_exit else wt_here
            # never aim shallower than the minimum rock cover under the surface ahead
            ahead = pos[:2] + horiz * step * 3
            z_goal = min(z_goal, float(t.h(ahead[0], ahead[1])) - cc.min_cover * 1.3)
            dz = np.clip((z_goal - pos[2]) / max(min(dist_h, step * 6), step), -0.6, 0.6)
            desired = _unit(np.array([horiz[0], horiz[1], dz]))

        cand = _unit(desired[None, :] + 0.6 * direction[None, :] + 0.55 * _unit(rng.normal(size=(10, 3))))
        down = np.array([0.0, 0.0, -1.0])
        cand = np.vstack([cand, _unit(desired + down)[None], _unit(desired + 2.5 * down)[None]])
        nxt = pos[None, :] + cand * step
        surf = t.h(nxt[:, 0], nxt[:, 1])
        cover = surf - nxt[:, 2]
        sol = t.solubility(nxt[:, 0], nxt[:, 1], nxt[:, 2])
        frac = t.fracture(nxt[:, 0], nxt[:, 1], nxt[:, 2])
        near_ends = k < 3 or (allow_exit and dist_h < 60.0)
        penalty = 0.0 if near_ends else 3.0 * np.clip((cc.min_cover - cover) / cc.min_cover, 0.0, None)
        if not near_ends:
            penalty = penalty + np.where(cover < cc.min_cover * 0.5, 50.0, 0.0)   # do not break out mid-way
        score = 1.3 * (cand @ desired) + 0.9 * sol + 0.35 * frac - penalty
        inside = np.array([t.inside(x, y) for x, y in nxt[:, :2]])
        score = np.where(inside, score, -1e9)
        best = int(np.argmax(score))
        if score[best] < -1e8:
            break
        new = nxt[best]
        base_r = cc.shaft_radius if phase == "vadose" else cc.tunnel_radius
        r0 = base_r * (0.75 + 0.5 * sol[best]) * rng.uniform(0.85, 1.15)
        segs.append([*pos, *new, r0, r0 * rng.uniform(0.9, 1.1)])
        direction = cand[best]
        pos = new
        travelled += step

        if rng.random() < cc.chamber_chance * (0.4 + sol[best]) and cover[best] > cc.min_cover + 2 * r0:
            room = cover[best] - cc.min_cover * 0.7
            rz = float(np.clip(r0 * rng.uniform(1.5, 2.6), r0, max(r0, room * 0.5)))
            ch = _fit_chamber(t, cc, [*pos, r0 * rng.uniform(2.2, 4.2), r0 * rng.uniform(2.2, 4.2), rz])
            if ch is not None:
                chambers.append(ch)
        if rng.random() < cc.branch_chance:
            ang = rng.choice([-1, 1]) * rng.uniform(np.radians(40), np.radians(95))
            c, s = np.cos(ang), np.sin(ang)
            d2 = np.array([direction[0] * c - direction[1] * s, direction[0] * s + direction[1] * c])
            branches.append((pos.copy(), d2, rng.uniform(80.0, 320.0), r0 * 0.75))
        if travelled > max_len:
            break
    return segs, chambers, reached, branches


def _branch_walk(t: _Terrain, cc, rng, start, d2, length, radius):
    pos = np.array(start, dtype=np.float64)
    direction = _unit(np.array([d2[0], d2[1], -0.1]))
    segs = []
    for _ in range(int(length / cc.step)):
        cand = _unit(direction[None, :] + 0.6 * _unit(rng.normal(size=(8, 3))))
        cand = np.vstack([cand, _unit(direction + np.array([0.0, 0.0, -1.5]))[None]])
        nxt = pos[None, :] + cand * cc.step
        cover = t.h(nxt[:, 0], nxt[:, 1]) - nxt[:, 2]
        sol = t.solubility(nxt[:, 0], nxt[:, 1], nxt[:, 2])
        penalty = 3.0 * np.clip((cc.min_cover - cover) / cc.min_cover, 0.0, None)
        score = 1.2 * (cand @ direction) + sol + 0.3 * t.fracture(nxt[:, 0], nxt[:, 1], nxt[:, 2]) - penalty
        inside = np.array([t.inside(x, y) for x, y in nxt[:, :2]])
        score = np.where(inside, score, -1e9)
        best = int(np.argmax(score))
        if score[best] < -1e8:
            break
        segs.append([*pos, *nxt[best], radius, radius * 0.95])
        pos = nxt[best]
        direction = cand[best]
        radius *= 0.985
    chamber = None
    if segs and rng.random() < 0.5:
        r = radius * rng.uniform(2.0, 3.2)
        chamber = _fit_chamber(t, cc, [*pos, r, r * rng.uniform(0.7, 1.2), r * 0.6])
    return segs, chamber


def simulate_caves(world) -> list[CaveSystem]:
    cfg: OFMConfig = world.cfg
    cc = cfg.caves
    rng = np.random.default_rng(derive_seed(cfg.world.seed, "caves"))
    t = _Terrain(world)
    h = world.height
    cell = world.cell
    water = world.water
    land = ~water
    s = slope(h, cell)
    surf_sol = world.strata.solubility(h)
    sub_sol = world.strata.solubility(h - 25.0)
    above_wt = h - world.water_table

    karst = land & (np.maximum(surf_sol, sub_sol) >= 0.5)
    karst_km2 = karst.sum() * cell * cell / 1e6
    n_sys = int(np.clip(round(cc.density * karst_km2 * 0.8), cc.min_systems, cc.max_systems))
    kind = "karst"
    sink_ok = karst & (s < np.tan(np.radians(26))) & (above_wt > cc.min_cover * 1.5)
    if sink_ok.sum() < n_sys * 4:
        # Too little exposed karst: fall back to fracture caves in any hilly terrain.
        sink_ok = land & (s < np.tan(np.radians(30))) & (above_wt > cc.min_cover * 1.5)
        kind = "fissure"
    outlet_ok = land & (s > np.tan(np.radians(24))) & ~world.river_mask
    if not sink_ok.any() or not outlet_ok.any():
        return _sea_caves(world, t, rng, [])

    sink_idx = np.flatnonzero(sink_ok.reshape(-1))
    weights = (world.precipitation.reshape(-1)[sink_idx] / 1000.0) * (0.3 + sub_sol.reshape(-1)[sink_idx])
    order = rng.choice(sink_idx, size=min(sink_idx.size, n_sys * 12), replace=False, p=weights / weights.sum())
    out_j, out_i = np.nonzero(outlet_ok)
    out_x, out_y = (out_i + 0.5) * cell, (out_j + 0.5) * cell
    out_h = h[out_j, out_i]
    # Karst springs rise where the soluble rock meets the valley side (often at its contact
    # with impermeable rock below), so outlets in soluble rock are strongly preferred.
    out_sol = np.maximum(surf_sol[out_j, out_i], world.strata.solubility(h - 8.0)[out_j, out_i])

    systems: list[CaveSystem] = []
    used: list[tuple[float, float]] = []
    min_sep = max(350.0, world.extent / (n_sys ** 0.5 + 1) * 0.6)
    for c in order.tolist():
        if len(systems) >= n_sys:
            break
        j, i = divmod(c, world.size)
        sx, sy = (i + 0.5) * cell, (j + 0.5) * cell
        if any(np.hypot(sx - ux, sy - uy) < min_sep for ux, uy in used):
            continue
        sz = float(t.h(sx, sy))
        d = np.hypot(out_x - sx, out_y - sy)
        drop = sz - out_h
        ok = (d > 250.0) & (d < cc.search_radius) & (drop > 25.0)
        if not ok.any():
            continue
        cost = np.where(ok, d / 600.0 - drop / 120.0 + 1.5 * (1.0 - out_sol) + rng.uniform(0, 0.6, d.shape),
                        np.inf)
        k = int(np.argmin(cost))
        tx, ty = float(out_x[k]), float(out_y[k])
        tz = float(t.h(tx, ty)) - cc.tunnel_radius * 0.25
        start = (sx, sy, sz + 1.5)
        segs, chambers, reached, branches = _walk(t, cc, rng, start, (tx, ty, tz), cc.max_length)
        if len(segs) < 6:
            continue
        main_count = len(segs)
        for b_pos, b_dir, b_len, b_r in branches[:4]:
            bsegs, bch = _branch_walk(t, cc, rng, b_pos, b_dir, b_len, b_r)
            segs.extend(bsegs)
            if bch:
                chambers.append(bch)
        entrances = [{
            "x": round(sx, 2), "y": round(sy, 2), "z": round(sz, 2), "kind": "sinkhole",
            "dir": [round(float(v), 3) for v in _unit(np.array(segs[0][3:6]) - np.array(segs[0][0:3]))],
        }]
        if reached:
            entrances.append({"x": round(tx, 2), "y": round(ty, 2), "z": round(float(t.h(tx, ty)), 2),
                              "kind": "spring", "dir": [0.0, 0.0, 0.0]})
        else:
            last = segs[-1]
            r = max(last[6], last[7]) * rng.uniform(2.0, 3.0)
            ch = _fit_chamber(t, cc, [last[3], last[4], last[5], r, r, r * 0.6])
            if ch is not None:
                chambers.append(ch)
        systems.append(CaveSystem(
            id=f"cave_{len(systems)}", kind=kind,
            segments=np.array(segs, dtype=np.float64),
            chambers=np.array(chambers, dtype=np.float64).reshape(-1, 6),
            entrances=entrances, main_count=main_count,
        ))
        used.append((sx, sy))
    return _sea_caves(world, t, rng, systems)


def _sea_caves(world, t: _Terrain, rng, systems: list[CaveSystem]) -> list[CaveSystem]:
    """Short wave-cut tunnels into steep coastal cliffs."""
    cfg = world.cfg
    sea = cfg.world.sea_level
    if world.coast_cliff is None or not world.coast_cliff.any():
        return systems
    cell = world.cell
    cand_j, cand_i = np.nonzero(world.coast_cliff)
    order = rng.permutation(len(cand_i))
    d_ocean = chamfer_distance(world.ocean, max_dist=6)
    inland_dir_y, inland_dir_x = np.gradient(blur(d_ocean, 1.5))
    made = []
    for k in order[:200]:
        if len(made) >= max(1, cfg.caves.max_systems // 5):
            break
        j, i = int(cand_j[k]), int(cand_i[k])
        x, y = (i + 0.5) * cell, (j + 0.5) * cell
        if any(np.hypot(x - mx, y - my) < 500 for mx, my in made):
            continue
        d = np.array([inland_dir_x[j, i], inland_dir_y[j, i]])
        if np.linalg.norm(d) < 1e-6:
            continue
        d = d / np.linalg.norm(d)
        z = sea + 2.5
        r = cfg.caves.tunnel_radius * rng.uniform(1.0, 1.5)
        start = np.array([x - d[0] * cell, y - d[1] * cell, z])
        if not t.inside(start[0], start[1]):
            continue
        segs = []
        pos = start.copy()
        length = rng.uniform(50.0, 130.0)
        for _ in range(int(length / cfg.caves.step)):
            ang = rng.normal(0, 0.25)
            c, s = np.cos(ang), np.sin(ang)
            d = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])
            nxt = pos + np.array([d[0], d[1], rng.normal(0, 0.15)]) * cfg.caves.step
            nxt[2] = np.clip(nxt[2], sea + 1.5, sea + 5.0)
            if not t.inside(nxt[0], nxt[1]):
                break
            if float(t.h(nxt[0], nxt[1])) - nxt[2] < 6.0 and len(segs) > 2:
                break
            segs.append([*pos, *nxt, r, r * 0.95])
            pos = nxt
            r *= 0.97
        if len(segs) < 4:
            continue
        chambers = np.array([[pos[0], pos[1], pos[2] + r * 0.5, r * 2.2, r * 2.0, r * 1.4]])
        systems.append(CaveSystem(
            id=f"cave_{len(systems)}", kind="sea", segments=np.array(segs), chambers=chambers,
            main_count=len(segs),
            entrances=[{"x": round(float(start[0]), 2), "y": round(float(start[1]), 2), "z": round(z, 2),
                        "kind": "sea_cave", "dir": [round(float(d[0]), 3), round(float(d[1]), 3), 0.0]}],
        ))
        made.append((x, y))
    return systems
