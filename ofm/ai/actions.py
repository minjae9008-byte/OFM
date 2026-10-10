"""Whitelisted world edits the evaluator (LLM or heuristic) may request.

The model never writes code or raw geometry. It proposes actions from this
fixed vocabulary; each is validated and clamped here, then the site is
chosen by the simulation (e.g. a watchtower goes to the spot with the best
prospect in the region, a camp to the best prospect-refuge spot).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import get_by_path, set_by_path
from ..grid import blur, chamfer_distance, slope, smoothstep
from ..noise import Noise, derive_seed
from ..volume.caves import CaveSystem
from ..volume.features import FEATURE_KINDS, make_feature
from ..world import POI, World

LANDMARK_KINDS = FEATURE_KINDS + ("peak",)
POI_KINDS = ("ruin", "camp", "shrine", "watchtower", "outpost", "treasure", "village", "bridge", "hermitage")

PARAM_LIMITS: dict[str, tuple[float, float]] = {
    "caves.density": (0.2, 3.0),
    "caves.branch_chance": (0.0, 0.1),
    "caves.chamber_chance": (0.0, 0.1),
    "cliffs.undercut_depth": (0.0, 14.0),
    "cliffs.terrace_strength": (0.0, 1.0),
    "hydrology.river_min_discharge": (0.3, 6.0),
    "detail.poi_count": (8, 60),
    "detail.poi_spacing": (120.0, 600.0),
    "climate.orographic": (0.1, 1.5),
}

ACTION_DOC = {
    "add_landmark": "region, kind in [" + ", ".join(LANDMARK_KINDS) + "]",
    "add_cave": "region",
    "add_lake": "region",
    "add_poi": "region, kind in [" + ", ".join(POI_KINDS) + "]",
    "raise_relief": "region, amount 0.1-0.6 (exaggerate local hills/valleys)",
    "smooth_terrain": "region, amount 0.1-0.8 (calm rugged ground)",
    "set_param": "path in [" + ", ".join(PARAM_LIMITS) + "], value (number)",
}


class ActionError(ValueError):
    pass


def validate_action(raw: Any, regions: list[str]) -> dict:
    if not isinstance(raw, dict):
        raise ActionError("action must be an object")
    kind = str(raw.get("type", raw.get("action", ""))).strip()
    if kind not in ACTION_DOC:
        raise ActionError(f"unknown action '{kind}'")
    out: dict[str, Any] = {"type": kind, "reason": str(raw.get("reason", ""))[:240]}
    if kind != "set_param":
        region = str(raw.get("region", "")).strip().upper()
        if region not in regions:
            raise ActionError(f"bad region '{region}'")
        out["region"] = region
    if kind == "add_landmark":
        k = str(raw.get("kind", "spire")).lower()
        out["kind"] = k if k in LANDMARK_KINDS else "spire"
    elif kind == "add_poi":
        k = str(raw.get("kind", "ruin")).lower()
        out["kind"] = k if k in POI_KINDS else "ruin"
    elif kind in ("raise_relief", "smooth_terrain"):
        lo, hi = (0.1, 0.6) if kind == "raise_relief" else (0.1, 0.8)
        try:
            amount = float(raw.get("amount", 0.3))
        except (TypeError, ValueError):
            amount = 0.3
        out["amount"] = float(np.clip(amount, lo, hi))
    elif kind == "set_param":
        path = str(raw.get("path", ""))
        if path not in PARAM_LIMITS:
            raise ActionError(f"parameter '{path}' is not adjustable")
        lo, hi = PARAM_LIMITS[path]
        try:
            value = float(raw.get("value"))
        except (TypeError, ValueError) as exc:
            raise ActionError("set_param needs a numeric value") from exc
        out["path"] = path
        out["value"] = float(np.clip(value, lo, hi))
    return out


def action_key(a: dict) -> str:
    return "|".join(str(a.get(k, "")) for k in ("type", "region", "kind", "path"))


# --------------------------------------------------------------------------------
def _region_mask(world: World, region: str, feather: float = 6.0) -> np.ndarray:
    sj, si = world.region_slices(region)
    m = np.zeros((world.size, world.size))
    m[sj, si] = 1.0
    return np.clip(blur(m, feather), 0, 1)


def _site(world: World, region: str, score: np.ndarray, avoid: float = 220.0) -> tuple[float, float] | None:
    sj, si = world.region_slices(region)
    s = np.full(score.shape, -np.inf)
    s[sj, si] = score[sj, si]
    cell = world.cell
    yy, xx = np.mgrid[0:world.size, 0:world.size]
    px, py = (xx + 0.5) * cell, (yy + 0.5) * cell
    for q in list(world.pois) + list(world.features):
        s = np.where(np.hypot(px - q.x, py - q.y) < avoid, -np.inf, s)
    m = 4
    s[:m, :] = s[-m:, :] = -np.inf
    s[:, :m] = s[:, -m:] = -np.inf
    if not np.isfinite(s).any():
        return None
    j, i = np.unravel_index(int(np.argmax(s)), s.shape)
    return float((i + 0.5) * cell), float((j + 0.5) * cell)


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _psych_field(world: World, key: str) -> np.ndarray:
    from ..grid import resample
    per = world.psych.get("perception")
    if per is None:
        return np.zeros((world.size, world.size))
    grid = getattr(per, key) if hasattr(per, key) else world.psych[key]
    return resample(np.asarray(grid, dtype=float), (world.size, world.size))


def apply_action(world: World, action: dict, rng: np.random.Generator) -> dict:
    """Apply one validated action. Returns a record for the edit log."""
    kind = action["type"]
    land = ~world.water
    h = world.height
    s = slope(h, world.cell)
    rec = dict(action)
    if kind == "set_param":
        old = get_by_path(world.cfg, action["path"])
        set_by_path(world.cfg, action["path"], action["value"])
        rec.update(old=old)
        return rec

    region = action["region"]
    if kind == "add_landmark":
        lk = action["kind"]
        if lk == "sea_stack":
            near = chamfer_distance(world.coast_cliff | (land & (chamfer_distance(world.ocean, 3) <= 1)), 6) <= 3
            depth = world.cfg.world.sea_level - h
            score = np.where(world.ocean & near & (depth > 1) & (depth < 12), 1.0 - depth / 12.0, -np.inf)
        elif lk == "peak":
            score = np.where(land, _norm(h) + 0.3 * rng.random(h.shape), -np.inf)
        else:
            prospect = _psych_field(world, "prospect")
            score = np.where(land & (s < 0.5), 0.6 * _norm(h) + 0.4 * prospect + 0.1 * rng.random(h.shape), -np.inf)
        site = _site(world, region, score)
        if site is None:
            raise ActionError(f"no site for {lk} in {region}")
        if lk == "peak":
            _raise_peak(world, site, rng)
            rec.update(x=site[0], y=site[1])
        else:
            f = make_feature(world, lk, site[0], site[1], rng, scale=1.15, source="ai")
            world.features.append(f)
            rec.update(x=site[0], y=site[1], feature=f.id)
    elif kind == "add_cave":
        score = np.where(land & (s > 0.3) & (s < 0.9), _norm(h) * 0.5 + rng.random(h.shape) * 0.5, -np.inf)
        site = _site(world, region, score, avoid=160.0)
        if site is None:
            raise ActionError(f"no hillside for a cave in {region}")
        cave = _grotto(world, site, rng)
        world.forced_caves.append(cave)
        rec.update(x=site[0], y=site[1], cave=cave.id)
    elif kind == "add_lake":
        score = np.where(land & ~world.river_mask, -blur(s, 3) + 0.2 * rng.random(h.shape), -np.inf)
        site = _site(world, region, score)
        if site is None:
            raise ActionError(f"no basin site in {region}")
        _dig_basin(world, site, rng)
        rec.update(x=site[0], y=site[1])
    elif kind == "add_poi":
        pk = action["kind"]
        site = _site(world, region, _poi_score(world, pk, land, s, rng))
        if site is None:
            raise ActionError(f"no site for {pk} in {region}")
        z = float(world.height_at(*site))
        world.pois.append(POI(id=f"poi_ai_{len(world.pois)}", kind=pk, x=site[0], y=site[1], z=z,
                              importance=0.8, region=region, source="ai"))
        rec.update(x=site[0], y=site[1])
    elif kind == "raise_relief":
        m = _region_mask(world, region)
        hb = world.height_base
        mean = (hb * m).sum() / max(m.sum(), 1)
        world.height_base = hb + m * (hb - mean) * action["amount"] * np.where(hb > world.cfg.world.sea_level, 1.0, 0.0)
    elif kind == "smooth_terrain":
        m = _region_mask(world, region) * action["amount"]
        hb = world.height_base
        world.height_base = hb + m * (blur(hb, 3.0) - hb)
    world.edits.append(rec)
    return rec


def _poi_score(world, pk, land, s, rng):
    prospect = _psych_field(world, "prospect")
    refuge = _psych_field(world, "refuge")
    mystery = _psych_field(world, "mystery")
    boredom = _norm(_psych_field(world, "boredom_grid")) if "boredom_grid" in world.psych else 0.0
    water_d = chamfer_distance(world.river_mask | world.lake_mask, max_dist=20)
    flat = 1.0 - smoothstep(0.03, 0.25, s)
    noise = 0.05 * rng.random(s.shape)
    base = {
        "watchtower": prospect + 0.3 * _norm(world.height),
        "camp": np.sqrt(np.clip(prospect, 0, 1) * np.clip(refuge, 0, 1)) + 0.2 * flat,
        "ruin": 0.5 * prospect + 0.3 * mystery + 0.3 * flat,
        "shrine": mystery + 0.3 * refuge,
        "outpost": 0.6 * prospect + 0.4 * flat,
        "treasure": refuge + mystery * 0.5 - prospect * 0.3,
        "village": flat * 1.2 + np.exp(-water_d / 4.0),
        "bridge": np.where(world.river_mask, 1.0, -np.inf),
        "hermitage": refuge + 0.4 * _norm(world.height),
    }[pk]
    walkable = land & (s < 0.7)
    if pk == "bridge":
        walkable = world.river_mask
    return np.where(walkable, base + 0.6 * boredom + noise, -np.inf)


def _raise_peak(world: World, site, rng) -> None:
    cell = world.cell
    yy, xx = np.mgrid[0:world.size, 0:world.size]
    px, py = (xx + 0.5) * cell, (yy + 0.5) * cell
    radius = rng.uniform(260.0, 420.0)
    height = rng.uniform(110.0, 190.0)
    noise = Noise(derive_seed(world.cfg.world.seed, "peak", len(world.edits)))
    r = np.hypot(px - site[0], py - site[1]) / radius
    ridge = noise.ridged2(px / 300.0, py / 300.0, 4)
    bump = np.exp(-r * r * 2.2) * height * (0.65 + 0.5 * ridge)
    world.height_base = world.height_base + bump


def _dig_basin(world: World, site, rng) -> None:
    cell = world.cell
    yy, xx = np.mgrid[0:world.size, 0:world.size]
    px, py = (xx + 0.5) * cell, (yy + 0.5) * cell
    radius = rng.uniform(180.0, 300.0)
    r = np.hypot(px - site[0], py - site[1]) / radius
    hb = world.height_base
    rim = (r > 1.0) & (r < 1.15)
    rim_min = float(hb[rim].min()) if rim.any() else float(world.height_at(*site))
    depth = rng.uniform(14.0, 26.0)
    inside = rim_min - 1.0 - depth * np.clip(1.0 - r * r, 0, None) ** 0.8
    outside = rim_min - 1.0 + np.clip(r - 1.0, 0, None) * radius * np.tan(np.radians(14.0))
    target = np.where(r < 1.0, inside, outside)
    w = 1.0 - smoothstep(2.0, 2.8, r)
    world.height_base = hb + w * (np.minimum(hb, target) - hb)


def _grotto(world: World, site, rng) -> CaveSystem:
    """A short cave dug straight into a hillside, ending in a chamber."""
    cfg = world.cfg.caves
    gx, gy = np.gradient(world.height, world.cell)
    i = int(np.clip(site[0] / world.cell, 0, world.size - 1))
    j = int(np.clip(site[1] / world.cell, 0, world.size - 1))
    d = np.array([gx[j, i], gy[j, i]])
    d = d / (np.linalg.norm(d) + 1e-9)          # uphill = into the hill
    z0 = float(world.height_at(*site)) + 1.0
    pos = np.array([site[0] - d[0] * 6.0, site[1] - d[1] * 6.0, z0])
    segs = []
    r = cfg.tunnel_radius * 1.1
    length = rng.uniform(120.0, 260.0)
    for _ in range(int(length / cfg.step)):
        ang = rng.normal(0, 0.22)
        c, s_ = np.cos(ang), np.sin(ang)
        d = np.array([d[0] * c - d[1] * s_, d[0] * s_ + d[1] * c])
        surf = float(world.height_at(pos[0] + d[0] * cfg.step, pos[1] + d[1] * cfg.step))
        dz = np.clip((surf - cfg.min_cover * 1.4 - pos[2]) / 20.0, -0.45, 0.1)
        nxt = pos + np.array([d[0], d[1], dz]) * cfg.step
        segs.append([*pos, *nxt, r, r * 0.98])
        pos = nxt
    ch_r = r * rng.uniform(2.6, 3.6)
    idx = len(world.caves) + len(world.forced_caves)
    return CaveSystem(
        id=f"cave_ai_{idx}", kind="grotto", segments=np.array(segs),
        chambers=np.array([[pos[0], pos[1], pos[2] + ch_r * 0.3, ch_r, ch_r * 0.9, ch_r * 0.6]]),
        entrances=[{"x": round(site[0], 2), "y": round(site[1], 2), "z": round(z0 - 1.0, 2), "kind": "grotto",
                    "dir": [round(float(d[0]), 3), round(float(d[1]), 3), 0.0]}],
        main_count=len(segs),
    )
