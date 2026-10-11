"""Points of interest derived from the simulated geography.

Every POI is anchored to something the simulation actually produced: cave
mouths, waterfalls (knickpoints), lake shores, river confluences, summits
with real prominence, coves, rock formations and settlement sites (flat,
fertile land near fresh water). Greedy selection by importance with a
minimum spacing keeps them evenly distributed.
"""
from __future__ import annotations

import numpy as np

from ..grid import bilinear, box_blur, chamfer_distance, sliding_max, sliding_min, slope
from ..noise import derive_seed
from ..physics.biome import B
from ..world import POI
from .lakes import lake_surface, under_lake


def _hz(world, x, y):
    return float(bilinear(world.hires, x / world.hires_spacing, y / world.hires_spacing))


def poi_candidates(world) -> list[POI]:
    cell = world.cell
    h = world.height
    cands: list[POI] = []
    surface = lake_surface(world)

    def add(kind, x, y, imp, z=None, props=()):
        z = float(_hz(world, x, y) if z is None else z)
        if under_lake(world, x, y, z, surface):          # drowned by a lake sheet
            return
        cands.append(POI(id="", kind=kind, x=float(x), y=float(y), z=z,
                         importance=float(imp), region=world.region_of(x, y), props=list(props)))

    for cave in world.caves:
        for e in cave.entrances:
            imp = 0.95 if cave.kind != "sea" else 0.8
            add("cave_entrance", e["x"], e["y"], imp, z=e["z"], props=[cave.id, e["kind"]])
    for f in world.waterfalls:
        add("waterfall", f["x"], f["y"], 0.7 + min(0.25, f["drop"] / 80.0), props=[f["id"]])
    for f in world.features:
        add("rock_formation", f.x, f.y, 0.85, props=[f.id, f.kind])

    # Summits with prominence
    r = max(3, int(350.0 / cell))
    peak = (h >= sliding_max(h, r) - 1e-6) & ~world.water
    prom = h - sliding_min(h, r * 2)
    jj, ii = np.nonzero(peak & (prom > 60.0))
    for j, i in zip(jj, ii):
        add("summit", (i + 0.5) * cell, (j + 0.5) * cell, 0.55 + min(0.35, prom[j, i] / 600.0))

    # Lake shores: flattest shore cell of each lake
    s = slope(h, cell)
    if world.lake_mask.any():
        d_lake = chamfer_distance(world.lake_mask, max_dist=3)
        shore = (d_lake > 0.5) & (d_lake <= 2.0) & ~world.water
        for lake in world.lakes:
            cx, cy = lake["center"]
            jj, ii = np.nonzero(shore)
            if len(ii) == 0:
                break
            d = np.hypot((ii + 0.5) * cell - cx, (jj + 0.5) * cell - cy)
            near = d < np.sqrt(lake["area_m2"]) * 1.2 + 3 * cell
            if not near.any():
                continue
            k = np.argmin(np.where(near, s[jj, ii], np.inf))
            add("lakeshore", (ii[k] + 0.5) * cell, (jj[k] + 0.5) * cell, 0.6 + (0.1 if lake["closed_basin"] else 0.0),
                props=[lake["id"]])

    # River confluences
    if world.river_mask.any():
        rec = world.receiver.reshape(-1)
        rflat = world.river_mask.reshape(-1)
        src = np.flatnonzero(rflat & (rec >= 0))
        donors = np.bincount(rec[src][rflat[rec[src]]], minlength=rflat.size)
        for c in np.flatnonzero(donors >= 2)[:40]:
            j, i = divmod(int(c), world.size)
            add("confluence", (i + 0.5) * cell, (j + 0.5) * cell, 0.45)

    # Coves: concave coastline (ocean wraps around the point)
    if world.ocean.any():
        enclosure = box_blur(world.ocean.astype(float), 4)
        coast = (chamfer_distance(world.ocean, max_dist=2) == 1) & (s < 0.2)
        jj, ii = np.nonzero(coast & (enclosure > 0.62))
        for j, i in list(zip(jj, ii))[:30]:
            add("cove", (i + 0.5) * cell, (j + 0.5) * cell, 0.5)

    # Settlement sites: flat, temperate, fresh water nearby, not too high
    fresh = chamfer_distance(world.river_mask | world.lake_mask, max_dist=8)
    good = ((s < 0.08) & (fresh <= 5) & ~world.water & (world.temperature > 4)
            & ~np.isin(world.biome, [B["wetland"], B["salt_flat"], B["desert"]]))
    jj, ii = np.nonzero(good)
    if len(ii):
        rng = np.random.default_rng(derive_seed(world.cfg.world.seed, "settle"))
        for k in rng.permutation(len(ii))[:30]:
            add("settlement", (ii[k] + 0.5) * cell, (jj[k] + 0.5) * cell, 0.75)
    return cands


def select_pois(cands: list[POI], count: int, spacing: float, existing: list[POI] | None = None) -> list[POI]:
    chosen: list[POI] = list(existing or [])
    by_kind: dict[str, int] = {}
    for q in chosen:
        by_kind[q.kind] = by_kind.get(q.kind, 0) + 1
    caps = {"settlement": 2, "confluence": 3, "cove": 2, "summit": 5, "cave_entrance": 8}
    pool = list(cands)
    while pool and len(chosen) < count:
        # diminishing returns per kind keeps the POI mix varied
        scores = [p.importance - 0.07 * by_kind.get(p.kind, 0) for p in pool]
        k = int(np.argmax(scores))
        p = pool.pop(k)
        if by_kind.get(p.kind, 0) >= caps.get(p.kind, 99):
            continue
        min_d = spacing * (0.5 if p.kind == "cave_entrance" else 1.0)
        if any(np.hypot(p.x - q.x, p.y - q.y) < min_d for q in chosen):
            continue
        chosen.append(p)
        by_kind[p.kind] = by_kind.get(p.kind, 0) + 1
    for k, p in enumerate(chosen):
        p.id = f"poi_{k}"
    return chosen


def place_pois(world) -> list[POI]:
    dc = world.cfg.detail
    pinned = [p for p in world.pois if p.source in ("ai", "psych")]
    for p in pinned:  # keep director-placed POIs on the rebuilt ground
        p.z = _hz(world, p.x, p.y)
        p.region = world.region_of(p.x, p.y)
    # The agents start at the first settlement. A director rebuild re-runs the selection and
    # an edit anywhere could move or remove that settlement, so candidates would be scored
    # from a different start. The selection itself runs unchanged (a rebuild without an edit
    # reproduces the same POIs); only the first settlement is made the previous spawn again,
    # unless the edit flooded a procedural one.
    spawn = next((p for p in world.pois if p.kind == "settlement"), None)
    chosen = select_pois(poi_candidates(world), dc.poi_count + len(pinned), dc.poi_spacing, pinned)
    if spawn is not None and spawn.source not in ("ai", "psych"):
        i = int(np.clip(spawn.x / world.cell, 0, world.size - 1))
        j = int(np.clip(spawn.y / world.cell, 0, world.size - 1))
        if not (world.water[j, i] or world.river_mask[j, i]):
            chosen = _keep_spawn(world, chosen, spawn, dc.poi_count + len(pinned))
            for k, p in enumerate(chosen):
                p.id = f"poi_{k}"
    return chosen


def _keep_spawn(world, chosen: list[POI], spawn: POI, limit: int) -> list[POI]:
    def same(p):
        return p.kind == "settlement" and abs(p.x - spawn.x) < 1e-6 and abs(p.y - spawn.y) < 1e-6

    first = next((k for k, p in enumerate(chosen) if p.kind == "settlement"), None)
    if first is not None and same(chosen[first]):
        return chosen
    k = next((k for k, p in enumerate(chosen) if same(p)), None)
    if k is not None:
        p = chosen.pop(k)
    else:
        p = spawn
        p.z = _hz(world, p.x, p.y)
        p.region = world.region_of(p.x, p.y)
        if len(chosen) >= limit:          # keep the count: drop the last procedural pick
            drop = next((k for k in range(len(chosen) - 1, -1, -1) if chosen[k].source == "procedural"), None)
            if drop is not None:
                chosen.pop(drop)
                if first is not None and drop < first:
                    first -= 1
    chosen.insert(first if first is not None else 0, p)
    return chosen
