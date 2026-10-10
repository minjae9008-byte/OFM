"""Surface hydrology: oceans, priority-flood drainage, water-balance lakes, rivers.

* Oceans are below-sea-level regions connected to the map border.
* Priority-Flood (Barnes et al. 2014) fills depressions and yields a drainage
  receiver for every cell, so all water finds a way to the sea or map edge.
* Each depression becomes a lake only if its inflow exceeds open-water
  evaporation; otherwise the level drops until the water balance closes
  (endorheic lake) and the rest of the basin dries into a salt flat (playa).
* Rivers follow the receiver tree where discharge exceeds a threshold; their
  channels are carved with hydraulic-geometry width/depth (w ~ Q^0.5).
"""
from __future__ import annotations

import heapq

import numpy as np

from ..config import OFMConfig
from ..grid import NEIGHBORS8, blur, border_mask, label
from .climate import evaporation_mm, runoff_fraction

SECONDS_PER_YEAR = 31_557_600.0


def ocean_mask(height: np.ndarray, sea_level: float) -> np.ndarray:
    below = height < sea_level
    if not below.any():
        return below
    lab, n = label(below, diagonal=False)
    touching = np.unique(lab[border_mask(height.shape) & below])
    touching = touching[touching > 0]
    return np.isin(lab, touching)


def _neighbor_table(h: int, w: int) -> list[tuple[int, ...]]:
    table = []
    for j in range(h):
        for i in range(w):
            nb = []
            for dy, dx in NEIGHBORS8:
                y, x = j + dy, i + dx
                if 0 <= y < h and 0 <= x < w:
                    nb.append(y * w + x)
            table.append(tuple(nb))
    return table


_NB_CACHE: dict[tuple[int, int], list[tuple[int, ...]]] = {}


def priority_flood(height: np.ndarray, sinks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (filled surface, receiver index per cell (-1 = outlet), flood order)."""
    hgt, wid = height.shape
    key = (hgt, wid)
    if key not in _NB_CACHE:
        _NB_CACHE[key] = _neighbor_table(hgt, wid)
    nbrs = _NB_CACHE[key]
    filled = height.reshape(-1).astype(np.float64).tolist()
    receiver = [-1] * (hgt * wid)
    closed = bytearray(hgt * wid)
    heap: list[tuple[float, int]] = []
    seeds = (sinks | border_mask(height.shape)).reshape(-1)
    for c in np.flatnonzero(seeds).tolist():
        closed[c] = 1
        heap.append((filled[c], c))
    heapq.heapify(heap)
    order = []
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        z, c = pop(heap)
        order.append(c)
        for nb in nbrs[c]:
            if closed[nb]:
                continue
            closed[nb] = 1
            if filled[nb] < z:
                filled[nb] = z
            receiver[nb] = c
            push(heap, (filled[nb], nb))
    return (np.array(filled).reshape(height.shape), np.array(receiver, dtype=np.int64),
            np.array(order, dtype=np.int64))


def accumulate(order: np.ndarray, receiver: np.ndarray, weights: np.ndarray,
               sink: np.ndarray | None = None) -> np.ndarray:
    acc = weights.reshape(-1).astype(np.float64).tolist()
    rec = receiver.tolist()
    blocked = sink.reshape(-1).tolist() if sink is not None else None
    for c in reversed(order.tolist()):
        r = rec[c]
        if r >= 0 and not (blocked and blocked[c]):
            acc[r] += acc[c]
    return np.array(acc).reshape(weights.shape)


def _chaikin(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    pts = points
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), pts.shape[1]))
        mid[0::2] = q
        mid[1::2] = r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def simulate_hydrology(cfg: OFMConfig, height: np.ndarray, temperature: np.ndarray,
                       precipitation: np.ndarray) -> dict:
    wc, hc = cfg.world, cfg.hydrology
    cell = wc.cell_size
    size = height.shape[0]
    ocean = ocean_mask(height, wc.sea_level)

    filled, receiver, order = priority_flood(height, ocean)
    depth = filled - height
    runoff_mm = precipitation * runoff_fraction(temperature, precipitation)
    # m3/s generated per cell
    local_q = runoff_mm / 1000.0 * cell * cell / SECONDS_PER_YEAR * hc.flow_scale
    local_q = np.where(ocean, 0.0, local_q)
    q = accumulate(order, receiver, local_q)

    # --- lakes from depressions -------------------------------------------
    candidate = (depth > hc.min_lake_depth) & ~ocean
    lab, n = label(candidate, diagonal=True)
    lake_mask = np.zeros_like(candidate)
    lake_level = np.full(height.shape, np.nan)
    playa = np.zeros_like(candidate)
    sink = np.zeros_like(candidate)
    lakes = []
    evap = evaporation_mm(temperature)
    h_out = height.copy()
    for k in range(1, n + 1):
        cells = lab == k
        count = int(cells.sum())
        if count < hc.min_lake_cells:
            h_out[cells] = filled[cells]           # tiny pits are filled in
            continue
        inflow = float(q[cells].max()) / hc.flow_scale   # real (unexaggerated) inflow, m3/s
        evap_rate = float(evap[cells].mean()) / 1000.0 / SECONDS_PER_YEAR  # m/s
        spill = float(filled[cells].max())
        full_loss = count * cell * cell * evap_rate
        floor = np.sort(height[cells])
        if inflow >= full_loss:
            level = spill
            closed = False
        else:
            n_wet = int(np.clip(inflow / (cell * cell * evap_rate), 1, count))
            level = float(floor[n_wet - 1]) + 0.3
            closed = True
        wet = cells & (height < level)
        if wet.sum() < max(2, hc.min_lake_cells // 2):
            playa |= cells
            sink |= cells
            continue
        lake_mask |= wet
        lake_level[wet] = level
        if closed:
            sink |= cells
            playa |= cells & ~wet
        jj, ii = np.nonzero(wet)
        lakes.append({
            "id": f"lake_{len(lakes)}",
            "level": level,
            "area_m2": float(wet.sum() * cell * cell),
            "max_depth": float((level - height[wet]).max()),
            "closed_basin": closed,
            "inflow_m3s": inflow * hc.flow_scale,
            "center": [float((ii.mean() + 0.5) * cell), float((jj.mean() + 0.5) * cell)],
            "saline": bool(closed and float(temperature[cells].mean()) > 8.0),
        })

    if sink.any():
        q = accumulate(order, receiver, local_q, sink=sink)

    # --- rivers -------------------------------------------------------------
    water = ocean | lake_mask
    river = (q >= hc.river_min_discharge) & ~water
    width = hc.channel_width_k * np.sqrt(np.maximum(q, 0.0))
    chan_depth = hc.channel_depth_k * np.maximum(q, 0.0) ** 0.4
    carve = np.where(river, chan_depth + hc.valley_carve * np.log1p(q), 0.0)
    carve = np.maximum(carve, blur(carve, max(1.0, 1.2)) * 0.8)
    h_out = h_out - np.where(lake_mask | ocean, 0.0, carve)

    rivers, waterfalls = _trace_rivers(river, receiver, q, h_out, width, chan_depth, water, cell, hc)

    # Water table: subdued copy of topography relative to local base level.
    base = blur(np.where(water | river, h_out, filled), size * 0.06)
    water_table = np.minimum(h_out - 2.0, base + 0.3 * (h_out - base))
    water_table = np.where(lake_mask, lake_level, water_table)

    return {
        "height": h_out,
        "ocean": ocean,
        "filled": filled,
        "receiver": receiver.reshape(height.shape),
        "order": order,
        "discharge": q,
        "lake_mask": lake_mask,
        "lake_level": np.where(lake_mask, lake_level, np.nan),
        "lakes": lakes,
        "playa": playa & ~lake_mask,
        "river_mask": river,
        "rivers": rivers,
        "waterfalls": waterfalls,
        "water_table": water_table,
    }


def _trace_rivers(river, receiver, q, h, width, chan_depth, water, cell, hc):
    size = river.shape[1]
    rec = receiver.reshape(-1)
    rflat = river.reshape(-1)
    donors_in_river = np.zeros(rflat.size, dtype=np.int32)
    src = np.flatnonzero(rflat & (rec >= 0))
    np.add.at(donors_in_river, rec[src], 1)
    heads = np.flatnonzero(rflat & (donors_in_river == 0))
    heads = heads[np.argsort(-h.reshape(-1)[heads])]
    visited = np.zeros(rflat.size, dtype=bool)
    hflat = h.reshape(-1)
    qflat = q.reshape(-1)
    wflat = water.reshape(-1)
    rivers, falls = [], []
    for head in heads.tolist():
        path = []
        c = head
        joined = None
        while c >= 0:
            if visited[c]:
                joined = c
                path.append(c)
                break
            visited[c] = True
            path.append(c)
            if wflat[c]:
                break
            nxt = rec[c]
            if nxt < 0:
                break
            if not rflat[nxt] and not wflat[nxt]:
                path.append(nxt)
                break
            c = nxt
        if len(path) < 4:
            continue
        p = np.array(path)
        jj, ii = np.divmod(p, size)
        z = hflat[p].copy()
        for k in range(1, len(z)):            # rivers never flow uphill
            z[k] = min(z[k], z[k - 1])
        own = p[:-1] if joined is not None else p   # a junction cell belongs to the river already traced
        hflat[own] = np.minimum(hflat[own], z[:len(own)])
        pts = np.stack([(ii + 0.5) * cell, (jj + 0.5) * cell, z, qflat[p]], axis=1)
        drops = np.concatenate([[0.0], z[:-1] - z[1:]])
        for k in range(1, len(z)):
            drop = drops[k]
            upstream = drops[max(1, k - 4):k]
            knick = drop > 2.2 * (upstream.mean() if len(upstream) else 0.0) + 1.0
            if drop >= hc.waterfall_min_drop and knick and qflat[p[k]] >= hc.river_min_discharge:
                falls.append({
                    "x": float(pts[k - 1, 0]), "y": float(pts[k - 1, 1]),
                    "top": float(z[k - 1]), "bottom": float(z[k]), "drop": float(drop),
                    "discharge": float(qflat[p[k]]),
                })
        smooth = _chaikin(pts, 2)
        widths = hc.channel_width_k * np.sqrt(np.maximum(smooth[:, 3], 0.0))
        depths = hc.channel_depth_k * np.maximum(smooth[:, 3], 0.0) ** 0.4
        rivers.append({
            "id": f"river_{len(rivers)}",
            "points": [[float(x), float(y), float(zz + d * 0.8)] for (x, y, zz, _), d in zip(smooth, depths)],
            "width": [float(w) for w in widths],
            "discharge": [float(v) for v in smooth[:, 3]],
            "mouth": "join" if joined is not None else ("water" if wflat[p[-1]] else "edge"),
            "length_m": float(np.sum(np.hypot(np.diff(smooth[:, 0]), np.diff(smooth[:, 1])))),
        })
    falls = _merge_falls(falls, cell * 2.5)
    return rivers, falls


def _merge_falls(falls: list[dict], radius: float) -> list[dict]:
    falls = sorted(falls, key=lambda f: -f["drop"])
    kept: list[dict] = []
    for f in falls:
        if all(np.hypot(f["x"] - k["x"], f["y"] - k["y"]) > radius for k in kept):
            kept.append(f)
    for i, f in enumerate(kept):
        f["id"] = f"waterfall_{i}"
    return kept
