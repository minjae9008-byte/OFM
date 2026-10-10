"""Curiosity-driven exploration agents (simulated play-testers).

Agents walk a navigation grid with Tobler's hiking function as the speed
model (players slow down uphill, cliffs and deep water are impassable, caves
connect their entrances). Each agent keeps a cognitive map of what it has
seen and repeatedly chooses the goal with the best utility per travel time:

  * known but unvisited POIs and landmarks (goal-directed / "achiever"),
  * frontier cells next to unseen land (information gain / "explorer"),

with a personality weighting and some randomness. While walking, it records
experience events (new landmark in view, POI discovered or reached, new
biome, vista, water). Long stretches without events are boredom: the time
and place are accumulated into a boredom map, which is exactly the signal
the AI director uses to decide where the map needs more content.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from ..grid import bilinear, box_blur
from ..noise import derive_seed
from .perception import Perception

PERSONALITIES = {
    "explorer": {"poi": 0.6, "landmark": 0.7, "frontier": 1.4},
    "achiever": {"poi": 1.5, "landmark": 0.9, "frontier": 0.5},
    "wanderer": {"poi": 0.8, "landmark": 1.0, "frontier": 1.0},
    "sightseer": {"poi": 0.7, "landmark": 1.6, "frontier": 0.7},
}

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class AgentLog:
    personality: str
    path: list[int] = field(default_factory=list)
    events: list[tuple[float, str, float, str]] = field(default_factory=list)  # t, kind, value, ref
    bored_seconds: float = 0.0
    duration: float = 0.0
    gaps: list[float] = field(default_factory=list)


def build_nav(world, per: Perception):
    pc = world.cfg.psych
    G = per.grid
    cs = per.cell
    cell = world.cell
    gi, gj = np.meshgrid(np.arange(G), np.arange(G))
    cx = (gi + 0.5) * cs
    cy = (gj + 0.5) * cs
    h = bilinear(world.height, cx / cell - 0.5, cy / cell - 0.5)
    # deep water blocks, rivers do not
    water_frac = box_blur((world.ocean | world.lake_mask).astype(float), max(1, int(cs / cell / 2)))
    water = bilinear(water_frac, cx / cell - 0.5, cy / cell - 0.5) > 0.5
    cliff_frac = bilinear(box_blur(world.cliff_mask.astype(float), max(1, int(cs / cell / 2))),
                          cx / cell - 0.5, cy / cell - 0.5)
    blocked = water | (cliff_frac > 0.55)
    max_s = np.tan(np.radians(pc.max_walk_slope_deg))
    edges: list[list[tuple[int, float]]] = [[] for _ in range(G * G)]
    tob0 = np.exp(-3.5 * 0.05)
    for j in range(G):
        for i in range(G):
            a = j * G + i
            if blocked[j, i]:
                continue
            for dy, dx in NB:
                y, x = j + dy, i + dx
                if not (0 <= y < G and 0 <= x < G) or blocked[y, x]:
                    continue
                d = cs * (1.4142 if dx and dy else 1.0)
                s = (h[y, x] - h[j, i]) / d
                if abs(s) > max_s:
                    continue
                speed = pc.run_speed * np.exp(-3.5 * abs(s + 0.05)) / tob0
                penalty = 1.0 + 1.5 * cliff_frac[y, x]
                edges[a].append((y * G + x, d / max(speed, 0.3) * penalty))
    # caves are tunnels between their entrances
    for cave in world.caves:
        ents = cave.entrances
        if len(ents) < 2:
            continue
        ids = [int(np.clip(e["y"] / cs, 0, G - 1)) * G + int(np.clip(e["x"] / cs, 0, G - 1)) for e in ents]
        t = cave.length / (pc.run_speed * 0.6)
        for a in ids:
            for b in ids:
                if a != b:
                    edges[a].append((b, t))
    return edges, blocked


def dijkstra(edges, src: int, limit: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
    n = len(edges)
    dist = [np.inf] * n
    prev = [-1] * n
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u] or d > limit:
            continue
        for v, w in edges[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return np.array(dist), np.array(prev)


def _path(prev: np.ndarray, dst: int) -> list[int]:
    out = []
    while dst >= 0:
        out.append(int(dst))
        dst = int(prev[dst])
    return out[::-1]


def simulate_agents(world, per: Perception, landmarks, lm_visible: np.ndarray) -> dict:
    pc = world.cfg.psych
    G = per.grid
    P = G * G
    cs = per.cell
    edges, blocked = build_nav(world, per)
    rng = np.random.default_rng(derive_seed(world.cfg.world.seed, "agents"))

    def cell_of(x, y):
        return int(np.clip(y / cs, 0, G - 1)) * G + int(np.clip(x / cs, 0, G - 1))

    poi_cells = np.array([cell_of(p.x, p.y) for p in world.pois], dtype=np.int64)
    poi_imp = np.array([p.importance for p in world.pois])
    lm_cells = np.array([cell_of(l.x, l.y) for l in landmarks], dtype=np.int64)
    lm_sal = np.array([l.salience for l in landmarks])
    bi = np.clip((per.ox / world.cell).astype(int), 0, world.size - 1)
    bj = np.clip((per.oy / world.cell).astype(int), 0, world.size - 1)
    cell_biome = world.biome[bj, bi]
    water_cells = np.flatnonzero(bilinear((world.water | world.river_mask).astype(float),
                                          per.ox / world.cell - 0.5, per.oy / world.cell - 0.5) > 0.3)
    vista = per.vista.ravel()
    discover_r = 650.0

    # spawn: a settlement, otherwise the most connected walkable cell near the centre
    walk = np.flatnonzero(~blocked.ravel() & np.array([len(e) > 0 for e in edges]))
    if walk.size == 0:
        return {"agents": [], "boredom": np.zeros((G, G)), "visits": np.zeros((G, G))}
    spawns = [p for p in world.pois if p.kind == "settlement"]
    if spawns:
        spawn = cell_of(spawns[0].x, spawns[0].y)
        if spawn not in set(walk.tolist()):
            spawn = int(walk[np.argmin(np.abs(walk - spawn))])
    else:
        cxy = np.stack([per.ox[walk], per.oy[walk]], 1)
        spawn = int(walk[np.argmin(np.hypot(cxy[:, 0] - world.extent / 2, cxy[:, 1] - world.extent / 2))])

    boredom = np.zeros(P)
    visits = np.zeros(P)
    logs: list[AgentLog] = []
    names = list(PERSONALITIES)
    total_time = pc.agent_minutes * 60.0
    n_agents = pc.agents or sum(max(0, p.count) for p in world.cfg.evaluation.profiles)
    for a in range(n_agents):
        pers = names[a % len(names)]
        wts = PERSONALITIES[pers]
        log = AgentLog(personality=pers)
        pos = spawn
        seen = per.visible[pos].copy()
        visited = np.zeros(P, dtype=bool)
        visited[pos] = True
        poi_seen = np.zeros(len(poi_cells), dtype=bool)
        poi_done = np.zeros(len(poi_cells), dtype=bool)
        lm_seen = np.zeros(len(lm_cells), dtype=bool)
        lm_done = np.zeros(len(lm_cells), dtype=bool)
        biomes = {int(cell_biome[pos])}
        saw_water = False
        last_vista = -1e9
        t = 0.0
        last_event = 0.0
        decisions = 0
        while t < total_time and decisions < 200:
            decisions += 1
            dist, prev = dijkstra(edges, pos, limit=total_time - t + 1.0)
            reach = np.isfinite(dist)
            cand_cells, cand_util = [], []
            for k in np.flatnonzero(poi_seen & ~poi_done):
                c = poi_cells[k]
                if reach[c] and c != pos:
                    cand_cells.append(c)
                    cand_util.append(wts["poi"] * (0.5 + poi_imp[k]))
            for k in np.flatnonzero(lm_seen & ~lm_done):
                c = lm_cells[k]
                if reach[c] and c != pos:
                    cand_cells.append(c)
                    cand_util.append(wts["landmark"] * (0.5 + lm_sal[k]))
            unseen_density = box_blur((~seen).reshape(G, G).astype(float), 3).ravel()
            frontier = np.flatnonzero(seen & ~visited & reach & (unseen_density > 0.05))
            if frontier.size:
                pick = frontier[np.argsort(-unseen_density[frontier])[:40]]
                for c in pick:
                    cand_cells.append(int(c))
                    cand_util.append(wts["frontier"] * (0.3 + 1.2 * unseen_density[c]))
            if not cand_cells:
                break
            cc = np.array(cand_cells)
            util = np.array(cand_util)
            score = np.log(util) - np.log1p(dist[cc] / 90.0) + rng.gumbel(size=len(cc)) * 0.25
            goal = int(cc[np.argmax(score)])
            route = _path(prev, goal)
            for u, v in zip(route[:-1], route[1:]):
                step_t = next((w for nb, w in edges[u] if nb == v), cs / pc.run_speed)
                t += step_t
                pos = v
                visited[v] = True
                visits[v] += step_t
                log.path.append(v)
                new_seen = per.visible[v] & ~seen
                seen |= per.visible[v]
                events = []
                # landmarks entering view
                if len(lm_cells):
                    nl = lm_visible[v] & ~lm_seen
                    for k in np.flatnonzero(nl):
                        events.append(("landmark_seen", 0.8, landmarks[k].id))
                    lm_seen |= lm_visible[v]
                    for k in np.flatnonzero((lm_cells == v) & ~lm_done):
                        lm_done[k] = True
                        events.append(("landmark_reached", 1.0, landmarks[k].id))
                if len(poi_cells):
                    d_poi = np.hypot(per.ox[poi_cells] - per.ox[v], per.oy[poi_cells] - per.oy[v])
                    newly = ~poi_seen & seen[poi_cells] & (d_poi < discover_r)
                    for k in np.flatnonzero(newly):
                        events.append(("poi_discovered", 0.5 * poi_imp[k] + 0.1, world.pois[k].id))
                    poi_seen |= newly
                    for k in np.flatnonzero((poi_cells == v) & ~poi_done):
                        poi_done[k] = True
                        poi_seen[k] = True
                        events.append(("poi_reached", 0.6 + 0.6 * poi_imp[k], world.pois[k].id))
                b = int(cell_biome[v])
                if b not in biomes:
                    biomes.add(b)
                    events.append(("new_biome", 0.4, str(b)))
                if vista[v] > 0.6 and t - last_vista > 180:
                    last_vista = t
                    events.append(("vista", 0.5, str(v)))
                if not saw_water and new_seen[water_cells].any():
                    saw_water = True
                    events.append(("water_seen", 0.35, ""))
                strong = [e for e in events if e[1] >= 0.35]
                for kind, val, ref in events:
                    log.events.append((round(t, 1), kind, round(float(val), 3), ref))
                if strong:
                    log.gaps.append(t - last_event)
                    last_event = t
                elif t - last_event > pc.boredom_seconds:
                    log.bored_seconds += step_t
                    boredom[v] += step_t
                if t >= total_time:
                    break
        log.duration = t
        if t > last_event:
            log.gaps.append(t - last_event)
        logs.append(log)
    return {
        "agents": logs,
        "boredom": boredom.reshape(G, G),
        "visits": visits.reshape(G, G),
        "spawn": spawn,
        "blocked": blocked,
    }
