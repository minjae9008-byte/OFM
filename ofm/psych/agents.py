"""Curiosity-driven exploration agents (simulated play-testers).

Agents walk a navigation grid with Tobler's hiking function as the speed
model (players slow down uphill, cliffs and deep water are impassable, caves
connect their entrances). Each agent keeps a cognitive map of what it has
seen and repeatedly chooses the goal with the best utility per travel time:

  * known but unvisited POIs and landmarks (goal-directed / "achiever"),
  * frontier cells next to unseen land (information gain / "explorer"),

weighted by its profile (``evaluation.profiles``) and some randomness.
While walking it records experience events (landmark in view, POI discovered
or reached, new biome, vista, water, cave entered) with designer-set values
(``evaluation.events``). The model also tracks:

* boredom - time since the last event at or above ``strong_threshold``
  beyond the profile's patience; accumulated per place into a boredom map;
* fatigue - climbing effort above the profile's stamina slows the agent until
  it recovers on level ground (energetic cost of uphill walking);
* orientation - after ``orientation_lost_s`` without any landmark in view the
  agent counts as lost (wayfinding needs visible anchors, Lynch 1960);
* backtracking - steps onto cells it already walked (route redundancy);
* intensity - the timeline of event values, i.e. the pacing curve.

Every play-test is repeated ``evaluation.runs`` times with independent seeds
(Monte Carlo), so the evaluation can report confidence intervals.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from ..config import AgentProfile
from ..grid import bilinear, box_blur
from ..noise import derive_seed
from .perception import Perception

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DISCOVER_RADIUS_M = 650.0
TOBLER0 = float(np.exp(-3.5 * 0.05))


@dataclass
class AgentLog:
    personality: str                       # profile name (kept for backwards compatibility)
    profile: str = ""
    run: int = 0
    path: list[int] = field(default_factory=list)
    events: list[tuple[float, str, float, str]] = field(default_factory=list)  # t, kind, value, ref
    bored_seconds: float = 0.0
    duration: float = 0.0
    gaps: list[float] = field(default_factory=list)
    lost_seconds: float = 0.0
    fatigue_seconds: float = 0.0
    steps: int = 0
    backtrack_steps: int = 0
    discoveries: int = 0
    intensity: list[tuple[float, float]] = field(default_factory=list)  # (t, summed event value)

    @property
    def backtrack_ratio(self) -> float:
        return self.backtrack_steps / self.steps if self.steps else 0.0

    @property
    def bored_fraction(self) -> float:
        return self.bored_seconds / self.duration if self.duration > 0 else 0.0

    @property
    def lost_fraction(self) -> float:
        return self.lost_seconds / self.duration if self.duration > 0 else 0.0


@dataclass
class NavGraph:
    """Edges as parallel lists per node: neighbour, base walking time (s at run speed),
    signed slope, steepness risk (0..1), and whether the edge is a cave tunnel."""
    nbr: list[list[int]]
    time: list[list[float]]
    slope: list[list[float]]
    risk: list[list[float]]
    cave: list[list[bool]]
    blocked: np.ndarray

    def cost_lists(self, risk_aversion: float) -> list[list[tuple[int, float]]]:
        """Planning costs for a profile: walking time inflated on risky (steep, cliff-side) edges."""
        out = []
        for nb, tm, rk in zip(self.nbr, self.time, self.risk):
            out.append([(v, t * (1.0 + 2.0 * risk_aversion * r)) for v, t, r in zip(nb, tm, rk)])
        return out


def build_nav(world, per: Perception) -> NavGraph:
    pc = world.cfg.psych
    G = per.grid
    cs = per.cell
    cell = world.cell
    gi, gj = np.meshgrid(np.arange(G), np.arange(G))
    cx = (gi + 0.5) * cs
    cy = (gj + 0.5) * cs
    h = bilinear(world.ground(), cx / cell - 0.5, cy / cell - 0.5)      # the exported ground
    # deep water blocks, rivers do not
    water_frac = box_blur((world.ocean | world.lake_mask).astype(float), max(1, int(cs / cell / 2)))
    water = bilinear(water_frac, cx / cell - 0.5, cy / cell - 0.5) > 0.5
    cliff_frac = bilinear(box_blur(world.cliff_mask.astype(float), max(1, int(cs / cell / 2))),
                          cx / cell - 0.5, cy / cell - 0.5)
    blocked = water | (cliff_frac > 0.55)
    max_s = np.tan(np.radians(pc.max_walk_slope_deg))
    n = G * G
    nav = NavGraph([[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)],
                   [[] for _ in range(n)], [[] for _ in range(n)], blocked)
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
                speed = pc.run_speed * np.exp(-3.5 * abs(s + 0.05)) / TOBLER0
                penalty = 1.0 + 1.5 * cliff_frac[y, x]
                nav.nbr[a].append(y * G + x)
                nav.time[a].append(d / max(speed, 0.3) * penalty)
                nav.slope[a].append(float(s))
                nav.risk[a].append(float(min(1.0, max(abs(s) / max_s, cliff_frac[y, x]))))
                nav.cave[a].append(False)
    # caves are tunnels between their entrances (slow, but a shortcut through the mountain)
    for cave in world.caves:
        ents = cave.entrances
        if len(ents) < 2:
            continue
        ids = [int(np.clip(e["y"] / cs, 0, G - 1)) * G + int(np.clip(e["x"] / cs, 0, G - 1)) for e in ents]
        t = cave.length / (pc.run_speed * 0.6)
        for a in ids:
            for b in ids:
                if a != b:
                    nav.nbr[a].append(b)
                    nav.time[a].append(t)
                    nav.slope[a].append(0.0)
                    nav.risk[a].append(0.2)
                    nav.cave[a].append(True)
    return nav


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


def agent_roster(world) -> list[AgentProfile]:
    """One entry per simulated agent: profile counts, or psych.agents cycling the profiles."""
    profiles = [p for p in world.cfg.evaluation.profiles if p.count > 0] or list(world.cfg.evaluation.profiles)
    n = world.cfg.psych.agents
    if n > 0:
        return [profiles[i % len(profiles)] for i in range(n)]
    return [p for p in profiles for _ in range(max(0, p.count))]


def simulate_agents(world, per: Perception, landmarks, lm_visible: np.ndarray) -> dict:
    pc = world.cfg.psych
    ev = world.cfg.evaluation
    E = ev.events
    G = per.grid
    P = G * G
    cs = per.cell
    nav = build_nav(world, per)
    blocked = nav.blocked

    def cell_of(x, y):
        return int(np.clip(y / cs, 0, G - 1)) * G + int(np.clip(x / cs, 0, G - 1))

    poi_cells = np.array([cell_of(p.x, p.y) for p in world.pois], dtype=np.int64)
    poi_imp = np.array([p.importance for p in world.pois])
    lm_cells = np.array([cell_of(l.x, l.y) for l in landmarks], dtype=np.int64)
    lm_sal = np.array([l.salience for l in landmarks])
    lm_any = lm_visible.any(axis=1) if lm_visible.size else np.zeros(P, dtype=bool)
    bi = np.clip((per.ox / world.cell).astype(int), 0, world.size - 1)
    bj = np.clip((per.oy / world.cell).astype(int), 0, world.size - 1)
    cell_biome = world.biome[bj, bi]
    water_cells = np.flatnonzero(bilinear((world.water | world.river_mask).astype(float),
                                          per.ox / world.cell - 0.5, per.oy / world.cell - 0.5) > 0.3)
    vista = per.vista.ravel()

    empty = {"agents": [], "boredom": np.zeros((G, G)), "visits": np.zeros((G, G)),
             "lost": np.zeros((G, G)), "spawn": -1, "blocked": blocked, "runs": ev.runs}
    walk = np.flatnonzero(~blocked.ravel() & np.array([len(e) > 0 for e in nav.nbr]))
    if walk.size == 0:
        return empty
    # spawn: fixed per world at its first play-test (the first settlement, otherwise the walkable
    # cell nearest the centre), so the AI director compares candidates from the same start even
    # if an edit adds, moves or floods a settlement; if the spot is not walkable any more the
    # nearest walkable cell is used
    cxy = np.stack([per.ox[walk], per.oy[walk]], 1)
    if world.spawn is None:
        spawns = [p for p in world.pois if p.kind == "settlement"]
        if spawns:
            world.spawn = [float(spawns[0].x), float(spawns[0].y)]
        else:
            k = int(np.argmin(np.hypot(cxy[:, 0] - world.extent / 2, cxy[:, 1] - world.extent / 2)))
            world.spawn = [float(cxy[k, 0]), float(cxy[k, 1])]
    sx, sy = world.spawn
    spawn = cell_of(sx, sy)
    if spawn not in set(walk.tolist()):
        spawn = int(walk[np.argmin(np.hypot(cxy[:, 0] - sx, cxy[:, 1] - sy))])

    boredom = np.zeros(P)
    visits = np.zeros(P)
    lost_map = np.zeros(P)
    logs: list[AgentLog] = []
    total_time = pc.agent_minutes * 60.0
    roster = agent_roster(world)
    cost_cache: dict[float, list] = {}
    sp_cache: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}

    def plan(risk: float, src: int):
        key = round(float(risk), 3)
        if key not in cost_cache:
            cost_cache[key] = nav.cost_lists(key)
        k2 = (key, src)
        if k2 not in sp_cache:
            sp_cache[k2] = dijkstra(cost_cache[key], src)
        return sp_cache[k2]

    for run in range(max(1, ev.runs)):
        rng = np.random.default_rng(derive_seed(world.cfg.world.seed, "agents", run))
        for a, prof in enumerate(roster):
            log = AgentLog(personality=prof.name, profile=prof.name, run=run)
            speed = max(0.05, prof.speed)
            patience = prof.boredom_s if prof.boredom_s > 0 else pc.boredom_seconds
            discover_r = DISCOVER_RADIUS_M * max(0.05, prof.view)
            stamina = max(1.0, prof.stamina_s)
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
            last_landmark = 0.0          # the start is familiar ground
            exertion = 0.0
            decisions = 0
            while t < total_time and decisions < 200:
                decisions += 1
                dist, prev = plan(prof.risk_aversion, pos)
                dist = dist / speed
                reach = np.isfinite(dist) & (dist <= total_time - t + 1.0)
                cand_cells, cand_util = [], []
                for k in np.flatnonzero(poi_seen & ~poi_done):
                    c = poi_cells[k]
                    if reach[c] and c != pos:
                        cand_cells.append(c)
                        cand_util.append(prof.poi * (0.5 + poi_imp[k]))
                for k in np.flatnonzero(lm_seen & ~lm_done):
                    c = lm_cells[k]
                    if reach[c] and c != pos:
                        cand_cells.append(c)
                        cand_util.append(prof.landmark * (0.5 + lm_sal[k]))
                unseen_density = box_blur((~seen).reshape(G, G).astype(float), 3).ravel()
                frontier = np.flatnonzero(seen & ~visited & reach & (unseen_density > 0.05))
                if frontier.size:
                    pick = frontier[np.argsort(-unseen_density[frontier])[:40]]
                    for c in pick:
                        cand_cells.append(int(c))
                        cand_util.append(prof.frontier * (0.3 + 1.2 * unseen_density[c]))
                cc = np.array(cand_cells, dtype=np.int64)
                util = np.array(cand_util, dtype=np.float64)
                ok = util > 0
                if not ok.any():
                    break
                cc, util = cc[ok], util[ok]
                score = np.log(util) - np.log1p(dist[cc] / 90.0) + rng.gumbel(size=len(cc)) * prof.randomness
                goal = int(cc[np.argmax(score)])
                route = _path(prev, goal)
                for u, v in zip(route[:-1], route[1:]):
                    k_edge = nav.nbr[u].index(v)
                    s = nav.slope[u][k_edge]
                    # fatigue: effort accumulates uphill, recovers on level or downhill ground
                    if s > 0.04:
                        exertion += nav.time[u][k_edge] / speed * min(1.0, s / 0.25)
                    elif s < 0.02:
                        exertion = max(0.0, exertion - 0.5 * nav.time[u][k_edge] / speed)
                    tired = float(np.clip((exertion - stamina) / stamina, 0.0, 1.0))
                    step_t = nav.time[u][k_edge] / speed / max(0.1, 1.0 - prof.fatigue_slowdown * tired)
                    if tired > 0:
                        log.fatigue_seconds += step_t
                    t += step_t
                    pos = v
                    log.steps += 1
                    if visited[v]:
                        log.backtrack_steps += 1
                    visited[v] = True
                    visits[v] += step_t
                    log.path.append(v)
                    new_seen = per.visible[v] & ~seen
                    seen |= per.visible[v]
                    events = []
                    if nav.cave[u][k_edge]:
                        events.append(("cave_entered", E.cave_entered, ""))
                    if len(lm_cells):
                        nl = lm_visible[v] & ~lm_seen
                        for k in np.flatnonzero(nl):
                            events.append(("landmark_seen", E.landmark_seen, landmarks[k].id))
                        lm_seen |= lm_visible[v]
                        for k in np.flatnonzero((lm_cells == v) & ~lm_done):
                            lm_done[k] = True
                            events.append(("landmark_reached", E.landmark_reached, landmarks[k].id))
                    if len(poi_cells):
                        d_poi = np.hypot(per.ox[poi_cells] - per.ox[v], per.oy[poi_cells] - per.oy[v])
                        newly = ~poi_seen & seen[poi_cells] & (d_poi < discover_r)
                        for k in np.flatnonzero(newly):
                            log.discoveries += 1
                            events.append(("poi_discovered",
                                           E.poi_discovered + E.poi_discovered_importance * poi_imp[k],
                                           world.pois[k].id))
                        poi_seen |= newly
                        for k in np.flatnonzero((poi_cells == v) & ~poi_done):
                            poi_done[k] = True
                            if not poi_seen[k]:
                                log.discoveries += 1
                            poi_seen[k] = True
                            events.append(("poi_reached", E.poi_reached + E.poi_reached_importance * poi_imp[k],
                                           world.pois[k].id))
                    b = int(cell_biome[v])
                    if b not in biomes:
                        biomes.add(b)
                        events.append(("new_biome", E.new_biome, str(b)))
                    if vista[v] > 0.6 and t - last_vista > E.vista_cooldown_s:
                        last_vista = t
                        events.append(("vista", E.vista, str(v)))
                    if not saw_water and new_seen[water_cells].any():
                        saw_water = True
                        events.append(("water_seen", E.water_seen, ""))
                    for kind, val, ref in events:
                        log.events.append((round(t, 1), kind, round(float(val), 3), ref))
                    if events:
                        log.intensity.append((round(t, 1), round(float(sum(e[1] for e in events)), 3)))
                    if any(e[1] >= E.strong_threshold for e in events):
                        log.gaps.append(t - last_event)
                        last_event = t
                    elif t - last_event > patience:
                        log.bored_seconds += step_t
                        boredom[v] += step_t
                    # orientation: a visible landmark resets the clock
                    if lm_any[v]:
                        last_landmark = t
                    elif t - last_landmark > ev.orientation_lost_s:
                        log.lost_seconds += step_t
                        lost_map[v] += step_t
                    if t >= total_time:
                        break
            log.duration = t
            if t > last_event:
                log.gaps.append(t - last_event)
            logs.append(log)
    runs = max(1, ev.runs)
    return {
        "agents": logs,
        "boredom": boredom.reshape(G, G) / runs,
        "visits": visits.reshape(G, G) / runs,
        "lost": lost_map.reshape(G, G) / runs,
        "spawn": spawn,
        "blocked": blocked,
        "runs": runs,
    }
