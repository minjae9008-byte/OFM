"""Combines perception, landmarks and agent play-tests into an interest model.

Per-location *preference* follows Kaplan & Kaplan's matrix (coherence,
complexity, legibility, mystery) plus Appleton's prospect-refuge, weighted by
``evaluation.preference``. The map-level sub-scores (0-10) are:

* scenic, variety, mystery, prospect_refuge - from perception and content;
* pacing      - mean gap between strong events and bored share of time, each
                scored against its designer band (``evaluation.targets``);
* exploration - share of reachable land seen and POI discoveries per hour;
* legibility  - share of land from which a landmark is visible;
* orientation - time spent lost and route backtracking;
* terrain_quality - the geomorphometric analysis (``ofm/quality``), if run.

``overall`` is their weighted mean (``evaluation.weights``, renormalised over
the available sub-scores) and is the objective the AI director optimises.
Agent-level metrics come with Monte Carlo statistics (mean, std, confidence
interval over all simulated agents); the overall score also gets an interval
over the ``evaluation.runs`` repetitions. Every profile gets its own breakdown.
"""
from __future__ import annotations

import numpy as np

from ..grid import resample
from ..physics.biome import BIOME_NAMES, B
from ..world import World
from .agents import AgentLog, simulate_agents
from .landmarks import select_landmarks
from .perception import compute_perception
from .scoring import band_score, rounded, summarize, weighted_overall


def _mean(xs) -> float:
    xs = [float(x) for x in xs]
    return float(np.mean(xs)) if xs else 0.0


def _agent_metrics(logs: list[AgentLog], per, reachable: np.ndarray, n_pois: int,
                   strong_threshold: float = 0.35) -> dict:
    """Pooled play-test metrics of a group of agents."""
    G = per.grid
    if not logs:
        return {"bored_fraction": 0.0, "mean_event_gap_s": 0.0, "p90_event_gap_s": 0.0, "events_per_minute": 0.0,
                "coverage_seen": 0.0, "poi_discovery": 0.0, "discoveries_per_agent_hour": 0.0,
                "lost_fraction": 0.0, "backtrack_ratio": 0.0, "fatigue_fraction": 0.0}
    total = sum(l.duration for l in logs) or 1.0
    gaps = np.concatenate([np.array(l.gaps) for l in logs if l.gaps]) if any(l.gaps for l in logs) \
        else np.array([total / len(logs)])
    strong = sum(1 for l in logs for e in l.events if e[2] >= strong_threshold)
    coverage = []
    found: set[str] = set()
    for l in logs:
        seen = np.zeros(G * G, dtype=bool)
        for c in set(l.path):
            seen |= per.visible[c]
        coverage.append(float(seen.reshape(G, G)[reachable].mean()) if reachable.any() else 0.0)
        found |= {e[3] for e in l.events if e[1] in ("poi_discovered", "poi_reached")}
    return {
        "bored_fraction": sum(l.bored_seconds for l in logs) / total,
        "mean_event_gap_s": float(gaps.mean()),
        "p90_event_gap_s": float(np.quantile(gaps, 0.9)),
        "events_per_minute": strong / (total / 60.0),
        "coverage_seen": _mean(coverage),
        "poi_discovery": len(found) / max(1, n_pois),
        "discoveries_per_agent_hour": sum(l.discoveries for l in logs) / (total / 3600.0),
        "lost_fraction": sum(l.lost_seconds for l in logs) / total,
        "backtrack_ratio": sum(l.backtrack_steps for l in logs) / max(1, sum(l.steps for l in logs)),
        "fatigue_fraction": sum(l.fatigue_seconds for l in logs) / total,
    }


def _behaviour_scores(m: dict, landmark_coverage: float, ev) -> dict:
    """Band-scored sub-scores that depend on how agents experienced the map."""
    tg, soft = ev.targets, ev.band_softness

    def avg(*xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    return {
        "pacing": avg(band_score(m["mean_event_gap_s"], tg.event_gap_s, soft),
                      band_score(m["bored_fraction"], tg.bored_fraction, soft)),
        "exploration": avg(band_score(m["coverage_seen"], tg.coverage_seen, soft),
                           band_score(m["discoveries_per_agent_hour"], tg.discoveries_per_hour, soft)),
        "legibility": band_score(landmark_coverage, tg.landmark_coverage, soft),
        "orientation": avg(band_score(m["lost_fraction"], tg.lost_fraction, soft),
                           band_score(m["backtrack_ratio"], tg.backtrack_ratio, soft)),
    }


def _intensity_curve(logs: list[AgentLog], minutes: float) -> list[float]:
    """Mean experienced event value per minute across agents (the pacing curve)."""
    n_bins = max(1, int(np.ceil(minutes)))
    curve = np.zeros(n_bins)
    for l in logs:
        for t, v in l.intensity:
            curve[min(n_bins - 1, int(t // 60))] += v
    return [round(float(c), 3) for c in curve / max(1, len(logs))]


def evaluate_psychology(world: World) -> dict:
    ev = world.cfg.evaluation
    per = compute_perception(world)
    landmarks, lm_vis = select_landmarks(world, per)
    world.landmarks = landmarks
    G = per.grid
    land = per.land
    pw = ev.preference
    legib = np.clip(lm_vis.sum(axis=1) / 3.0, 0, 1).reshape(G, G) if lm_vis.size else np.zeros((G, G))
    pr = np.sqrt(np.clip(per.prospect, 0, 1) * np.clip(per.refuge + 0.15, 0, 1))
    wsum = max(1e-9, pw.coherence + pw.complexity + pw.legibility + pw.mystery + pw.prospect_refuge)
    preference = (pw.coherence * per.coherence + pw.complexity * per.complexity + pw.legibility * legib
                  + pw.mystery * per.mystery + pw.prospect_refuge * pr) / wsum
    sim = simulate_agents(world, per, landmarks, lm_vis)
    logs = sim["agents"]
    reachable = land & ~sim["blocked"]

    land_full = ~world.water
    biome_counts = np.bincount(world.biome[land_full].ravel(), minlength=len(BIOME_NAMES)).astype(float)
    p = biome_counts[biome_counts > 0] / biome_counts.sum() if biome_counts.sum() else np.array([1.0])
    biome_entropy = float(-(p * np.log(p)).sum() / np.log(8))
    feature_kinds = {
        "caves": bool(world.caves), "waterfalls": bool(world.waterfalls), "lakes": bool(world.lakes),
        "rivers": bool(world.rivers), "coast": bool(world.ocean.any()),
        "rock_formations": bool(world.features), "cliffs": bool(world.cliff_mask.any()),
    }
    variety = 0.55 * min(1.0, biome_entropy) + 0.45 * (sum(feature_kinds.values()) / len(feature_kinds))
    lm_cov = float((legib[land] > 0).mean()) if land.any() else 0.0
    cave_mouths = sum(len(c.entrances) for c in world.caves)
    mystery_score = 0.7 * float(per.mystery[land].mean() if land.any() else 0) + 0.3 * min(1.0, cave_mouths / 8.0)
    terrain = (world.evaluation.get("terrain") or {}).get("score")

    static = {
        "scenic": 10.0 * float(np.clip(preference[land].mean() * 1.6 if land.any() else 0, 0, 1)),
        "variety": 10.0 * float(np.clip(variety, 0, 1)),
        "mystery": 10.0 * float(np.clip(mystery_score * 1.4, 0, 1)),
        "prospect_refuge": 10.0 * float(np.clip(pr[land].mean() * 1.8 if land.any() else 0, 0, 1)),
        "terrain_quality": None if terrain is None else float(terrain),
    }

    def score_group(group: list[AgentLog]) -> tuple[dict, dict]:
        m = _agent_metrics(group, per, reachable, len(world.pois), ev.events.strong_threshold)
        sc = dict(static)
        sc.update(_behaviour_scores(m, lm_cov, ev))
        overall, _ = weighted_overall(sc, ev.weights)
        sc["overall"] = overall
        return sc, m

    scores, pooled = score_group(logs)
    _, eff_weights = weighted_overall({k: v for k, v in scores.items() if k != "overall"}, ev.weights)

    # per profile and per run
    profiles: dict[str, dict] = {}
    for name in dict.fromkeys(l.profile for l in logs):
        group = [l for l in logs if l.profile == name]
        sc, m = score_group(group)
        profiles[name] = {"agents": len(group), "scores": rounded({k: sc[k] for k in
                          ("pacing", "exploration", "orientation", "overall") if sc.get(k) is not None}, 2),
                          "metrics": rounded({k: m[k] for k in ("bored_fraction", "mean_event_gap_s",
                                              "discoveries_per_agent_hour", "lost_fraction", "backtrack_ratio",
                                              "fatigue_fraction")}, 3)}
    run_overall = [score_group([l for l in logs if l.run == r])[0]["overall"] for r in range(sim.get("runs", 1))]
    conf = ev.confidence
    per_agent = {
        "bored_fraction": [l.bored_fraction for l in logs],
        "mean_event_gap_s": [float(np.mean(l.gaps)) for l in logs if l.gaps],
        "discoveries_per_agent_hour": [l.discoveries / (l.duration / 3600.0) for l in logs if l.duration > 0],
        "lost_fraction": [l.lost_fraction for l in logs],
        "backtrack_ratio": [l.backtrack_ratio for l in logs],
    }
    fractions = {"bored_fraction", "lost_fraction", "backtrack_ratio"}
    statistics = {k: rounded(summarize(v, conf, (0.0, 1.0) if k in fractions else (0.0, None)), 4)
                  for k, v in per_agent.items()}
    statistics["overall_per_run"] = rounded(summarize([x for x in run_overall if x is not None], conf,
                                                      (0.0, 10.0)), 4)

    bored_map = sim["boredom"]
    interest = np.clip(preference - 0.5 * (bored_map / max(bored_map.max(), 1e-6)), 0, None)
    world.psych = {
        "perception": per,
        "landmark_visibility": lm_vis,
        "preference": preference,
        "legibility": legib,
        "prospect_refuge": pr,
        "boredom_grid": bored_map,
        "lost_grid": sim["lost"],
        "visits": sim["visits"],
        "agents": logs,
        "interest_grid": interest,
        # full-resolution versions for previews
        "interest": resample(interest, (world.size, world.size)),
        "boredom": resample(bored_map, (world.size, world.size)),
    }
    out_scores = {k: round(float(v), 2) for k, v in scores.items() if v is not None and k != "overall"}
    out_scores["overall"] = round(float(scores["overall"] or 0.0), 3)
    metrics = rounded(pooled, 3)
    metrics["mean_event_gap_s"] = round(pooled["mean_event_gap_s"], 1)
    metrics["p90_event_gap_s"] = round(pooled["p90_event_gap_s"], 1)
    metrics["discoveries_per_agent_hour"] = round(pooled["discoveries_per_agent_hour"], 1)
    metrics.update({
        "landmark_coverage": round(lm_cov, 3),
        "biome_entropy": round(biome_entropy, 3),
        "feature_kinds": feature_kinds,
        "cave_systems": len(world.caves),
        "cave_entrances": cave_mouths,
        "lakes": len(world.lakes),
        "rivers": len(world.rivers),
        "waterfalls": len(world.waterfalls),
        "landmarks": len(landmarks),
        "pois": len(world.pois),
        "agents": len(logs),
        "runs": sim.get("runs", 1),
    })
    report = {
        "scores": out_scores,
        "metrics": metrics,
        "regions": region_table(world),
        "weights": {k: round(v, 4) for k, v in eff_weights.items()},
        "profiles": profiles,
        "statistics": statistics,
        "intensity_per_minute": _intensity_curve(logs, world.cfg.psych.agent_minutes),
    }
    world.evaluation["simulation"] = report
    return report


def region_table(world: World) -> list[dict]:
    """Compact per-region summary (4 x 4 grid, rows A-D south to north, columns 1-4 west to east)."""
    psy = world.psych
    G = psy["perception"].grid
    out = []
    for region in world.all_regions():
        sj, si = world.region_slices(region)
        h = world.height[sj, si]
        water = world.water[sj, si]
        land = ~water
        biomes = world.biome[sj, si][land]
        dom = []
        if biomes.size:
            cnt = np.bincount(biomes.ravel(), minlength=len(BIOME_NAMES))
            for b in np.argsort(-cnt)[:3]:
                if cnt[b] > 0.08 * biomes.size:
                    dom.append(BIOME_NAMES[b])
        gsl = (slice(sj.start * G // world.size, sj.stop * G // world.size),
               slice(si.start * G // world.size, si.stop * G // world.size))
        rx0, ry0 = si.start * world.cell, sj.start * world.cell
        rx1, ry1 = si.stop * world.cell, sj.stop * world.cell

        def inside(x, y):
            return rx0 <= x < rx1 and ry0 <= y < ry1

        out.append({
            "region": region,
            "land_fraction": round(float(land.mean()), 2),
            "elevation_mean": round(float(h[land].mean()) if land.any() else 0.0, 1),
            "relief": round(float(np.ptp(h[land])) if land.any() else 0.0, 1),
            "biomes": dom,
            "interest": round(float(psy["interest_grid"][gsl].mean()), 3),
            "boredom_seconds": round(float(psy["boredom_grid"][gsl].sum()), 1),
            "lost_seconds": round(float(psy["lost_grid"][gsl].sum()), 1) if "lost_grid" in psy else 0.0,
            "visits_seconds": round(float(psy["visits"][gsl].sum()), 1),
            "mystery": round(float(psy["perception"].mystery[gsl].mean()), 3),
            "legibility": round(float(psy["legibility"][gsl].mean()), 3),
            "pois": [p.kind for p in world.pois if inside(p.x, p.y)],
            "landmarks": [l.kind for l in world.landmarks if inside(l.x, l.y)],
            "caves": sum(1 for c in world.caves for e in c.entrances if inside(e["x"], e["y"])),
            "lakes": sum(1 for lk in world.lakes if inside(*lk["center"])),
            "rivers": bool(world.river_mask[sj, si].any()),
            "flat_fraction": round(float((np.abs(np.gradient(h, world.cell)[0]) < 0.08).mean()), 2),
            "dry": bool(np.isin(world.biome[sj, si], [B["desert"], B["salt_flat"], B["shrubland"]]).mean() > 0.4),
        })
    return out
