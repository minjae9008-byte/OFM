"""Combines perception, landmarks and agent play-tests into an interest model.

Per-location *preference* follows Kaplan & Kaplan's matrix (coherence,
complexity, legibility, mystery) plus Appleton's prospect-refuge. The global
score adds variety, pacing and exploration measured by the agents. Every
score is on a 0-10 scale; ``overall`` is the objective the AI director
optimises.
"""
from __future__ import annotations

import numpy as np

from ..grid import resample
from ..physics.biome import BIOME_NAMES, B
from ..world import World
from .agents import simulate_agents
from .landmarks import select_landmarks
from .perception import compute_perception

WEIGHTS = {
    "scenic": 0.16, "variety": 0.14, "legibility": 0.14, "mystery": 0.14,
    "pacing": 0.18, "exploration": 0.12, "prospect_refuge": 0.12,
}


def evaluate_psychology(world: World) -> dict:
    per = compute_perception(world)
    landmarks, lm_vis = select_landmarks(world, per)
    world.landmarks = landmarks
    G = per.grid
    land = per.land
    legib = np.clip(lm_vis.sum(axis=1) / 3.0, 0, 1).reshape(G, G) if lm_vis.size else np.zeros((G, G))
    pr = np.sqrt(np.clip(per.prospect, 0, 1) * np.clip(per.refuge + 0.15, 0, 1))
    preference = (0.24 * per.coherence + 0.18 * per.complexity + 0.2 * legib + 0.26 * per.mystery
                  + 0.12 * pr)
    sim = simulate_agents(world, per, landmarks, lm_vis)

    logs = sim["agents"]
    total = sum(l.duration for l in logs) or 1.0
    bored = sum(l.bored_seconds for l in logs) / total
    gaps = np.concatenate([np.array(l.gaps) for l in logs if l.gaps]) if any(l.gaps for l in logs) else np.array([total])
    mean_gap = float(gaps.mean())
    p90_gap = float(np.quantile(gaps, 0.9))
    events_per_min = sum(len([e for e in l.events if e[2] >= 0.35]) for l in logs) / (total / 60.0)
    reachable = land & ~sim["blocked"]
    seen_any = np.zeros(G * G, dtype=bool)
    for l in logs:
        for c in set(l.path):
            seen_any |= per.visible[c]
    coverage_seen = float(seen_any.reshape(G, G)[reachable].mean()) if reachable.any() else 0.0
    poi_found = set()
    for l in logs:
        poi_found |= {e[3] for e in l.events if e[1] in ("poi_discovered", "poi_reached")}
    poi_frac = len(poi_found) / max(1, len(world.pois))
    # discoveries per agent-hour against a target cadence (content density, not a ratio that
    # would punish adding content)
    discoveries_per_hour = len(poi_found) / max(1e-6, total / 3600.0) * len(logs)
    discovery_rate = min(1.0, discoveries_per_hour / (len(logs) * 30.0))

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
    pacing = (1.0 - bored) * (1.0 - 0.5 * np.clip((mean_gap - 40.0) / 120.0, 0, 1))
    exploration = 0.6 * coverage_seen + 0.4 * discovery_rate
    scores01 = {
        "scenic": float(np.clip(preference[land].mean() * 1.6 if land.any() else 0, 0, 1)),
        "variety": float(np.clip(variety, 0, 1)),
        "legibility": float(np.clip(lm_cov, 0, 1)),
        "mystery": float(np.clip(mystery_score * 1.4, 0, 1)),
        "pacing": float(np.clip(pacing, 0, 1)),
        "exploration": float(np.clip(exploration, 0, 1)),
        "prospect_refuge": float(np.clip(pr[land].mean() * 1.8 if land.any() else 0, 0, 1)),
    }
    scores = {k: round(10.0 * v, 2) for k, v in scores01.items()}
    scores["overall"] = round(sum(scores[k] * w for k, w in WEIGHTS.items()) / sum(WEIGHTS.values()), 3)

    bored_map = sim["boredom"]
    interest = np.clip(preference - 0.5 * (bored_map / max(bored_map.max(), 1e-6)), 0, None)
    world.psych = {
        "perception": per,
        "landmark_visibility": lm_vis,
        "preference": preference,
        "legibility": legib,
        "prospect_refuge": pr,
        "boredom_grid": bored_map,
        "visits": sim["visits"],
        "agents": logs,
        "interest_grid": interest,
        # full-resolution versions for previews
        "interest": resample(interest, (world.size, world.size)),
        "boredom": resample(bored_map, (world.size, world.size)),
    }
    metrics = {
        "bored_fraction": round(bored, 3),
        "mean_event_gap_s": round(mean_gap, 1),
        "p90_event_gap_s": round(p90_gap, 1),
        "events_per_minute": round(events_per_min, 2),
        "coverage_seen": round(coverage_seen, 3),
        "poi_discovery": round(poi_frac, 3),
        "discoveries_per_agent_hour": round(discoveries_per_hour / max(1, len(logs)), 1),
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
    }
    report = {"scores": scores, "metrics": metrics, "regions": region_table(world)}
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
