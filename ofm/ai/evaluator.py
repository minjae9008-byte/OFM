"""Map interest evaluation: simulation metrics + local LLM critique.

The simulation report (psychology scores, agent boredom, per-region table)
is condensed into a prompt for a local model acting as an open-world level
designer. The model returns scores, a critique and a list of actions from the
whitelisted vocabulary. Without a model, or if its answer is unusable, a
deterministic heuristic produces the same structure from the same metrics.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..physics.biome import BIOME_NAMES
from ..world import World
from .actions import ACTION_DOC, LANDMARK_KINDS, POI_KINDS, ActionError, action_key, validate_action
from .llm import LocalLLM

SCORE_KEYS = ("exploration", "variety", "landmarks", "pacing", "mystery", "coherence", "overall")

SYSTEM_PROMPT = """You are a senior open-world level designer reviewing a procedurally generated map.
You receive measurements from physical and psychological simulations: Kaplan preference factors,
prospect-refuge, landmark visibility, and play-test agents that walked the map and recorded where they
were bored. Judge how interesting the map is to explore and propose concrete improvements.
Rules:
- Respond with ONE JSON object only, no prose outside it.
- Scores are numbers from 0 to 10.
- Only use actions from the allowed list with the listed parameters. Regions are named A1..D4
  (rows A=south .. D=north, columns 1=west .. 4=east).
- Prefer targeted fixes where agents were bored, where landmarks are missing, or where variety is low.
- Never repeat an action listed as rejected."""


def _world_summary(world: World) -> dict:
    land = ~world.water
    counts = np.bincount(world.biome[land].ravel(), minlength=len(BIOME_NAMES)) if land.any() else np.zeros(1)
    total = max(1, int(counts.sum()))
    shares = {BIOME_NAMES[i]: round(float(c) / total, 3) for i, c in enumerate(counts) if c / total > 0.02}
    return {
        "extent_m": world.extent,
        "shape": world.cfg.world.shape,
        "land_fraction": round(float(land.mean()), 3),
        "elevation_range_m": [round(float(world.height.min()), 1), round(float(world.height.max()), 1)],
        "biome_shares": shares,
        "landmarks": [{"kind": l.kind, "region": world.region_of(l.x, l.y), "salience": round(l.salience, 2)}
                      for l in world.landmarks],
        "poi_kinds": sorted({p.kind for p in world.pois}),
        "caves": [{"kind": c.kind, "length_m": round(c.length), "entrances": len(c.entrances)} for c in world.caves],
        "lakes": [{"closed_basin": lk["closed_basin"], "area_m2": round(lk["area_m2"])} for lk in world.lakes],
    }


def build_prompt(world: World, rejected: list[str], max_actions: int, language: str) -> str:
    sim = world.evaluation["simulation"]
    agents = world.psych.get("agents", [])
    agent_rows = [{"personality": a.personality, "minutes": round(a.duration / 60, 1),
                   "bored_minutes": round(a.bored_seconds / 60, 1),
                   "events": len(a.events)} for a in agents]
    payload = {
        "world": _world_summary(world),
        "simulation_scores_0_10": sim["scores"],
        "metrics": sim["metrics"],
        "regions": sim["regions"],
        "agents": agent_rows,
        "allowed_actions": ACTION_DOC,
        "rejected_actions": rejected,
    }
    lang = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(language, language)
    score_fields = ", ".join('"%s": 0-10' % k for k in SCORE_KEYS)
    example = '{"type": "add_poi", "region": "B3", "kind": "ruin", "reason": "..."}'
    return (
        "Map report:\n" + json.dumps(payload, ensure_ascii=False) + "\n\n"
        + 'Return JSON: {"scores": {' + score_fields + '}, '
        + '"critique": "2-4 sentences in ' + lang + '", '
        + '"actions": [up to ' + str(max_actions) + " objects like " + example + "]}"
    )


def sim_to_scores(sim_scores: dict) -> dict:
    return {
        "exploration": sim_scores["exploration"],
        "variety": sim_scores["variety"],
        "landmarks": sim_scores["legibility"],
        "pacing": sim_scores["pacing"],
        "mystery": sim_scores["mystery"],
        "coherence": sim_scores["scenic"],
        "overall": sim_scores["overall"],
    }


def heuristic_review(world: World, rejected: list[str], max_actions: int, language: str = "ko") -> dict:
    sim = world.evaluation["simulation"]
    sc, mt, regions = sim["scores"], sim["metrics"], sim["regions"]
    actions: list[dict] = []
    rej = set(rejected)

    def push(a: dict) -> None:
        if len(actions) < max_actions and action_key(a) not in rej and \
                all(action_key(a) != action_key(b) for b in actions):
            actions.append(a)

    used_kinds = [p.kind for p in world.pois]

    def varied(preferred: list[str]) -> str:
        # first preferred kind that is not already common on the map
        for k in preferred:
            if used_kinds.count(k) + sum(a.get("kind") == k for a in actions) < 2:
                return k
        return preferred[0]

    land_regions = [r for r in regions if r["land_fraction"] > 0.35]
    if not land_regions:
        return {"scores": sim_to_scores(sc), "critique": "", "actions": [], "source": "heuristic"}
    by_bored = sorted(land_regions, key=lambda r: -r["boredom_seconds"])
    for r in by_bored[:2]:
        if r["boredom_seconds"] <= 30:
            continue
        if r["dry"]:
            kind = varied(["ruin", "outpost", "treasure", "shrine"])
        elif r["relief"] > 250:
            kind = varied(["watchtower", "hermitage", "camp", "ruin"])
        elif r["rivers"] and "settlement" not in r["pois"] and "village" not in r["pois"]:
            kind = varied(["village", "bridge", "camp"])
        elif r["mystery"] > 0.3:
            kind = varied(["shrine", "treasure", "ruin"])
        else:
            kind = varied(["camp", "ruin", "outpost"])
        push({"type": "add_poi", "region": r["region"], "kind": kind,
              "reason": f"agents were bored for {r['boredom_seconds']:.0f}s here"})
    if sc["legibility"] < 7.5:
        r = min(land_regions, key=lambda r: (r["legibility"], -r["land_fraction"]))
        coastal = r["land_fraction"] < 0.75
        kind = "sea_stack" if coastal and world.ocean.any() else ("spire" if r["dry"] else
                                                                   ("arch" if r["relief"] < 200 else "monolith"))
        push({"type": "add_landmark", "region": r["region"], "kind": kind,
              "reason": "few landmarks visible here; players lose orientation"})
    if sc["mystery"] < 7.0 or mt["cave_entrances"] < 5:
        cands = [r for r in land_regions if r["caves"] == 0]
        if cands:
            r = max(cands, key=lambda r: r["relief"])
            push({"type": "add_cave", "region": r["region"], "reason": "no hidden places in this region"})
    if mt["lakes"] < 2:
        wet = [r for r in land_regions if not r["dry"] and r["lakes"] == 0]
        if wet:
            r = max(wet, key=lambda r: r["flat_fraction"])
            push({"type": "add_lake", "region": r["region"], "reason": "water bodies add variety and rest points"})
    if sc["exploration"] < 6.5:
        r = min(land_regions, key=lambda r: r["visits_seconds"])
        push({"type": "add_poi", "region": r["region"],
              "kind": varied(["outpost", "ruin", "shrine"] if r["relief"] > 150 else ["ruin", "camp", "treasure"]),
              "reason": "unexplored region needs a lure"})
    flat_dull = [r for r in land_regions if r["relief"] < 80 and r["interest"] < 0.35]
    if flat_dull:
        push({"type": "raise_relief", "region": flat_dull[0]["region"], "amount": 0.35,
              "reason": "flat, low-interest terrain"})

    weakest = sorted(((k, v) for k, v in sc.items() if k != "overall"), key=lambda kv: kv[1])[:2]
    names_ko = {"scenic": "경관", "variety": "다양성", "legibility": "랜드마크 가독성", "mystery": "미스터리",
                "pacing": "페이싱", "exploration": "탐험 범위", "prospect_refuge": "조망-피신"}
    if language == "ko":
        critique = (f"종합 {sc['overall']:.1f}/10. 약점은 " + ", ".join(f"{names_ko.get(k, k)}({v:.1f})" for k, v in weakest)
                    + f"입니다. 플레이테스트 에이전트는 시간의 {mt['bored_fraction'] * 100:.0f}%를 지루하게 보냈고 "
                    f"평균 이벤트 간격은 {mt['mean_event_gap_s']:.0f}초입니다.")
    else:
        critique = (f"Overall {sc['overall']:.1f}/10. Weakest: " + ", ".join(f"{k} ({v:.1f})" for k, v in weakest)
                    + f". Agents were bored {mt['bored_fraction'] * 100:.0f}% of the time; mean gap between "
                    f"events {mt['mean_event_gap_s']:.0f}s.")
    return {"scores": sim_to_scores(sc), "critique": critique, "actions": actions, "source": "heuristic"}


def _clean_scores(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    out = {}
    for k in SCORE_KEYS:
        try:
            out[k] = float(np.clip(float(raw[k]), 0.0, 10.0))
        except (KeyError, TypeError, ValueError):
            return None
    return out


def review(world: World, llm: LocalLLM | None, rejected: list[str] | None = None) -> dict:
    """Evaluate the current world. Returns scores/critique/actions plus provenance."""
    ai = world.cfg.ai
    rejected = rejected or []
    regions = world.all_regions()
    heur = heuristic_review(world, rejected, ai.max_actions, ai.language)
    heur["actions"] = [validate_action(a, regions) for a in heur["actions"]]
    result = dict(heur)
    result["sim_scores"] = world.evaluation["simulation"]["scores"]
    result["llm"] = None
    if llm is not None and llm.available:
        images = None
        if ai.vision:
            from ..preview import png_bytes, render_map
            images = [png_bytes(render_map(world, "biome"))]
        raw = llm.chat_json(SYSTEM_PROMPT, build_prompt(world, rejected, ai.max_actions, ai.language), images)
        if raw is not None:
            scores = _clean_scores(raw.get("scores"))
            actions, errors = [], []
            for a in raw.get("actions", []) if isinstance(raw.get("actions"), list) else []:
                try:
                    va = validate_action(a, regions)
                except ActionError as exc:
                    errors.append(str(exc))
                    continue
                if action_key(va) not in rejected:
                    actions.append(va)
            result["llm"] = {
                "scores": scores, "critique": str(raw.get("critique", ""))[:1200],
                "actions": actions, "invalid_actions": errors,
            }
            if actions:
                result["actions"] = actions[: ai.max_actions]
                result["source"] = "llm"
            if result["llm"]["critique"]:
                result["critique"] = result["llm"]["critique"]
    result["objective"] = objective(result, ai.llm_score_weight)
    return result


def objective(result: dict, llm_weight: float) -> float:
    sim = float(result["sim_scores"]["overall"])
    llm = result.get("llm") or {}
    if llm.get("scores"):
        return round((1 - llm_weight) * sim + llm_weight * llm["scores"]["overall"], 3)
    return round(sim, 3)


__all__ = ["review", "heuristic_review", "build_prompt", "SYSTEM_PROMPT", "LANDMARK_KINDS", "POI_KINDS"]
