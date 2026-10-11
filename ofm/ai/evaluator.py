"""Map interest evaluation: simulation metrics + local LLM critique.

The simulation report (psychology scores, agent boredom, per-region table)
is condensed into a prompt for a local model acting as an open-world level
designer. The model returns scores, a critique and a list of actions from the
whitelisted vocabulary. Without a model, or if its answer is unusable, a
deterministic heuristic produces the same structure from the same metrics.

Scope and robustness (``ai.scope``):
  * The prompt states the sandbox (``ScopePolicy.describe``: allowed actions
    with remaining quota, allowed regions, protected zones, parameter ranges,
    change limits) and the reply schema narrows the action ``type``,
    ``region`` and ``kind`` enums to exactly that sandbox.
  * The prompt is capped to ``max_prompt_chars`` (and to what fits in
    ``num_ctx``) by dropping low-priority content first: regions outside the
    scope, then the regions table down to the most bored / least interesting
    regions, then agent rows, then world-summary details (for tiny limits
    finally everything except scores, sandbox and instructions; ``fits``
    reports whether the cap was met). The payload is always re-serialised,
    never cut, so it stays valid JSON.
  * Self-consistency (Wang et al. 2022): with ``samples > 1`` the model is
    asked several times with different seeds; scores are merged by the median
    and only actions proposed by a strict majority of the valid samples are
    kept, ordered by vote count (amounts / values take the median).
  * Every proposed action is validated, then checked against the scope; the
    ones that fail are reported with reasons in ``result["llm"]["rejected"]``.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np

from ..noise import derive_seed
from ..physics.biome import BIOME_NAMES
from ..world import REGION_ROWS, World
from .actions import LANDMARK_KINDS, POI_KINDS, ActionError, action_key, validate_action
from .llm import LocalLLM
from .schema import review_schema
from .scope import ScopePolicy, _compact

SCORE_KEYS = ("exploration", "variety", "landmarks", "pacing", "mystery", "coherence", "overall")

SYSTEM_PROMPT = """You are a senior open-world level designer reviewing a procedurally generated map.
You receive measurements from physical and psychological simulations: Kaplan preference factors,
prospect-refuge, landmark visibility, and play-test agents that walked the map and recorded where they
were bored. Judge how interesting the map is to explore and propose concrete improvements.
Rules:
- Respond with ONE JSON object only, no prose outside it.
- Scores are numbers from 0 to 10.
- You work inside a sandbox described under "scope": use only the actions listed in
  scope.allowed_actions (never more than their "remaining" count), only regions in
  scope.allowed_regions, and only parameters in scope.param_ranges. Protected zones are off-limits.
- Regions are named by a row letter (A = southernmost row) and a column number (1 = westernmost column).
- Prefer targeted fixes where agents were bored, where landmarks are missing, or where variety is low.
- Never repeat an action listed as rejected."""

REGION_KEYS_COMPACT = ("region", "land_fraction", "relief", "biomes", "interest", "boredom_seconds",
                       "visits_seconds", "mystery", "legibility", "pois", "caves", "lakes", "rivers", "dry")


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


def _need(row: dict) -> tuple:
    """Sort key: most bored first, then least interesting."""
    return (-float(row.get("boredom_seconds", 0.0) or 0.0), float(row.get("interest", 0.0) or 0.0))


def _agent_rows(world: World) -> list[dict]:
    return [{"personality": a.personality, "minutes": round(a.duration / 60, 1),
             "bored_minutes": round(a.bored_seconds / 60, 1), "events": len(a.events)}
            for a in world.psych.get("agents", [])]


def _instructions(world: World, policy: ScopePolicy, max_actions: int, language: str) -> str:
    lang = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(language, language)
    n = world.region_divs
    score_fields = ", ".join('"%s": 0-10' % k for k in SCORE_KEYS)
    # the example must itself be in scope: a hard-coded region steers small models out of it
    example = None
    if policy.allowed_regions:
        region = policy.allowed_regions[0]
        if "add_poi" in policy.allowed_actions:
            example = {"type": "add_poi", "region": region, "kind": "ruin", "reason": "..."}
        else:
            t = next((a for a in policy.allowed_actions if a != "set_param"), None)
            if t is not None:
                example = {"type": t, "region": region, "reason": "..."}
    if example is None and "set_param" in policy.allowed_actions and policy.param_ranges:
        path, (lo, hi) = next(iter(policy.param_ranges.items()))
        example = {"type": "set_param", "path": path, "value": round((lo + hi) / 2, 3), "reason": "..."}
    if example is None or max_actions <= 0:
        actions_text = '"actions": [] (no action is available in your scope)}\n'
    else:
        actions_text = ('"actions": [up to ' + str(max_actions) + " objects like "
                        + json.dumps(example, ensure_ascii=False) + "]}\n")
    return ('Return JSON: {"scores": {' + score_fields + '}, '
            + '"critique": "2-4 sentences in ' + lang + '", '
            + actions_text
            + f"Regions: rows {REGION_ROWS[0]}-{REGION_ROWS[n - 1]} south to north, columns 1-{n} west to east. "
            + "Stay inside scope; an action outside it is rejected and wastes your turn.")


def build_prompt_info(world: World, rejected: list[str], max_actions: int, language: str,
                      policy: ScopePolicy | None = None, max_chars: int | None = None) -> tuple[str, dict]:
    """Prompt text plus {"chars", "limit", "trimmed": [steps applied], "fits"}."""
    policy = policy or ScopePolicy(world.cfg.ai.scope, world)
    limit = int(world.cfg.ai.scope.max_prompt_chars if max_chars is None else max_chars)
    sim = world.evaluation.get("simulation", {})
    allowed = set(policy.allowed_regions)
    rows = sorted(sim.get("regions", []), key=lambda r: (r.get("region") not in allowed, _need(r)))
    payload: dict[str, Any] = {
        "scope": policy.describe(),
        "simulation_scores_0_10": sim.get("scores", {}),
        "metrics": sim.get("metrics", {}),
        "regions": rows,
        "agents": _agent_rows(world),
        "world": _world_summary(world),
        "rejected_actions": list(rejected)[-20:],
    }
    tail = _instructions(world, policy, max_actions, language)

    def render() -> str:
        return ("Map report:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
                + "\n\n" + tail)

    def keep_regions(k: int):
        def step():
            payload["regions"] = payload["regions"][:k]
        return step

    def in_scope():
        inside = [r for r in payload["regions"] if r.get("region") in allowed]
        if inside:
            payload["regions"] = inside

    def compact_regions():
        out = []
        for r in payload["regions"]:
            c = {k: r[k] for k in REGION_KEYS_COMPACT if k in r}
            if isinstance(c.get("pois"), list):
                c["pois"] = c["pois"][:4]
            out.append(c)
        payload["regions"] = out

    def agents_summary():
        rows_ = payload.get("agents") or []
        if rows_:
            payload["agents"] = {"count": len(rows_),
                                 "mean_bored_minutes": round(float(np.mean([a["bored_minutes"] for a in rows_])), 1),
                                 "mean_events": round(float(np.mean([a["events"] for a in rows_])), 1)}

    def drop_agents():
        payload.pop("agents", None)

    def world_details():
        w = payload["world"]
        w["landmarks"] = dict(Counter(lm["kind"] for lm in w.get("landmarks", [])))
        w["caves"] = len(w.get("caves", []))
        w["lakes"] = len(w.get("lakes", []))
        w["biome_shares"] = dict(sorted(w.get("biome_shares", {}).items(), key=lambda kv: -kv[1])[:4])

    def metrics_scalars():
        payload["metrics"] = {k: v for k, v in payload["metrics"].items() if isinstance(v, (int, float, str))}

    def rejected_short():
        payload["rejected_actions"] = payload["rejected_actions"][-5:]

    def scope_minimal():
        sc = payload["scope"]
        for z in sc.get("protected_zones", []):
            z.pop("reason", None)
        sc["allowed_actions"] = {t: a.get("remaining", 0) for t, a in sc.get("allowed_actions", {}).items()}
        if sc.get("allowed_regions") == policy.all_regions:
            sc["allowed_regions"] = "all"
        for k in ("protected_regions", "protected_zones"):
            if not sc.get(k):
                sc.pop(k, None)

    def metrics_core():
        keep = ("bored_fraction", "mean_event_gap_s", "coverage_seen", "cave_entrances", "lakes", "landmarks", "pois")
        payload["metrics"] = {k: v for k, v in payload["metrics"].items() if k in keep}

    def world_minimal():
        w = payload["world"]
        payload["world"] = {k: w[k] for k in ("extent_m", "shape", "land_fraction") if k in w}

    def drop_rejected():
        payload.pop("rejected_actions", None)

    def drop(key: str):
        def step():
            payload.pop(key, None)
        return step

    def param_paths():
        # values are clamped into range anyway; the path list keeps set_param usable
        sc = payload["scope"]
        if isinstance(sc.get("param_ranges"), dict):
            sc["param_ranges"] = list(sc["param_ranges"])

    steps = [("regions_in_scope", in_scope), ("regions_compact", compact_regions),
             ("regions_top8", keep_regions(8)), ("regions_top4", keep_regions(4)),
             ("agents_summary", agents_summary), ("agents_dropped", drop_agents),
             ("world_details", world_details), ("metrics_scalars", metrics_scalars),
             ("regions_top2", keep_regions(2)), ("rejected_last5", rejected_short),
             ("scope_minimal", scope_minimal), ("world_minimal", world_minimal), ("metrics_core", metrics_core),
             ("regions_top1", keep_regions(1)), ("rejected_dropped", drop_rejected),
             # last resort for tiny limits: only the scores and the sandbox remain
             ("world_dropped", drop("world")), ("metrics_dropped", drop("metrics")),
             ("regions_dropped", drop("regions")), ("param_ranges_paths", param_paths)]
    text = render()
    trimmed: list[str] = []
    for name, step in steps:
        if len(text) <= limit:
            break
        step()
        trimmed.append(name)
        text = render()
    return text, {"chars": len(text), "limit": limit, "trimmed": trimmed, "fits": len(text) <= limit}


def build_prompt(world: World, rejected: list[str], max_actions: int, language: str,
                 policy: ScopePolicy | None = None, max_chars: int | None = None) -> str:
    return build_prompt_info(world, rejected, max_actions, language, policy, max_chars)[0]


def sim_to_scores(sim_scores: dict) -> dict:
    overall = float(sim_scores.get("overall", 0.0))
    src = {"exploration": "exploration", "variety": "variety", "landmarks": "legibility", "pacing": "pacing",
           "mystery": "mystery", "coherence": "scenic", "overall": "overall"}
    return {k: float(sim_scores.get(v, overall)) for k, v in src.items()}


def heuristic_review(world: World, rejected: list[str], max_actions: int, language: str = "ko",
                     policy: ScopePolicy | None = None) -> dict:
    """Deterministic reviewer. With a ``policy`` it only proposes in-scope actions within quota."""
    sim = world.evaluation["simulation"]
    sc, mt, regions = sim.get("scores", {}), sim.get("metrics", {}), sim.get("regions", [])
    actions: list[dict] = []
    rej = set(rejected)
    planned: Counter = Counter()

    def push(a: dict) -> None:
        if len(actions) >= max_actions or action_key(a) in rej or \
                any(action_key(a) == action_key(b) for b in actions):
            return
        if policy is not None:
            ok, _ = policy.check(a, planned)
            if not ok:
                return
            policy.plan(planned, a)
        actions.append(a)

    used_kinds = [p.kind for p in world.pois]

    def varied(preferred: list[str]) -> str:
        # first preferred kind that is not already common on the map
        for k in preferred:
            if used_kinds.count(k) + sum(a.get("kind") == k for a in actions) < 2:
                return k
        return preferred[0]

    allowed = set(policy.allowed_regions) if policy is not None else None
    land_regions = [r for r in regions if r.get("land_fraction", 0.0) > 0.35
                    and (allowed is None or r.get("region") in allowed)]
    overall = float(sc.get("overall", 0.0))
    if not land_regions:
        return {"scores": sim_to_scores(sc), "critique": "", "actions": [], "source": "heuristic"}
    by_bored = sorted(land_regions, key=lambda r: -r.get("boredom_seconds", 0.0))
    for r in by_bored[:2]:
        if r.get("boredom_seconds", 0.0) <= 30:
            continue
        pois = r.get("pois", [])
        if r.get("dry"):
            kind = varied(["ruin", "outpost", "treasure", "shrine"])
        elif r.get("relief", 0.0) > 250:
            kind = varied(["watchtower", "hermitage", "camp", "ruin"])
        elif r.get("rivers") and "settlement" not in pois and "village" not in pois:
            kind = varied(["village", "bridge", "camp"])
        elif r.get("mystery", 0.0) > 0.3:
            kind = varied(["shrine", "treasure", "ruin"])
        else:
            kind = varied(["camp", "ruin", "outpost"])
        push({"type": "add_poi", "region": r["region"], "kind": kind,
              "reason": f"agents were bored for {r.get('boredom_seconds', 0.0):.0f}s here"})
    if sc.get("legibility", overall) < 7.5:
        r = min(land_regions, key=lambda r: (r.get("legibility", 0.0), -r.get("land_fraction", 0.0)))
        coastal = r.get("land_fraction", 1.0) < 0.75
        kind = "sea_stack" if coastal and world.ocean.any() else ("spire" if r.get("dry") else
                                                                   ("arch" if r.get("relief", 0.0) < 200 else "monolith"))
        push({"type": "add_landmark", "region": r["region"], "kind": kind,
              "reason": "few landmarks visible here; players lose orientation"})
    if sc.get("mystery", overall) < 7.0 or mt.get("cave_entrances", 0) < 5:
        cands = [r for r in land_regions if r.get("caves", 0) == 0]
        if cands:
            r = max(cands, key=lambda r: r.get("relief", 0.0))
            push({"type": "add_cave", "region": r["region"], "reason": "no hidden places in this region"})
    if mt.get("lakes", 0) < 2:
        wet = [r for r in land_regions if not r.get("dry") and r.get("lakes", 0) == 0]
        if wet:
            r = max(wet, key=lambda r: r.get("flat_fraction", 0.0))
            push({"type": "add_lake", "region": r["region"], "reason": "water bodies add variety and rest points"})
    if sc.get("exploration", overall) < 6.5:
        r = min(land_regions, key=lambda r: r.get("visits_seconds", 0.0))
        push({"type": "add_poi", "region": r["region"],
              "kind": varied(["outpost", "ruin", "shrine"] if r.get("relief", 0.0) > 150 else ["ruin", "camp", "treasure"]),
              "reason": "unexplored region needs a lure"})
    flat_dull = [r for r in land_regions if r.get("relief", 0.0) < 80 and r.get("interest", 1.0) < 0.35]
    if flat_dull:
        push({"type": "raise_relief", "region": flat_dull[0]["region"], "amount": 0.35,
              "reason": "flat, low-interest terrain"})

    weakest = sorted(((k, v) for k, v in sc.items() if k != "overall"), key=lambda kv: kv[1])[:2]
    names_ko = {"scenic": "경관", "variety": "다양성", "legibility": "랜드마크 가독성", "mystery": "미스터리",
                "pacing": "페이싱", "exploration": "탐험 범위", "prospect_refuge": "조망-피신"}
    bored = float(mt.get("bored_fraction") or 0.0) * 100
    gap = float(mt.get("mean_event_gap_s") or 0.0)
    if language == "ko":
        critique = (f"종합 {overall:.1f}/10. 약점은 " + ", ".join(f"{names_ko.get(k, k)}({v:.1f})" for k, v in weakest)
                    + f"입니다. 플레이테스트 에이전트는 시간의 {bored:.0f}%를 지루하게 보냈고 "
                    f"평균 이벤트 간격은 {gap:.0f}초입니다.")
    else:
        critique = (f"Overall {overall:.1f}/10. Weakest: " + ", ".join(f"{k} ({v:.1f})" for k, v in weakest)
                    + f". Agents were bored {bored:.0f}% of the time; mean gap between "
                    f"events {gap:.0f}s.")
    return {"scores": sim_to_scores(sc), "critique": critique, "actions": actions, "source": "heuristic"}


def _clean_scores(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    out = {}
    for k in SCORE_KEYS:
        try:
            v = float(raw[k])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(v):
            return None
        out[k] = float(np.clip(v, 0.0, 10.0))
    return out


def _parse_sample(raw: dict, regions: list[str], policy: ScopePolicy, rejected: list[str]) -> dict:
    """Validate one model reply: scores, unique in-scope actions, and why the others were dropped."""
    actions, invalid, refused, seen = [], [], [], set()
    items = raw.get("actions") if isinstance(raw.get("actions"), list) else []
    for a in items:
        try:
            va = validate_action(a, regions, policy.param_ranges)
            ok, reason = policy.check(va)
        except ActionError as exc:
            invalid.append(str(exc))
            continue
        except Exception as exc:        # model data must never escape the reviewer as an exception
            invalid.append(f"unusable action ({type(exc).__name__})")
            continue
        if not ok:
            refused.append({"action": _compact(va), "reason": reason})
            continue
        key = action_key(va)
        if key in rejected:
            refused.append({"action": _compact(va), "reason": "already rejected in an earlier iteration"})
            continue
        if key not in seen:
            seen.add(key)
            actions.append(va)
    return {"scores": _clean_scores(raw.get("scores")), "actions": actions, "invalid": invalid,
            "rejected": refused, "critique": str(raw.get("critique", "") or "")[:1200]}


def merge_samples(parsed: list[dict], policy: ScopePolicy, max_actions: int) -> dict:
    """Self-consistency merge: median scores, majority-voted actions ordered by votes."""
    n = len(parsed)
    score_sets = [p["scores"] for p in parsed if p["scores"]]
    scores = ({k: round(float(np.median([s[k] for s in score_sets])), 3) for k in SCORE_KEYS}
              if score_sets else None)
    votes: Counter = Counter()
    first: dict[str, tuple[int, int]] = {}
    inst: dict[str, list[dict]] = {}
    for i, p in enumerate(parsed):
        for j, a in enumerate(p["actions"]):
            k = action_key(a)
            votes[k] += 1
            first.setdefault(k, (i, j))
            inst.setdefault(k, []).append(a)
    majority = sorted((k for k in votes if 2 * votes[k] > n), key=lambda k: (-votes[k], first[k]))
    refused = []
    seen_refusals = set()
    for p in parsed:
        for r in p["rejected"]:
            key = (json.dumps(r["action"], sort_keys=True), r["reason"])
            if key not in seen_refusals:
                seen_refusals.add(key)
                refused.append(r)
    planned: Counter = Counter()
    merged = []
    for k in majority:
        a = dict(inst[k][0])
        for f in ("amount", "value"):
            vals = [x[f] for x in inst[k] if f in x]
            if len(vals) > 1:
                a[f] = float(np.median(vals))
        if len(merged) >= max_actions:
            refused.append({"action": _compact(a), "reason": f"more than {max_actions} actions this iteration"})
            continue
        ok, reason = policy.check(a, planned)
        if not ok:
            refused.append({"action": _compact(a), "reason": reason})
            continue
        policy.plan(planned, a)
        merged.append(a)
    return {
        "scores": scores,
        "critique": next((p["critique"] for p in parsed if p["critique"]), ""),
        "actions": merged,
        "invalid_actions": [e for p in parsed for e in p["invalid"]],
        "rejected": refused,
        "votes": {k: votes[k] for k in sorted(votes, key=lambda k: (-votes[k], first[k]))},
        "minority": [k for k in votes if 2 * votes[k] <= n],
    }


def _llm_review(world: World, llm: LocalLLM, rejected: list[str], policy: ScopePolicy,
                max_actions: int) -> dict | None:
    ai = world.cfg.ai
    regions = world.all_regions()
    # only actions that could pass the policy: region actions need an allowed region,
    # set_param needs a configured range
    types = [a for a in policy.allowed_actions if policy.remaining(a) > 0
             and (policy.param_ranges if a == "set_param" else policy.allowed_regions)]
    schema = review_schema(SCORE_KEYS, types, policy.allowed_regions, LANDMARK_KINDS, POI_KINDS,
                           list(policy.param_ranges), max_actions if types else 0)
    prompt, pinfo = build_prompt_info(world, rejected, max_actions, ai.language, policy,
                                      llm.prompt_char_budget(SYSTEM_PROMPT))
    images = None
    if ai.vision:
        from ..preview import png_bytes, render_map
        images = [png_bytes(render_map(world, "biome"))]
    def check(obj: dict) -> list[str]:
        """Malformed actions the schema cannot express (missing region, path or value)."""
        errs = []
        items = obj.get("actions") if isinstance(obj.get("actions"), list) else []
        for i, a in enumerate(items):
            try:
                validate_action(a, regions, policy.param_ranges)
            except ActionError as exc:
                errs.append(f"$.actions[{i}]: {exc}")
            except Exception as exc:
                errs.append(f"$.actions[{i}]: unusable ({type(exc).__name__})")
        return errs

    calls0, repairs0 = llm.calls, llm.repairs
    # LLM-interaction knobs (samples, retries, budget, prompt size) come from the client's
    # scope, like every other request setting; the sandbox itself comes from ``policy``.
    samples = max(1, int(llm.scope.samples))
    parsed, schema_errors, formats = [], [], []
    for k in range(samples):
        seed = derive_seed(world.cfg.world.seed, "llm-review", llm.calls, k)
        raw = llm.chat_json(SYSTEM_PROMPT, prompt, images, schema=schema, seed=seed, name="ofm_review",
                            check=check)
        info = llm.last_info
        if info.get("errors"):
            schema_errors.append(list(info["errors"][:6]))
        if info.get("format"):
            formats.append(info["format"])
        if raw is None:
            break                      # budget spent or server failing: keep what we have
        parsed.append(_parse_sample(raw, regions, policy, rejected))
    if not parsed:
        return None
    out = merge_samples(parsed, policy, max_actions)
    out.update(samples=len(parsed), samples_requested=samples, calls=llm.calls - calls0,
               repairs=llm.repairs - repairs0, schema_errors=schema_errors,
               format=formats[-1] if formats else None, prompt_chars=pinfo["chars"],
               prompt_limit=pinfo["limit"], prompt_trimmed=pinfo["trimmed"])
    return out


def review(world: World, llm: LocalLLM | None, rejected: list[str] | None = None,
           policy: ScopePolicy | None = None) -> dict:
    """Evaluate the current world. Returns scores/critique/actions plus provenance."""
    ai = world.cfg.ai
    rejected = list(rejected or [])
    policy = policy or ScopePolicy(ai.scope, world)
    regions = world.all_regions()
    max_actions = policy.iteration_capacity()
    heur = heuristic_review(world, rejected, max_actions, ai.language, policy=policy)
    heur["actions"] = [validate_action(a, regions, policy.param_ranges) for a in heur["actions"]]
    result = dict(heur)
    result["sim_scores"] = world.evaluation["simulation"]["scores"]
    result["llm"] = None
    if llm is not None and llm.available:
        if llm.budget_exhausted:
            result["llm_fallback"] = "LLM budget exhausted: " + llm.budget_status()["reason"]
        else:
            part = _llm_review(world, llm, rejected, policy, max_actions)
            if part is None:
                why = "LLM budget exhausted: " + llm.exhausted_reason if llm.budget_exhausted else \
                    (llm.last_error or "no usable reply")
                result["llm_fallback"] = why
            else:
                result["llm"] = part
                if part["actions"]:
                    result["actions"] = part["actions"]
                    result["source"] = "llm"
                if part["critique"]:
                    result["critique"] = part["critique"]
        result["llm_budget"] = llm.budget_status()
    result["objective"] = objective(result, ai.llm_score_weight)
    return result


def objective(result: dict, llm_weight: float) -> float:
    sim = float(result["sim_scores"]["overall"])
    llm = result.get("llm") or {}
    if llm.get("scores"):
        return round((1 - llm_weight) * sim + llm_weight * llm["scores"]["overall"], 3)
    return round(sim, 3)


__all__ = ["review", "heuristic_review", "build_prompt", "build_prompt_info", "merge_samples", "SYSTEM_PROMPT",
           "SCORE_KEYS", "LANDMARK_KINDS", "POI_KINDS"]
