"""AI director: an evaluate -> edit -> rebuild -> re-evaluate improvement loop.

Each iteration the reviewer (local LLM, or the heuristic fallback) proposes
whitelisted actions. They pass the scope policy (``ai.scope``, see
``scope.py``) and are applied one by one to a copy of the world; after each
action the change of ``height_base`` is verified and a violating action is
reverted on its own (snapshot / restore), so one oversized edit does not
spoil the rest. Every downstream simulation is then re-run, and the copy is
kept only if the objective (simulation score, optionally blended with the
model's own score) rises by more than ``scope.min_objective_gain``. The rebuild
restores protected content and discards the candidate if protected terrain,
water, caves or the spawn settlement changed anyway (``protect_rebuild``). Rejected
actions are fed back so the next proposal differs.

The loop stops when no allowed action remains (quotas spent), when the
reviewer proposes nothing, or after ``ai.iterations``. When the LLM budget
runs out the heuristic reviewer takes over and the switch is logged in the
history. With ``scope.dry_run`` the first proposal is only checked and logged
(status ``dry_run``) and the world is returned unchanged. Every decision is
stored in ``world.evaluation["ai_audit"]`` (stage ``apply``), together with the
model proposals the reviewer already refused as out of scope or malformed
(stage ``review``); a run summary (policy, quota use, LLM budget) goes to
``world.evaluation["ai_scope"]``.
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from ..noise import derive_seed
from ..pipeline import rebuild
from ..world import World
from .actions import ActionError, action_key, apply_action
from .evaluator import review
from .llm import LocalLLM
from .scope import ScopePolicy, ScopeViolation


def clone_world(world: World) -> World:
    psych = world.psych
    world.psych = {}
    try:
        c = copy.deepcopy(world)
    finally:
        world.psych = psych
    return c


def _snapshot(world: World, action: dict) -> dict:
    """Everything ``apply_action`` may change, cheap enough to take before every action."""
    snap = {"height_base": world.height_base.copy(), "pois": list(world.pois), "features": list(world.features),
            "forced_caves": list(world.forced_caves), "edits": len(world.edits)}
    if action.get("type") == "set_param":
        snap["cfg"] = copy.deepcopy(world.cfg)
    return snap


def _restore(world: World, snap: dict) -> None:
    world.height_base = snap["height_base"]
    world.pois = snap["pois"]
    world.features = snap["features"]
    world.forced_caves = snap["forced_caves"]
    del world.edits[snap["edits"]:]
    if "cfg" in snap:
        world.cfg = snap["cfg"]


def _round(a: dict) -> dict:
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items()}


def _compare(before: dict, after: dict) -> tuple[float, float]:
    """Objectives on a common footing: if either review lacks LLM scores (budget ran out,
    bad reply), compare the simulation scores alone."""
    if bool((before.get("llm") or {}).get("scores")) != bool((after.get("llm") or {}).get("scores")):
        return float(before["sim_scores"]["overall"]), float(after["sim_scores"]["overall"])
    return float(before["objective"]), float(after["objective"])


def run_director(world: World, llm: LocalLLM | None, iterations: int | None = None,
                 progress: Callable[[str], None] | None = None) -> World:
    ai = world.cfg.ai
    scope = ai.scope
    iterations = ai.iterations if iterations is None else iterations
    policy = ScopePolicy(scope, world, ai.max_actions)
    rejected: list[str] = []
    fallback_logged = False

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    def note_fallback(entry: dict, rev: dict) -> None:
        nonlocal fallback_logged
        why = rev.get("llm_fallback")
        if why:
            entry["llm_fallback"] = why
            if not fallback_logged and "budget" in why:
                fallback_logged = True
                say(f"director: {why}; continuing with the heuristic reviewer")

    def log_refusals(rev: dict, iteration: int) -> None:
        """Model proposals refused inside the reviewer (out of scope, malformed) go into the audit too."""
        info = rev.get("llm") or {}
        for r in info.get("rejected", []):
            policy.record(r.get("action") or {}, "rejected", r.get("reason", ""), iteration=iteration, stage="review")
        for err in info.get("invalid_actions", []):
            policy.record({}, "rejected", f"invalid proposal: {err}", iteration=iteration, stage="review")

    current = review(world, llm, rejected, policy)
    log_refusals(current, 0)
    entry0 = {"iteration": 0, "accepted": True, "objective": current["objective"],
              "sim_scores": current["sim_scores"], "source": current["source"],
              "critique": current["critique"], "actions": []}
    note_fallback(entry0, current)
    world.history.append(entry0)
    for it in range(1, iterations + 1):
        if not policy.has_capacity():
            say(f"director: scope quota spent after {it - 1} iteration(s)")
            world.history[-1]["stop_reason"] = "no allowed action remains"
            break
        actions = current["actions"]
        if not actions:
            say(f"director: no further actions after {it - 1} iteration(s)")
            break
        policy.begin_iteration(it)
        cand = world if scope.dry_run else clone_world(world)
        if not scope.dry_run:
            cand.psych = world.psych     # read-only until rebuild() replaces it; lets sites use psych fields
        rng = np.random.default_rng(derive_seed(world.cfg.world.seed, "director", it))
        applied, failed = [], []
        for a in actions:
            ok, reason = policy.check(a)
            if not ok:
                policy.record(a, "rejected", reason)
                failed.append({"action": a, "error": reason})
                continue
            if scope.dry_run:
                policy.record(a, "dry_run", "dry run: not applied")
                applied.append(dict(a))
                continue
            snap = _snapshot(cand, a)
            try:
                rec = apply_action(cand, a, rng, policy)
            except ActionError as exc:
                _restore(cand, snap)
                policy.record(a, "rejected", str(exc))
                failed.append({"action": a, "error": str(exc)})
                continue
            ok, reason, stats = policy.verify_edit(snap["height_base"], cand.height_base)
            if not ok:
                _restore(cand, snap)
                policy.record(a, "reverted", reason, stats)
                failed.append({"action": a, "error": reason})
                continue
            policy.record(rec, "applied", "", stats)
            applied.append(rec)
        if scope.dry_run:
            policy.commit()
            world.history.append({
                "iteration": it, "accepted": False, "dry_run": True,
                "objective_before": current["objective"], "objective": current["objective"],
                "sim_scores": current["sim_scores"], "source": current["source"], "critique": current["critique"],
                "actions": [_round(a) for a in applied], "failed": failed,
            })
            say(f"director: dry run, logged {len(applied)} action(s) without applying them")
            break
        rejected += [action_key(f["action"]) for f in failed]   # never propose a failed action again
        if not applied:
            policy.rollback()
            rejected += [action_key(a) for a in actions]
            entry = {"iteration": it, "accepted": False, "objective_before": current["objective"],
                     "objective": current["objective"], "gain": 0.0, "sim_scores": current["sim_scores"],
                     "source": current["source"], "critique": current["critique"], "actions": [],
                     "failed": failed, "note": "no proposed action could be applied"}
            world.history.append(entry)
            current = review(world, llm, rejected, policy)
            log_refusals(current, it)
            note_fallback(entry, current)
            say(f"director: iteration {it}: none of {len(actions)} action(s) could be applied")
            continue
        say(f"director: iteration {it}, applying {len(applied)} action(s) and rebuilding")
        try:
            rebuild(cand, protect=policy.protect_rebuild(world))
        except ScopeViolation as exc:
            why = f"rebuild changed protected content: {exc.reason}"
            policy.revert_iteration(why, exc.stats)
            rejected += [action_key(a) for a in applied]
            entry = {"iteration": it, "accepted": False, "objective_before": current["objective"],
                     "objective": current["objective"], "gain": 0.0, "sim_scores": current["sim_scores"],
                     "source": current["source"], "critique": current["critique"],
                     "actions": [_round(a) for a in applied], "failed": failed, "note": why}
            world.history.append(entry)
            current = review(world, llm, rejected, policy)
            log_refusals(current, it)
            note_fallback(entry, current)
            say(f"director: iteration {it} discarded ({why})")
            continue
        policy.commit()                  # tentative: the candidate's proposal must see the remaining quota
        cand_review = review(cand, llm, rejected, policy)
        log_refusals(cand_review, it)
        before, after = _compare(current, cand_review)
        gain = after - before
        accepted = gain > scope.min_objective_gain
        entry = {
            "iteration": it,
            "accepted": accepted,
            "objective_before": round(before, 3),
            "objective": round(after, 3),
            "gain": round(gain, 3),
            "sim_scores": cand_review["sim_scores"],
            "source": current["source"],
            "critique": current["critique"],
            "actions": [_round(a) for a in applied],
            "failed": failed,
        }
        note_fallback(entry, cand_review)
        if accepted:
            cand.history = world.history + [entry]
            world = cand
            current = cand_review
        else:
            policy.rollback()
            world.history.append(entry)
            rejected += [action_key(a) for a in applied]
            current = review(world, llm, rejected, policy)
            log_refusals(current, it)
            note_fallback(entry, current)
        say(f"director: iteration {it} {'accepted' if accepted else 'rejected'} "
            f"(objective {entry['objective_before']:.2f} -> {entry['objective']:.2f})")
    world.evaluation["review"] = {k: v for k, v in current.items() if k != "actions"}
    world.evaluation["pending_actions"] = current["actions"]
    world.evaluation["ai_audit"] = policy.audit
    summary = policy.summary()
    if llm is not None and llm.available:
        summary["llm_budget"] = llm.budget_status()
    world.evaluation["ai_scope"] = summary
    return world
