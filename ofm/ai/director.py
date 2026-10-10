"""AI director: an evaluate -> edit -> rebuild -> re-evaluate improvement loop.

Each iteration the reviewer (local LLM, or the heuristic fallback) proposes
whitelisted actions. They are applied to a copy of the world, every
downstream simulation is re-run, and the copy is kept only if the objective
(simulation score, optionally blended with the model's own score) improves.
Rejected actions are fed back so the next proposal differs.
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


def clone_world(world: World) -> World:
    psych = world.psych
    world.psych = {}
    try:
        c = copy.deepcopy(world)
    finally:
        world.psych = psych
    return c


def run_director(world: World, llm: LocalLLM | None, iterations: int | None = None,
                 progress: Callable[[str], None] | None = None) -> World:
    ai = world.cfg.ai
    iterations = ai.iterations if iterations is None else iterations
    rejected: list[str] = []
    current = review(world, llm, rejected)
    world.history.append({"iteration": 0, "accepted": True, "objective": current["objective"],
                          "sim_scores": current["sim_scores"], "source": current["source"],
                          "critique": current["critique"], "actions": []})
    for it in range(1, iterations + 1):
        actions = current["actions"]
        if not actions:
            if progress:
                progress(f"director: no further actions after {it - 1} iteration(s)")
            break
        cand = clone_world(world)
        rng = np.random.default_rng(derive_seed(world.cfg.world.seed, "director", it))
        applied, failed = [], []
        for a in actions:
            try:
                applied.append(apply_action(cand, a, rng))
            except ActionError as exc:
                failed.append({"action": a, "error": str(exc)})
        if not applied:
            rejected += [action_key(a) for a in actions]
            current = review(world, llm, rejected)
            continue
        if progress:
            progress(f"director: iteration {it}, applying {len(applied)} action(s) and rebuilding")
        rebuild(cand)
        cand_review = review(cand, llm, rejected)
        accepted = cand_review["objective"] > current["objective"]
        world_hist_entry = {
            "iteration": it,
            "accepted": accepted,
            "objective_before": current["objective"],
            "objective": cand_review["objective"],
            "sim_scores": cand_review["sim_scores"],
            "source": current["source"],
            "critique": current["critique"],
            "actions": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items()} for a in applied],
            "failed": failed,
        }
        if accepted:
            cand.history = world.history + [world_hist_entry]
            world = cand
            current = cand_review
        else:
            world.history.append(world_hist_entry)
            rejected += [action_key(a) for a in applied]
            current = review(world, llm, rejected)
        if progress:
            progress(f"director: iteration {it} {'accepted' if accepted else 'rejected'} "
                     f"(objective {world_hist_entry['objective_before']:.2f} -> {world_hist_entry['objective']:.2f})")
    world.evaluation["review"] = {k: v for k, v in current.items() if k != "actions"}
    world.evaluation["pending_actions"] = current["actions"]
    return world
