import copy

import numpy as np
import pytest

from ofm.ai.director import clone_world
from ofm.config import AgentProfile
from ofm.psych.agents import agent_roster, simulate_agents
from ofm.psych.interest import evaluate_psychology
from ofm.psych.landmarks import select_landmarks
from ofm.psych.perception import compute_perception


def _world(small_world, **cfg):
    w = clone_world(small_world)
    w.cfg = copy.deepcopy(small_world.cfg)
    for path, value in cfg.items():
        obj = w.cfg
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)
    return w


def test_overall_follows_weights(small_world):
    w = _world(small_world)
    for k in ("scenic", "variety", "legibility", "mystery", "exploration", "prospect_refuge",
              "orientation", "terrain_quality"):
        setattr(w.cfg.evaluation.weights, k, 0.0)
    w.cfg.evaluation.weights.pacing = 1.0
    r = evaluate_psychology(w)
    assert r["scores"]["overall"] == pytest.approx(r["scores"]["pacing"], abs=0.01)
    assert r["weights"] == {"pacing": 1.0}


def test_missing_terrain_score_is_renormalised(small_world):
    w = _world(small_world)
    w.evaluation["terrain"] = {"score": None}
    r = evaluate_psychology(w)
    assert "terrain_quality" not in r["weights"] and "terrain_quality" not in r["scores"]
    assert sum(r["weights"].values()) == pytest.approx(1.0, abs=1e-3)
    w2 = _world(small_world)
    w2.evaluation["terrain"] = {"score": 2.0}
    r2 = evaluate_psychology(w2)
    assert r2["scores"]["terrain_quality"] == 2.0 and "terrain_quality" in r2["weights"]


def test_roster_counts_and_cycling(small_world):
    w = _world(small_world, **{"psych.agents": 0})
    w.cfg.evaluation.profiles = [AgentProfile("a", 2), AgentProfile("b", 3), AgentProfile("c", 0)]
    assert [p.name for p in agent_roster(w)] == ["a", "a", "b", "b", "b"]
    w.cfg.psych.agents = 4
    assert [p.name for p in agent_roster(w)] == ["a", "b", "a", "b"]


def test_monte_carlo_runs_and_statistics(small_world):
    w = _world(small_world, **{"evaluation.runs": 3, "psych.agents": 2})
    r = evaluate_psychology(w)
    assert r["metrics"]["agents"] == 6 and r["metrics"]["runs"] == 3
    st = r["statistics"]["overall_per_run"]
    assert st["n"] == 3 and st["ci_low"] <= st["mean"] <= st["ci_high"]
    for k in ("bored_fraction", "lost_fraction", "backtrack_ratio"):
        s = r["statistics"][k]
        assert 0.0 <= s["ci_low"] <= s["mean"] <= s["ci_high"] <= 1.0
    assert set(r["profiles"]) <= {p.name for p in w.cfg.evaluation.profiles}
    assert len(r["intensity_per_minute"]) == int(np.ceil(w.cfg.psych.agent_minutes))


def _sim(w, landmarks_visible=True):
    per = compute_perception(w)
    lms, vis = select_landmarks(w, per)
    if not landmarks_visible:
        vis = np.zeros_like(vis)
    return simulate_agents(w, per, lms, vis)


def test_fatigue_slows_agents(small_world):
    base = dict(randomness=0.0, risk_aversion=0.0, frontier=1.0, poi=1.0, landmark=1.0, count=1)
    fit = _world(small_world, **{"psych.agents": 0, "evaluation.runs": 1})
    fit.cfg.evaluation.profiles = [AgentProfile("fit", stamina_s=1e6, **base)]
    tired = _world(small_world, **{"psych.agents": 0, "evaluation.runs": 1})
    tired.cfg.evaluation.profiles = [AgentProfile("tired", stamina_s=5.0, fatigue_slowdown=0.6, **base)]
    a = _sim(fit)["agents"][0]
    b = _sim(tired)["agents"][0]
    assert a.fatigue_seconds == 0.0 and b.fatigue_seconds > 0.0
    assert b.steps < a.steps


def test_lost_time_grows_without_landmarks(small_world):
    w = _world(small_world, **{"evaluation.runs": 1, "evaluation.orientation_lost_s": 30.0})
    with_lm = sum(l.lost_seconds for l in _sim(w)["agents"])
    without = sum(l.lost_seconds for l in _sim(w, landmarks_visible=False)["agents"])
    assert without > with_lm
    assert without > 0


def test_deterministic(small_world):
    r1 = evaluate_psychology(_world(small_world))
    r2 = evaluate_psychology(_world(small_world))
    assert r1["scores"] == r2["scores"] and r1["statistics"] == r2["statistics"]


def test_stricter_targets_lower_the_score(small_world):
    loose = evaluate_psychology(_world(small_world))
    strict_w = _world(small_world)
    strict_w.cfg.evaluation.targets.event_gap_s = [5.0, 10.0]
    strict_w.cfg.evaluation.targets.bored_fraction = [0.0, 0.01]
    strict = evaluate_psychology(strict_w)
    assert strict["scores"]["pacing"] < loose["scores"]["pacing"] or loose["scores"]["pacing"] == 0.0
    assert strict["metrics"]["mean_event_gap_s"] == loose["metrics"]["mean_event_gap_s"]


def test_finer_region_grid(small_world):
    w = _world(small_world, **{"evaluation.region_divs": 6})
    r = evaluate_psychology(w)
    names = [row["region"] for row in r["regions"]]
    assert len(names) == 36 and names[0] == "A1" and names[-1] == "F6"
    assert all("lost_seconds" in row for row in r["regions"])
