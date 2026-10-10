"""AI sandbox (ai.scope): capability checks, protected ground, edit verification, dry run,
LLM budget, structured outputs, repair loop, self-consistency and prompt capping."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from ofm.ai import director as director_mod
from ofm.ai.actions import ActionError, apply_action, validate_action
from ofm.ai.director import clone_world, run_director
from ofm.ai.evaluator import SCORE_KEYS, build_prompt_info, review
from ofm.ai.llm import LocalLLM
from ofm.ai.lore import _items, generate_lore
from ofm.ai.schema import review_schema, validate
from ofm.ai.scope import ScopePolicy
from ofm.config import ParamRange, ProtectedZone, load_config
from ofm.world import POI


# -- helpers --------------------------------------------------------------------------
def _world(small_world, **scope):
    """Independent copy of the shared small world with ai.scope overrides."""
    w = clone_world(small_world)
    w.psych = small_world.psych
    w.evaluation = dict(small_world.evaluation)
    w.history = []
    for k, v in scope.items():
        setattr(w.cfg.ai.scope, k, v)
    return w


def _land_region(world, min_relief=40.0):
    rows = [r for r in world.evaluation["simulation"]["regions"]
            if r["land_fraction"] > 0.9 and r["relief"] > min_relief]
    return max(rows, key=lambda r: r["relief"])["region"]


def _placeable_region(world, action):
    """A land region where ``action`` finds a site without any policy (the test world is steep,
    so not every region has walkable ground)."""
    rng = np.random.default_rng(0)
    rows = sorted(world.evaluation["simulation"]["regions"], key=lambda r: -r["relief"])
    for r in rows:
        if r["land_fraction"] < 0.5:
            continue
        probe = clone_world(world)
        probe.psych = world.psych
        try:
            apply_action(probe, validate_action(dict(action, region=r["region"]), world.all_regions()), rng)
        except ActionError:
            continue
        return r["region"]
    pytest.skip(f"no region of the test world can host {action}")


def _region_center(world, region):
    sj, si = world.region_slices(region)
    return (si.start + si.stop) / 2 * world.cell, (sj.start + sj.stop) / 2 * world.cell


SCORES = {k: 6.0 for k in SCORE_KEYS}


def _reply(actions, overall=6.0, critique="테스트 비평"):
    return {"scores": dict(SCORES, overall=overall), "critique": critique, "actions": actions}


class _Handler(BaseHTTPRequestHandler):
    replies: list = []          # queue of reply payloads (dict -> JSON text, str -> raw text); last one repeats
    requests: list = []
    reject_schema = False       # answer 400 when a JSON schema is sent (old servers)

    def log_message(self, *args):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "test-model:latest"}]})
        elif self.path == "/v1/models":
            self._send({"data": [{"id": "test-model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        req = json.loads(self.rfile.read(n))
        cls = type(self)
        cls.requests.append((self.path, req))
        schema_sent = isinstance(req.get("format"), dict) or \
            (req.get("response_format") or {}).get("type") == "json_schema"
        if cls.reject_schema and schema_sent:
            self._send({"error": "unsupported format"}, 400)
            return
        item = cls.replies[0] if len(cls.replies) == 1 else cls.replies.pop(0)
        content = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        if self.path == "/api/chat":
            self._send({"message": {"role": "assistant", "content": content}})
        else:
            self._send({"choices": [{"message": {"role": "assistant", "content": content}}]})


@pytest.fixture()
def server():
    _Handler.requests = []
    _Handler.replies = [_reply([])]
    _Handler.reject_schema = False
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _llm(url, provider="ollama", **scope):
    cfg = load_config(None, {"ai.provider": provider, "ai.base_url": url, "ai.model": "test-model"})
    for k, v in scope.items():
        setattr(cfg.ai.scope, k, v)
    llm = LocalLLM(cfg.ai)
    assert llm.available
    return llm


def _posts():
    return [(p, b) for p, b in _Handler.requests if p in ("/api/chat", "/v1/chat/completions")]


# -- capability checks ------------------------------------------------------------------
def test_disallowed_action_and_region_rejected(small_world):
    w = _world(small_world, allowed_actions=["add_poi", "set_param"], allowed_regions=["A1", "A2", "B1"],
               protected_regions=["B1"])
    pol = ScopePolicy(w.cfg.ai.scope, w)
    assert pol.allowed_regions == ["A1", "A2"]
    ok, why = pol.check({"type": "add_cave", "region": "A1"})
    assert not ok and "not allowed" in why
    ok, why = pol.check({"type": "add_poi", "region": "B1", "kind": "ruin"})
    assert not ok and "protected" in why
    ok, why = pol.check({"type": "add_poi", "region": "C3", "kind": "ruin"})
    assert not ok and "outside" in why
    assert pol.check({"type": "add_poi", "region": "A2", "kind": "ruin"})[0]
    # set_param: only configured paths, values clamped into the configured range
    w.cfg.ai.scope.param_ranges = [ParamRange("caves.density", 0.5, 1.5)]
    pol = ScopePolicy(w.cfg.ai.scope, w)
    a = {"type": "set_param", "path": "caves.density", "value": 9.0}
    assert pol.check(a)[0] and a["value"] == 1.5
    assert not pol.check({"type": "set_param", "path": "caves.max_length", "value": 9.0})[0]
    assert validate_action({"type": "set_param", "path": "caves.density", "value": 0.1}, ["A1"],
                           {"caves.density": (0.5, 1.5)})["value"] == 0.5
    with pytest.raises(ActionError):
        validate_action({"type": "set_param", "path": "climate.orographic", "value": 1}, ["A1"],
                        w.cfg.ai.scope.param_ranges)
    # the model can never widen its own sandbox or objective, even if a range is configured
    w.cfg.ai.scope.param_ranges = [ParamRange("ai.llm_score_weight", 0.0, 1.0),
                                   ParamRange("ai.scope.max_total_actions", 0, 99), ParamRange("world.size", 8, 64)]
    pol_deny = ScopePolicy(w.cfg.ai.scope, w)
    assert pol_deny.param_ranges == {} and len(pol_deny.warnings) == 3
    assert not pol_deny.check({"type": "set_param", "path": "ai.llm_score_weight", "value": 1.0})[0]
    with pytest.raises(ActionError):
        validate_action({"type": "set_param", "path": "ai.llm_score_weight", "value": 1}, ["A1"],
                        w.cfg.ai.scope.param_ranges)
    w.cfg.ai.scope.param_ranges = [ParamRange("caves.density", 0.5, 1.5)]
    # the forbidden mask covers every region outside the scope
    m = pol.forbidden_mask()
    sj, si = w.region_slices("A1")
    assert not m[sj, si].any()
    sj, si = w.region_slices("D4")
    assert m[sj, si].all()
    # the heuristic reviewer stays in scope as well
    r = review(w, None, policy=pol)
    assert all(a["type"] in ("add_poi", "set_param") and a.get("region", "A1") in ("A1", "A2") for a in r["actions"])


def test_protected_zone_and_region_respected(small_world):
    # Protect exactly the spot each action picks without a sandbox: with the policy the same
    # action (same rng) must land outside the zone or be refused. Choosing the zone this way
    # keeps the test independent of how rugged the small test world happens to be.
    regions = small_world.all_regions()
    rows = sorted(small_world.evaluation["simulation"]["regions"], key=lambda r: -r["land_fraction"])
    templates = ({"type": "add_poi", "kind": "camp"}, {"type": "add_poi", "kind": "shrine"},
                 {"type": "add_landmark", "kind": "monolith"}, {"type": "add_cave"})
    radius = 60.0
    moved = []
    for row in rows[:8]:
        for t in templates:
            a = validate_action(dict(t, region=row["region"]), regions)
            try:
                free = apply_action(_world(small_world), dict(a), np.random.default_rng(0))
            except ActionError:
                continue
            zx, zy = free["x"], free["y"]
            w = _world(small_world, protected_zones=[ProtectedZone(zx, zy, radius, "hand-made town")])
            pol = ScopePolicy(w.cfg.ai.scope, w)
            assert pol.forbidden_mask()[int(zy / w.cell), int(zx / w.cell)]
            try:
                rec = apply_action(w, dict(a), np.random.default_rng(0), pol)
            except ActionError:
                continue                                  # refusing is a valid outcome too
            moved.append((a["type"], a.get("kind")))
            assert np.hypot(rec["x"] - zx, rec["y"] - zy) > radius
            for obj in [p for p in w.pois if p.source == "ai"] + [f for f in w.features if f.source == "ai"]:
                assert np.hypot(obj.x - zx, obj.y - zy) > radius
            for f in (f for f in w.features if f.source == "ai"):
                reach = f.height * 0.75 + f.radius if f.kind == "arch" else f.radius
                assert np.hypot(f.x - zx, f.y - zy) > radius + reach - w.cell, "footprint overlaps the zone"
            for cave in w.forced_caves:
                seg = cave.segments
                assert np.hypot(seg[:, [0, 3]] - zx, seg[:, [1, 4]] - zy).min() > radius
                assert np.hypot(cave.chambers[:, 0] - zx, cave.chambers[:, 1] - zy).min() > radius
        if len(moved) >= 3:
            break
    assert moved, "at least one object must be re-sited outside the zone"

    # terrain edits fade out and leave the zone bit-identical
    w = _world(small_world)
    region = _land_region(w)
    cx, cy = _region_center(w, region)
    w.cfg.ai.scope.protected_zones = [ProtectedZone(cx, cy, 80.0, "hand-made town")]
    pol = ScopePolicy(w.cfg.ai.scope, w)
    zone = pol.forbidden_mask()
    assert zone.any()
    before = w.height_base.copy()
    relief = validate_action({"type": "raise_relief", "region": region, "amount": 0.6}, regions)
    apply_action(w, dict(relief), np.random.default_rng(0), pol)
    assert np.array_equal(w.height_base[zone], before[zone])
    sj, si = w.region_slices(region)
    assert np.abs(w.height_base - before)[sj, si].max() > 1.0
    ok, _, stats = pol.verify_edit(before, w.height_base)
    assert stats["forbidden_cells_changed"] == 0
    # without the policy the same edit would have touched the zone (the test is meaningful)
    w2 = _world(small_world)
    apply_action(w2, dict(relief), np.random.default_rng(0))
    assert not np.array_equal(w2.height_base[zone], before[zone])

    # a protected region is refused by check() and, even past it, offers no site
    w3 = _world(small_world, protected_regions=[region])
    pol3 = ScopePolicy(w3.cfg.ai.scope, w3)
    assert not pol3.check({"type": "add_poi", "region": region, "kind": "camp"})[0]
    sj, si = w3.region_slices(region)
    assert pol3.forbidden_mask()[sj, si].all() and not pol3.edit_weight()[sj, si].any()
    with pytest.raises(ActionError):
        apply_action(w3, validate_action({"type": "add_poi", "region": region, "kind": "camp"}, regions),
                     np.random.default_rng(0), pol3)


def test_spawn_settlement_is_protected(small_world):
    w = _world(small_world, protect_spawn_radius=200.0)
    x, y = _region_center(w, "B2")
    w.pois = [POI(id="poi_s", kind="settlement", x=x, y=y, z=0.0, importance=0.8)] + list(w.pois)
    pol = ScopePolicy(w.cfg.ai.scope, w)
    assert pol.spawn == (x, y)
    assert any(z["reason"] == "agent spawn settlement" for z in pol.zones())
    m = pol.forbidden_mask()
    i, j = int(x / w.cell), int(y / w.cell)
    assert m[j, i] and pol.edit_weight()[j, i] == 0.0
    assert 0 < m.mean() < 0.5


def test_per_action_and_total_limits(small_world):
    w = _world(small_world, max_total_actions=3, max_actions_per_iteration=2)
    w.cfg.ai.scope.limits.add_poi = 1
    pol = ScopePolicy(w.cfg.ai.scope, w)
    poi = {"type": "add_poi", "region": "A1", "kind": "ruin"}
    cave = {"type": "add_cave", "region": "A1"}
    pol.begin_iteration(1)
    assert pol.check(poi)[0]
    pol.record(poi, "applied")
    ok, why = pol.check(dict(poi, kind="camp"))
    assert not ok and "add_poi" in why
    pol.record(cave, "applied")
    ok, why = pol.check(cave)
    assert not ok and "per-iteration" in why
    pol.commit()
    pol.begin_iteration(2)
    assert pol.check(cave)[0]
    pol.record(cave, "applied")
    pol.commit()
    ok, why = pol.check(cave)
    assert not ok and "run limit" in why
    assert not pol.has_capacity()
    # a discarded candidate gives its quota back
    pol2 = ScopePolicy(w.cfg.ai.scope, w)
    pol2.begin_iteration(1)
    pol2.record(poi, "applied")
    pol2.commit()
    pol2.rollback()
    assert pol2.remaining("add_poi") == 1 and pol2.audit[-1]["kept"] is False
    # director run: the final world never holds more AI POIs than the limit
    w = _world(small_world)
    w.cfg.ai.scope.limits.add_poi = 1
    out = run_director(w, None, iterations=2)
    assert sum(p.source == "ai" for p in out.pois) <= 1
    assert out.evaluation["ai_scope"]["used"].get("add_poi", 0) <= 1
    assert isinstance(out.evaluation["ai_audit"], list)


def test_height_delta_cap_reverts_only_that_action(small_world, monkeypatch):
    w = _world(small_world, max_height_delta_m=1.0, protect_spawn_radius=0.0)
    region = _placeable_region(w, {"type": "add_poi", "kind": "camp"})
    base = w.height_base.copy()
    proposal = [{"type": "raise_relief", "region": region, "amount": 0.6, "reason": "t"},
                {"type": "add_poi", "region": region, "kind": "camp", "reason": "t"}]
    calls = []

    def fake_review(world, llm, rejected=None, policy=None):
        calls.append(1)
        return {"scores": {}, "critique": "", "source": "test", "llm": None,
                "actions": [dict(a) for a in proposal] if len(calls) == 1 else [],
                "sim_scores": {"overall": float(len(calls))}, "objective": float(len(calls))}

    monkeypatch.setattr(director_mod, "review", fake_review)
    out = run_director(w, None, iterations=1)
    audit = out.evaluation["ai_audit"]
    status = {e["action"]["type"]: e for e in audit}
    assert status["raise_relief"]["status"] == "reverted"
    assert "limit 1 m" in status["raise_relief"]["reason"]
    assert status["raise_relief"]["stats"]["max_delta_m"] > 1.0
    assert status["add_poi"]["status"] == "applied" and status["add_poi"]["kept"] is True
    assert np.array_equal(out.height_base, base), "the reverted edit left no trace"
    assert out.history[-1]["accepted"] and [a["type"] for a in out.history[-1]["actions"]] == ["add_poi"]
    # cumulative and area caps
    pol = ScopePolicy(w.cfg.ai.scope, w)
    bumped = pol.origin + 0.6
    ok, why, stats = pol.verify_edit(pol.origin, bumped)
    assert ok is False and "modified land area" in why and stats["modified_area_fraction"] > 0.9


def test_dry_run_logs_without_changing_the_world(small_world):
    w = _world(small_world, dry_run=True)
    base = w.height_base.copy()
    pois = [(p.id, p.x, p.y) for p in w.pois]
    feats = [(f.id, f.x, f.y) for f in w.features]
    caves = len(w.forced_caves)
    out = run_director(w, None, iterations=3)
    assert np.array_equal(out.height_base, base)
    assert [(p.id, p.x, p.y) for p in out.pois] == pois
    assert [(f.id, f.x, f.y) for f in out.features] == feats
    assert len(out.forced_caves) == caves
    audit = out.evaluation["ai_audit"]
    assert audit and all(e["status"] in ("dry_run", "rejected") for e in audit)
    assert any(e["status"] == "dry_run" for e in audit)
    assert len(out.history) == 2 and out.history[-1]["dry_run"] and not out.history[-1]["accepted"]


# -- LLM client ---------------------------------------------------------------------------
def test_budget_exhaustion_falls_back_to_heuristics(small_world, server):
    region = _land_region(small_world)
    _Handler.replies = [_reply([{"type": "add_poi", "region": region, "kind": "shrine", "reason": "x"}])]
    w = _world(small_world)
    llm = _llm(server, max_llm_calls=1, retries=0)
    r1 = review(w, llm)
    assert r1["source"] == "llm" and llm.calls == 1
    r2 = review(w, llm)
    assert r2["llm"] is None and r2["source"] == "heuristic" and "budget" in r2["llm_fallback"]
    st = llm.budget_status()
    assert st["exhausted"] and "call budget" in st["reason"]
    assert len(_posts()) == 1, "no request once the budget is spent"
    assert llm.chat_json("s", "u") is None
    # in the director the switch is logged in the history and the run completes
    w = _world(small_world)
    llm = _llm(server, max_llm_calls=1, retries=0)
    out = run_director(w, llm, iterations=1)
    assert any("budget" in h.get("llm_fallback", "") for h in out.history)
    assert out.evaluation["ai_scope"]["llm_budget"]["exhausted"]
    # a zero time budget disables the model as well
    llm = _llm(server, max_llm_seconds=0.0)
    assert llm.chat_json("s", "u") is None and "time budget" in llm.budget_status()["reason"]


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_structured_output_sends_schema(small_world, server, provider):
    w = _world(small_world, allowed_actions=["add_poi", "add_cave"], allowed_regions=["A1", "B2"])
    _Handler.replies = [_reply([{"type": "add_poi", "region": "B2", "kind": "ruin", "reason": "x"},
                                {"type": "add_lake", "region": "B2", "reason": "not allowed"}])]
    llm = _llm(server, provider, retries=0)
    r = review(w, llm)
    path, body = _posts()[-1]
    if provider == "ollama":
        schema = body["format"]
        assert body["options"]["num_ctx"] == 8192 and body["options"]["num_predict"] == 1500
        assert "seed" in body["options"]
    else:
        assert body["response_format"]["type"] == "json_schema"
        schema = body["response_format"]["json_schema"]["schema"]
        assert body["max_tokens"] == 1500 and "seed" in body
    item = schema["properties"]["actions"]["items"]["properties"]
    assert item["type"]["enum"] == ["add_cave", "add_poi"]
    assert item["region"]["enum"] == ["A1", "B2"]
    assert "path" not in item, "set_param is outside the scope"
    assert [a["type"] for a in r["actions"]] == ["add_poi"]
    # the out-of-scope action is either refused by schema validation or by the scope check
    assert r["llm"]["invalid_actions"] or r["llm"]["rejected"] or r["llm"]["schema_errors"]


def test_schema_rejected_by_server_falls_back_to_json_mode(server):
    _Handler.reject_schema = True
    _Handler.replies = [{"a": 1}]
    llm = _llm(server, retries=0)
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
    assert llm.chat_json("s", "u", schema=schema) == {"a": 1}
    bodies = [b for _, b in _posts()]
    assert isinstance(bodies[0]["format"], dict) and bodies[1]["format"] == "json"
    assert llm.calls == 1 and llm.last_info["format"] == "json"
    llm.chat_json("s", "u", schema=schema)
    assert _posts()[-1][1]["format"] == "json", "the working format is remembered"


def test_repair_loop_fixes_invalid_reply(server):
    schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "number", "maximum": 5}}}
    _Handler.replies = ["Sorry, I cannot produce JSON today.", {"n": 3}]
    llm = _llm(server, retries=2)
    assert llm.chat_json("sys", "give n", schema=schema) == {"n": 3}
    assert llm.calls == 2 and llm.repairs == 1
    msgs = _posts()[-1][1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "not a JSON object" in msgs[-1]["content"]
    # schema violations are quoted back as well; the best attempt is returned when retries run out
    _Handler.requests = []
    _Handler.replies = [{"n": 9}, {"n": 4}]
    assert llm.chat_json("sys", "give n", schema=schema) == {"n": 4}
    assert "above the maximum" in _posts()[-1][1]["messages"][-1]["content"]
    _Handler.replies = [{"n": 9}]
    assert llm.chat_json("sys", "give n", schema=schema, retries=1) == {"n": 9}
    assert llm.last_info["valid"] is False and llm.last_info["errors"]


def test_samples_majority_merge(small_world, server):
    w = _world(small_world)
    poi = {"type": "add_poi", "region": "B2", "kind": "shrine", "reason": "a"}
    cave = {"type": "add_cave", "region": "C3", "reason": "b"}
    _Handler.replies = [
        _reply([poi, cave, {"type": "add_lake", "region": "A1", "reason": "c"}], overall=4.0, critique="첫 번째"),
        _reply([cave, dict(poi, reason="again")], overall=6.0, critique="두 번째"),
        _reply([poi, {"type": "raise_relief", "region": "D4", "amount": 0.2, "reason": "d"}], overall=9.0),
    ]
    llm = _llm(server, samples=3, retries=0)
    r = review(w, llm)
    assert len(_posts()) == 3
    seeds = {b["options"]["seed"] for _, b in _posts()}
    assert len(seeds) == 3, "each sample uses its own seed"
    assert r["llm"]["samples"] == 3
    assert r["llm"]["scores"]["overall"] == 6.0
    assert [(a["type"], a["region"]) for a in r["actions"]] == [("add_poi", "B2"), ("add_cave", "C3")]
    assert r["critique"] == "첫 번째"
    assert set(r["llm"]["minority"]) == {"add_lake|A1||", "raise_relief|D4||"}
    assert r["objective"] == pytest.approx(0.7 * r["sim_scores"]["overall"] + 0.3 * 6.0, abs=1e-3)


def test_prompt_respects_max_prompt_chars(small_world, server):
    w = _world(small_world)
    full, info = build_prompt_info(w, [], 5, "ko")
    assert not info["trimmed"]
    for limit in (len(full) - 1, 6000, 4000, 2500, 2000):
        text, info = build_prompt_info(w, ["add_poi|A1|ruin|"] * 30, 5, "ko", max_chars=limit)
        assert len(text) <= limit and info["fits"], (limit, info)
        body = text.split("Map report:\n", 1)[1].split("\n\nReturn JSON:", 1)[0]
        payload = json.loads(body)                       # never cut mid-JSON
        assert "scope" in payload and payload["regions"]
    # regions are trimmed to the most bored ones first
    text, info = build_prompt_info(w, [], 5, "ko", max_chars=4000)
    kept = [r["region"] for r in json.loads(text.split("\n", 1)[1].split("\n\nReturn JSON:")[0])["regions"]]
    regions = sorted(w.evaluation["simulation"]["regions"], key=lambda r: -r["boredom_seconds"])
    assert kept[0] == regions[0]["region"]
    # end to end: the request the model receives obeys the cap
    _Handler.replies = [_reply([])]
    llm = _llm(server, max_prompt_chars=2500, retries=0)
    r = review(w, llm)
    user = [m for m in _posts()[-1][1]["messages"] if m["role"] == "user"][0]["content"]
    assert len(user) <= 2500 and r["llm"]["prompt_trimmed"]


def test_review_refusals_are_audited(small_world, server):
    w = _world(small_world, allowed_regions=["C1", "D2"])
    _Handler.replies = [_reply([{"type": "add_poi", "region": "C1", "kind": "camp", "reason": "in scope"},
                                {"type": "add_poi", "region": "A4", "kind": "ruin", "reason": "outside"},
                                {"type": "explode", "region": "C1", "reason": "not an action"}])]
    llm = _llm(server, retries=0)
    out = run_director(w, llm, iterations=1)
    review_rows = [e for e in out.evaluation["ai_audit"] if e["stage"] == "review"]
    assert any(e["action"].get("region") == "A4" and "outside the allowed regions" in e["reason"]
               for e in review_rows)
    assert any("unknown action 'explode'" in e["reason"] for e in review_rows)
    assert all(e["status"] == "rejected" for e in review_rows)
    applied = [e for e in out.evaluation["ai_audit"] if e["stage"] == "apply"]
    assert applied and all(e["action"].get("region") in ("C1", "D2", None) for e in applied)
    assert all(p.region in ("C1", "D2") for p in out.pois if p.source == "ai")


def test_lore_respects_llm_budget(small_world, server):
    w = _world(small_world)
    llm = _llm(server, max_llm_calls=0)
    lore = generate_lore(w, llm)
    assert lore["source"] == "procedural" and "call budget" in lore["llm_skipped"]
    assert not _posts(), "no naming request once the budget is spent"
    assert all(p.name for p in w.pois)
    # with budget left the names come from the model, one schema-constrained request per chunk
    w = _world(small_world)
    llm = _llm(server, max_llm_calls=8, retries=0)
    ids = [it["id"] for it in _items(w)]
    _Handler.replies = [{i: {"name": f"N{k}", "description": "d"} for k, i in enumerate(ids)}]
    lore = generate_lore(w, llm)
    assert lore["source"] == "llm" and not lore["llm_skipped"]
    body = _posts()[-1][1]
    assert isinstance(body["format"], dict) and set(body["format"]["required"]) <= set(ids)
    assert all(p.name.startswith("N") for p in w.pois)


def test_schema_validator_subset():
    s = review_schema(SCORE_KEYS, ["add_poi"], ["A1"], ["spire"], ["ruin"], [], 2)
    good = {"scores": dict(SCORES), "critique": "ok", "actions": [{"type": "add_poi", "region": "A1", "kind": "ruin"}]}
    assert validate(good, s) == []
    bad = {"scores": dict(SCORES, overall=11), "critique": 3,
           "actions": [{"type": "add_cave", "region": "Z9"}, {}, {"type": "add_poi"}]}
    errs = validate(bad, s)
    assert any("maximum" in e for e in errs) and any("expected string" in e for e in errs)
    assert any("at most 2" in e for e in errs) and any("'Z9'" in e for e in errs)
    assert any("missing required key 'type'" in e for e in errs)
    assert validate(True, {"type": "number"}) and not validate(2, {"type": "integer"})
