"""Regression tests for sandbox holes found in review: malformed model data, non-finite
numbers, configs the pipeline rejects, transient server errors, truncated replies and
protected content that changed during a rebuild."""
import copy
import json

import numpy as np
import pytest

from ofm.ai import director as director_mod
from ofm.ai.actions import ActionError, apply_action, validate_action
from ofm.ai.director import clone_world, run_director
from ofm.ai.evaluator import _clean_scores, _parse_sample, review
from ofm.ai.llm import extract_json, salvage_object
from ofm.ai.lore import generate_lore
from ofm.ai.schema import validate
from ofm.ai.scope import ScopePolicy, ScopeViolation
from ofm.config import ConfigError, ParamRange, load_config
from ofm.pipeline import rebuild
from tests.test_ai_scope import SCORES, _Handler, _llm, _posts, _reply, _world, server  # noqa: F401

HUGE = json.loads("1" + "0" * 400)          # a JSON integer beyond the float range


def test_huge_and_non_finite_numbers_never_escape_as_exceptions(small_world):
    regions = small_world.all_regions()
    with pytest.raises(ActionError):
        validate_action({"type": "set_param", "path": "caves.density", "value": HUGE}, regions)
    a = validate_action({"type": "raise_relief", "region": "A1", "amount": HUGE}, regions)
    assert 0.1 <= a["amount"] <= 0.8
    assert _clean_scores(dict(SCORES, overall=HUGE)) is None
    assert _clean_scores(dict(SCORES, overall=float("nan"))) is None
    w = _world(small_world)
    pol = ScopePolicy(w.cfg.ai.scope, w)
    out = _parse_sample({"scores": dict(SCORES, overall=6.0), "actions": [
        {"type": "set_param", "path": "caves.density", "value": HUGE}, "not an object",
        {"type": "add_poi", "region": "B2", "kind": "ruin"}]}, regions, pol, [])
    assert len(out["invalid"]) == 2 and [x["type"] for x in out["actions"]] == ["add_poi"]


def test_model_region_names_are_normalised(small_world):
    regions = small_world.all_regions()
    assert validate_action({"type": "add_poi", "region": " ｂ ２ ", "kind": "ruin"}, regions)["region"] == "B2"


def test_schema_rejects_non_finite_numbers():
    schema = {"type": "object", "properties": {"x": {"type": "number", "minimum": 0, "maximum": 10}}}
    for bad in (float("nan"), float("inf"), HUGE):
        assert any("not a finite number" in e for e in validate({"x": bad}, schema))
    assert validate({"x": 5}, schema) == []


def test_verify_edit_is_not_blind_to_nan(small_world):
    w = _world(small_world, protected_regions=["A1"])
    pol = ScopePolicy(w.cfg.ai.scope, w)
    after = pol.origin.copy()
    after[:] = np.nan
    ok, reason, stats = pol.verify_edit(pol.origin, after)
    assert not ok and "non-finite" in reason
    assert stats["forbidden_cells_changed"] == int(pol.forbidden_mask().sum()) > 0
    json.dumps(stats, allow_nan=False)          # the audit stays valid JSON


@pytest.mark.parametrize("key,value", [("ai.scope.max_height_delta_m", "inf"),
                                       ("ai.scope.max_total_height_delta_m", "nan"),
                                       ("ai.scope.max_modified_area_fraction", "0"),
                                       ("ai.scope.protect_spawn_radius", "-1")])
def test_config_rejects_non_finite_scope_caps(key, value):
    with pytest.raises(ConfigError):
        load_config(None, {key: value})


def test_protected_regions_fail_closed():
    cfg = load_config(None, {"ai.scope.protected_regions": "Ｂ３, c 1"})
    assert cfg.ai.scope.protected_regions == ["Ｂ３", "c 1"]
    with pytest.raises(ConfigError, match="unknown regions"):
        load_config(None, {"ai.scope.protected_regions": "E5"})
    with pytest.raises(ConfigError, match="unknown regions"):
        load_config(None, {"ai.scope.allowed_regions": "B3.", "evaluation.region_divs": 4})


def test_policy_normalises_and_fails_closed(small_world):
    w = _world(small_world, protected_regions=["Ｂ３"])
    pol = ScopePolicy(w.cfg.ai.scope, w)
    assert pol.protected_regions == ["B3"] and "B3" not in pol.allowed_regions
    assert not pol.check(validate_action({"type": "raise_relief", "region": "B3"}, w.all_regions()))[0]
    w2 = _world(small_world, protected_regions=["Z9"])       # config edited after validation
    with pytest.raises(ConfigError):
        ScopePolicy(w2.cfg.ai.scope, w2)


def test_set_param_cannot_leave_an_invalid_config(small_world):
    w = _world(small_world, param_ranges=[ParamRange("fluvial.resolution_divisor", 1, 4)])
    w.cfg = copy.deepcopy(w.cfg)
    pol = ScopePolicy(w.cfg.ai.scope, w)
    a = validate_action({"type": "set_param", "path": "fluvial.resolution_divisor", "value": 3},
                        w.all_regions(), pol.param_ranges)
    assert pol.check(a)[0]
    before = w.cfg.fluvial.resolution_divisor
    with pytest.raises(ActionError, match="invalid"):
        apply_action(w, a, np.random.default_rng(0), pol)
    assert w.cfg.fluvial.resolution_divisor == before


def test_grid_changing_params_are_never_settable(small_world):
    # detail.upscale changes the detail grid, which the protection masks are built on
    w = _world(small_world, param_ranges=[ParamRange("detail.upscale", 1, 4)])
    pol = ScopePolicy(w.cfg.ai.scope, w)
    assert "detail.upscale" not in pol.param_ranges and pol.warnings


def test_add_lake_with_unbounded_cut_stays_finite(small_world):
    w = _world(small_world)
    w.cfg.ai.scope.max_height_delta_m = float("inf")         # bypasses validate on purpose
    pol = ScopePolicy(w.cfg.ai.scope, w)
    for region in w.all_regions():
        try:
            apply_action(w, validate_action({"type": "add_lake", "region": region}, w.all_regions()),
                         np.random.default_rng(0), pol)
        except ActionError:
            continue
        break
    assert np.isfinite(w.height_base).all()


# -- LLM transport -------------------------------------------------------------------------
def test_transient_5xx_does_not_disable_structured_output(server):
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
    _Handler.replies = [{"a": 1}]
    _Handler.fail_codes = [500, 500, 500]
    llm = _llm(server, retries=0)
    assert llm.chat_json("s", "u", schema=schema) is None          # every format level failed
    assert llm.chat_json("s", "u", schema=schema) == {"a": 1}
    assert isinstance(_posts()[-1][1]["format"], dict), "the server recovered: schema mode again"


def test_repair_turns_do_not_accumulate(server):
    schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "number", "maximum": 5}}}
    _Handler.replies = [{"n": 9}, {"n": 8}, {"n": 3}]
    llm = _llm(server, retries=2)
    assert llm.chat_json("sys", "give n", schema=schema) == {"n": 3}
    roles = [[m["role"] for m in b["messages"]] for _, b in _posts()]
    assert roles[-1] == ["system", "user", "assistant", "user"]
    assert json.loads(_posts()[-1][1]["messages"][2]["content"]) == {"n": 8}


def test_semantic_errors_trigger_a_repair(small_world, server):
    w = _world(small_world)
    poi = {"type": "add_poi", "region": "B2", "kind": "ruin", "reason": "x"}
    _Handler.replies = [_reply([{"type": "add_poi", "kind": "ruin"}]), _reply([poi])]
    llm = _llm(server, retries=1)
    r = review(w, llm, [], ScopePolicy(w.cfg.ai.scope, w))
    assert r["llm"]["repairs"] == 1
    assert "bad region" in _posts()[-1][1]["messages"][-1]["content"]


def test_prompt_example_stays_in_scope(small_world, server):
    w = _world(small_world, allowed_regions=[], allowed_actions=["add_poi", "set_param"],
               protected_regions=list(small_world.all_regions()))
    llm = _llm(server)
    review(w, llm, [], ScopePolicy(w.cfg.ai.scope, w))
    body = _posts()[-1][1]
    prompt = body["messages"][1]["content"]
    assert '"type": "add_poi"' not in prompt and '"set_param"' in prompt
    assert body["format"]["properties"]["actions"]["items"]["properties"]["type"]["enum"] == ["set_param"]


# -- truncated replies ------------------------------------------------------------------------
def test_truncated_reply_keeps_complete_members():
    text = '{"a": {"name": "x", "description": "y"}, "b": {"name": "z", "desc'
    assert extract_json(text) == {"a": {"name": "x", "description": "y"}}
    assert salvage_object('{"k": [1, 2], "j": 3') == {"k": [1, 2]}
    assert extract_json('{"a": {"b": 1}} trailing') == {"a": {"b": 1}}


def test_lore_chunks_fit_the_output_budget(small_world, server):
    w = _world(small_world)
    _Handler.replies = [{}]
    llm = _llm(server, max_output_tokens=400, retries=0)
    generate_lore(w, llm)
    for _, body in _posts():
        ids = json.loads(body["messages"][1]["content"])["items"]
        assert len(ids) <= max(4, int(0.8 * 400 / 90))


# -- protection across rebuilds ------------------------------------------------------------------
@pytest.fixture(scope="module")
def mid_world():
    """128 cells: big enough for a settlement (the agents' spawn) and global POI re-selection."""
    from ofm.pipeline import generate
    from tests.conftest import small_config
    return generate(small_config(**{"world.size": 128}))


def _scoped(world, **scope):
    w = clone_world(world)
    w.psych = world.psych
    w.evaluation = dict(world.evaluation)
    w.history = []
    w.cfg = copy.deepcopy(world.cfg)
    for k, v in scope.items():
        setattr(w.cfg.ai.scope, k, v)
    return w


def _inside(pol, objs):
    return sorted((o.kind, round(o.x, 3), round(o.y, 3)) for o in objs if pol.in_forbidden(o.x, o.y)[0])


def test_rebuild_restores_protected_content(mid_world):
    # the reviewer's case: smoothing B2 moved B3's spire, re-picked POIs and reshaped B3's
    # detail terrain by metres, and removed the spawn settlement
    w0 = _scoped(mid_world, protected_regions=["B3"])
    pol = ScopePolicy(w0.cfg.ai.scope, w0)
    cand = _scoped(w0)
    edit = validate_action({"type": "smooth_terrain", "region": "B2", "amount": 0.8}, w0.all_regions())
    assert pol.check(edit)[0]
    apply_action(cand, edit, np.random.default_rng(0), pol)
    probe = _scoped(cand)
    rebuild(probe, psych=False)
    _, fv = pol._vertex_weight(probe.hires.shape[0], probe.hires_spacing)
    assert np.abs(probe.hires - w0.hires)[fv].max() > 0.5, "without protection the edit leaks into B3"
    rebuild(cand, psych=False, protect=pol.protect_rebuild(w0))
    assert np.array_equal(cand.hires[fv], w0.hires[fv])
    assert _inside(pol, cand.features) == _inside(pol, w0.features)
    assert _inside(pol, cand.pois) == _inside(pol, w0.pois)
    sp0 = next(p for p in w0.pois if p.kind == "settlement")
    sp1 = next(p for p in cand.pois if p.kind == "settlement")      # the agents start at the first one
    assert (sp1.x, sp1.y) == (sp0.x, sp0.y)
    for kind in set(cand.scatter) | set(w0.scatter):
        a = cand.scatter.get(kind, np.zeros((0, 5)))
        b = w0.scatter.get(kind, np.zeros((0, 5)))
        a = a[pol.in_forbidden(a[:, 0], a[:, 1])] if len(a) else a
        b = b[pol.in_forbidden(b[:, 0], b[:, 1])] if len(b) else b
        assert np.array_equal(a[np.lexsort(a.T)], b[np.lexsort(b.T)]), kind
    feats = {f.id: f for f in cand.features}
    assert len(feats) == len(cand.features)
    for p in cand.pois:
        if p.kind == "rock_formation":
            f = feats[p.props[0]]
            assert (f.x, f.y) == (p.x, p.y)


def test_spawn_settlement_survives_rebuild(mid_world):
    w = _scoped(mid_world, protect_spawn_radius=0.0)
    sp0 = next(p for p in w.pois if p.kind == "settlement")
    w.cfg.detail.poi_spacing = 900.0           # a global change that re-picks every other POI
    rebuild(w, psych=False)
    sp1 = next(p for p in w.pois if p.kind == "settlement")
    assert (sp1.x, sp1.y) == (sp0.x, sp0.y)


def test_verify_rebuild_detects_protected_changes(mid_world):
    w0 = _scoped(mid_world)
    pol = ScopePolicy(w0.cfg.ai.scope, w0)
    assert pol.spawn is not None and pol.forbidden_mask().any()
    cand = _scoped(w0)
    assert pol.verify_rebuild(cand, w0)[0]
    j, i = np.argwhere(pol.forbidden_mask())[0]
    cand.river_mask = cand.river_mask.copy()
    cand.river_mask[j, i] = ~cand.river_mask[j, i]
    ok, reason, _ = pol.verify_rebuild(cand, w0)
    assert not ok and "river changed on 1 protected cell" in reason
    cand2 = _scoped(w0)
    cand2.pois = [p for p in cand2.pois if p.kind != "settlement"]
    assert "spawn settlement moved" in pol.verify_rebuild(cand2, w0)[1]
    cand3 = _scoped(w0)
    cand3.height = cand3.height.copy()
    cand3.height[j, i] = np.nan
    assert "non-finite" in pol.verify_rebuild(cand3, w0)[1]


def test_director_discards_candidates_that_change_protected_content(mid_world, monkeypatch):
    w = _scoped(mid_world)
    real = director_mod.rebuild

    def flooding_rebuild(world, progress=None, psych=True, protect=None):
        def hook(cand, stage):
            if stage == "objects":
                cand.ocean = np.ones_like(cand.ocean)    # the edit "drowned" everything, protected cells too
            protect(cand, stage)
        return real(world, progress, psych, protect=hook)

    monkeypatch.setattr(director_mod, "rebuild", flooding_rebuild)
    out = run_director(w, None, iterations=1)
    entry = out.history[-1]
    assert entry["accepted"] is False and "protected content" in entry["note"]
    audit = out.evaluation["ai_audit"]
    assert any(e["status"] == "reverted" and "ocean changed" in e["reason"] for e in audit)
    assert not any(e["status"] == "applied" for e in audit)
    assert out.evaluation["ai_scope"]["total_used"] == 0
    assert np.array_equal(out.height_base, mid_world.height_base)


# -- review round 2: attacks on the rebuild protection ----------------------------------------
@pytest.fixture(scope="module")
def cave_world():
    """Seed 7 at 128 cells: a karst cave runs under region C2."""
    from ofm.pipeline import generate
    from tests.conftest import small_config
    return generate(small_config(**{"world.size": 128, "world.seed": 7}))


def _guarded_rebuild(w0, action, **scope):
    w = _scoped(w0, **scope)
    pol = ScopePolicy(w.cfg.ai.scope, w)
    cand = _scoped(w)
    apply_action(cand, validate_action(action, w.all_regions()), np.random.default_rng(0), pol)
    rebuild(cand, psych=False, protect=pol.protect_rebuild(w))
    return w, pol, cand


def test_objects_in_the_blend_band_sit_on_the_exported_terrain(mid_world):
    from ofm.grid import bilinear
    w, pol, cand = _guarded_rebuild(mid_world, {"type": "smooth_terrain", "region": "B2", "amount": 0.8},
                                    protected_regions=["B3"])
    sp = cand.hires_spacing
    wgt = pol.edit_weight()
    for arr in cand.scatter.values():
        if not len(arr):
            continue
        i = np.clip((arr[:, 0] / pol.cell).astype(int), 0, pol.size - 1)
        j = np.clip((arr[:, 1] / pol.cell).astype(int), 0, pol.size - 1)
        band = (wgt[j, i] < 1.0) & ~pol.in_forbidden(arr[:, 0], arr[:, 1])
        ground = bilinear(cand.hires, arr[band, 0] / sp, arr[band, 1] / sp)
        assert np.abs(arr[band, 2] - ground).max(initial=0.0) < 0.25
    for p in cand.pois:
        if p.kind != "cave_entrance":            # a cave mouth keeps the passage's own height
            assert abs(p.z - float(bilinear(cand.hires, p.x / sp, p.y / sp))) < 0.25, p.kind


def test_protected_cells_keep_every_detail_vertex_and_material(mid_world):
    from ofm.grid import bilinear
    w, pol, cand = _guarded_rebuild(mid_world, {"type": "raise_relief", "region": "C3", "amount": 0.5},
                                    protected_regions=["B3"])
    F = pol.forbidden_mask()
    jj, ii = np.nonzero(F)
    # corners and centres of every protected cell, on the exported heightmap
    for dx, dy in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)):
        x, y = (ii + dx) * pol.cell / cand.hires_spacing, (jj + dy) * pol.cell / cand.hires_spacing
        assert np.abs(bilinear(cand.hires, x, y) - bilinear(w.hires, x, y)).max() < 1e-6
    assert np.array_equal(cand.biome[F], w.biome[F])
    assert np.abs(cand.splat[F] - w.splat[F]).max() < 1e-9


def test_caves_under_protected_cells_are_kept(cave_world):
    w, pol, cand = _guarded_rebuild(cave_world, {"type": "smooth_terrain", "region": "A1", "amount": 0.25},
                                    protected_regions=["C2"])

    def touching(world):
        return {c.id: np.asarray(c.segments).round(3).tolist() for c in world.caves if pol._touches(c)}

    assert touching(w), "the test world needs a cave under C2"
    assert touching(cand) == touching(w)
    ids = [c.id for c in cand.caves]
    assert len(ids) == len(set(ids))
    for p in cand.pois:
        if p.kind == "cave_entrance":
            assert p.props[0] in ids


def test_features_next_to_a_protected_area_keep_out_of_it(mid_world):
    from ofm.ai.actions import FOOTPRINT_M
    from ofm.ai.scope import RebuildGuard
    from ofm.volume.features import make_feature
    w = _scoped(mid_world, protected_regions=["B3"])
    pol = ScopePolicy(w.cfg.ai.scope, w)
    F = pol.forbidden_mask()
    jj, ii = np.nonzero(F)
    # just outside the western edge of B3, close enough for an arch footprint to reach in
    x = (ii.min() - 1 + 0.5) * pol.cell
    y = float(np.median(jj) + 0.5) * pol.cell
    cand = _scoped(w)
    arch = make_feature(cand, "arch", x, y, np.random.default_rng(0))
    arch.source = "procedural"
    cand.features = list(cand.features) + [arch]
    RebuildGuard(pol, w).surface(cand)
    assert all(np.hypot(f.x - x, f.y - y) > 1.0 for f in cand.features), "the arch would reach into B3"
    assert FOOTPRINT_M["arch"] > pol.cell
    ids = [f.id for f in cand.features]
    assert len(ids) == len(set(ids))


def test_a_rebuild_without_edits_reproduces_the_world(mid_world):
    w = _scoped(mid_world)
    before = [(p.kind, p.x, p.y) for p in w.pois]
    rebuild(w, psych=True)
    assert [(p.kind, p.x, p.y) for p in w.pois] == before
    assert w.evaluation["simulation"]["scores"] == mid_world.evaluation["simulation"]["scores"]


def test_a_schema_valid_reply_beats_a_repair_that_lost_a_key(small_world, server):
    w = _world(small_world)
    no_region = {"type": "add_poi", "kind": "ruin", "reason": "x"}
    good = {"type": "add_poi", "region": "B2", "kind": "ruin", "reason": "x"}
    _Handler.replies = [_reply([good, no_region, no_region], overall=9.0), {"critique": "c", "actions": []}]
    r = review(w, _llm(server, retries=1), [], ScopePolicy(w.cfg.ai.scope, w))
    assert r["llm"]["scores"] is not None and r["llm"]["scores"]["overall"] == 9.0
    assert [a["region"] for a in r["llm"]["actions"]] == ["B2"]


def test_prompt_example_skips_actions_without_quota(small_world, server):
    w = _world(small_world)
    w.cfg = copy.deepcopy(w.cfg)
    w.cfg.ai.scope.limits.add_poi = 0
    pol = ScopePolicy(w.cfg.ai.scope, w)
    review(w, _llm(server), [], pol)
    body = _posts()[-1][1]
    assert "add_poi" not in body["format"]["properties"]["actions"]["items"]["properties"]["type"]["enum"]
    assert '"type": "add_poi"' not in body["messages"][1]["content"]


def test_truncated_wrappers_and_numbers():
    from ofm.ai.llm import json_candidates
    cands = json_candidates('{"review": {"scores": {"overall": 7}, "critique": "c", "actions": []}')
    assert {"scores": {"overall": 7}, "critique": "c", "actions": []} in cands
    assert salvage_object('{"a": 1, "n": 1.') == {"a": 1}
    assert salvage_object('{"a": 1, "n": 12e') == {"a": 1}
    assert salvage_object('{"a": 1 "b": 2, "c": 3') is None
    assert salvage_object('{"a": "x", "b": {"c": 1}') == {"a": "x", "b": {"c": 1}}
