"""Regression tests for evaluation/integration issues found in review round 2."""
import json
import math

import numpy as np
import pytest

from ofm.ai.director import clone_world
from ofm.config import ConfigError, load_config
from ofm.export.exporter import strict_json
from ofm.psych.interest import evaluate_psychology
from ofm.psych.scoring import summarize, t_quantile
from ofm.quality.metrics import landform_fractions, entropy01
from tests.conftest import small_config


def _fbm(n, beta2, seed=1):
    rng = np.random.default_rng(seed)
    k = np.hypot(np.fft.fftfreq(n)[None, :], np.fft.fftfreq(n)[:, None])
    k[0, 0] = 1.0
    a = k ** (-beta2 / 2)
    a[0, 0] = 0.0
    z = np.real(np.fft.ifft2(a * np.exp(2j * np.pi * rng.random((n, n)))))
    return z / z.std()


def test_landforms_measure_the_terrain_not_its_own_spread():
    z = _fbm(256, 3.2)
    land = np.ones(z.shape, dtype=bool)
    plain = landform_fractions(z * 1.0, 16.0, land)            # a few metres of relief over 4 km
    hills = landform_fractions(z * 60.0, 16.0, land)
    assert plain["flat"] > 0.8 and plain["ridge"] + plain["valley"] < 0.1
    assert hills["ridge"] > 0.2 and hills["valley"] > 0.2 and hills["flat"] < 0.05
    assert entropy01(plain) < 0.3 < entropy01(hills)


def test_flat_ground_is_flat_not_pit():
    f = landform_fractions(np.full((64, 64), 50.0), 16.0, np.ones((64, 64), dtype=bool))
    assert f["flat"] == 1.0 and f["pit"] == 0.0 and f["peak"] == 0.0


def test_unmeasured_playtest_is_not_a_perfect_score(small_world):
    w = clone_world(small_world)
    w.cfg.psych.agent_minutes = 0.0
    r = evaluate_psychology(w)
    for k in ("pacing", "exploration", "orientation"):
        assert k not in r["scores"], k
    assert r["metrics"]["bored_fraction"] is None
    assert "orientation" not in r["weights"]


def test_overall_falls_back_when_no_weighted_score_exists(small_world):
    w = clone_world(small_world)
    for k in ("scenic", "variety", "legibility", "mystery", "pacing", "exploration", "prospect_refuge",
              "orientation"):
        setattr(w.cfg.evaluation.weights, k, 0.0)
    w.cfg.evaluation.weights.terrain_quality = 1.0
    w.evaluation["terrain"] = {"score": None}
    r = evaluate_psychology(w)
    assert r["overall_basis"] == "unweighted mean"
    others = [v for k, v in r["scores"].items() if k != "overall"]
    assert r["scores"]["overall"] == pytest.approx(np.mean(others), abs=0.02)
    with pytest.raises(ConfigError, match="terrain_quality"):
        load_config(None, {**{f"evaluation.weights.{k}": 0.0 for k in ("scenic", "variety", "legibility",
                           "mystery", "pacing", "exploration", "prospect_refuge", "orientation")},
                           "quality.enabled": False})


def test_single_sample_has_no_interval():
    s = summarize([6.85], 0.9, (0.0, 10.0))
    assert s["n"] == 1 and s["mean"] == 6.85 and s["ci_low"] is None and s["ci_high"] is None
    with pytest.raises(ValueError):
        t_quantile(1.0, 5)


@pytest.mark.parametrize("override", [
    {"evaluation.confidence": "95"}, {"evaluation.confidence": "1.0"}, {"quality.spectral_slope": "2.0"},
    {"quality.coastline_dimension": "1.3,1.1"}, {"fluvial.dt_years": "0"}, {"ai.scope.max_llm_seconds": "inf"},
    {"evaluation.targets.event_gap_s": "100,20"}, {"evaluation.profiles": "explorer"},
    {"quality.band_softness": "0"}, {"tectonics.coast_roughness": "nan"}])
def test_bad_settings_are_config_errors(override):
    with pytest.raises(ConfigError):
        load_config(None, override)


def test_set_parses_typed_lists_and_tables():
    cfg = load_config(None, {"quality.slope_median_deg": "5,20", "evaluation.targets.event_gap_s": "9,30",
                             "ai.scope.protected_zones": '[{x = 800, y = 900, radius = 200, reason = "town"}]',
                             "evaluation.profiles": '[{name = "solo", count = 2, poi = 1.5}]'})
    assert cfg.quality.slope_median_deg == [5.0, 20.0]
    assert cfg.evaluation.targets.event_gap_s == [9.0, 30.0]
    z = cfg.ai.scope.protected_zones[0]
    assert (z.x, z.y, z.radius, z.reason) == (800.0, 900.0, 200.0, "town")
    assert [(p.name, p.count, p.poi) for p in cfg.evaluation.profiles] == [("solo", 2, 1.5)]


def test_manifest_json_is_strict():
    data = {"a": float("nan"), "b": [np.float32(1.5), np.inf], "c": np.arange(3), "d": {"e": np.bool_(True)}}
    out = strict_json(data)
    assert out == {"a": None, "b": [1.5, None], "c": [0, 1, 2], "d": {"e": True}}
    json.dumps(out, allow_nan=False)


def test_exported_world_is_strict_json(small_world, tmp_path):
    from ofm.export.exporter import export_world
    from ofm.volume.builder import build_volume
    w = clone_world(small_world)
    w.psych = small_world.psych
    w.evaluation["terrain"] = dict(w.evaluation.get("terrain") or {}, probe=float("nan"))
    path = export_world(w, build_volume(w), tmp_path)

    def reject(c):
        raise ValueError(c)

    for p in (path, tmp_path / "report.json"):
        json.loads(p.read_text(encoding="utf-8"), parse_constant=reject)
    assert json.loads(path.read_text(encoding="utf-8"))["evaluation"]["terrain"]["probe"] is None


def test_non_finite_terrain_is_an_error(monkeypatch):
    import ofm.pipeline as P
    monkeypatch.setattr(P, "carve_basins", lambda h, cfg: np.where(h > 0, np.nan, h))
    with pytest.raises(ConfigError, match="non-finite"):
        P.generate_base(small_config())
