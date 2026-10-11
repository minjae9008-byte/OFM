import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from ofm.psych.scoring import band_score, summarize, t_quantile, weighted_overall
from ofm.quality.metrics import (analyze_terrain, coastline_dimension, format_quality_markdown, grid_alignment,
                                 pit_fraction, spectral_slope, spike_fraction)


def _fbm(n, beta2, seed=0):
    """Isotropic random surface with radial power spectrum ~ k^-beta2."""
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(n)[None, :]
    k = np.hypot(kx, ky)
    k[0, 0] = 1.0
    amp = k ** (-beta2 / 2.0)
    amp[0, 0] = 0.0
    phase = np.exp(2j * np.pi * rng.random((n, n)))
    z = np.real(np.fft.ifft2(amp * phase))
    return z / z.std() * 50.0


@pytest.mark.parametrize("beta2", [2.6, 3.2, 3.8])
def test_spectral_slope_recovers_known_exponent(beta2):
    est = [spectral_slope(_fbm(256, beta2, s)) for s in range(3)]
    assert abs(np.mean(est) - (beta2 - 1.0)) < 0.3


def test_spike_detection():
    h = _fbm(128, 3.4, 1)
    land = np.ones_like(h, dtype=bool)
    base = spike_fraction(h, land)
    spiky = h.copy()
    rng = np.random.default_rng(3)
    idx = rng.choice(h.size, 40, replace=False)
    spiky.flat[idx] += rng.choice([-1, 1], 40) * 40.0
    assert base < 0.0005
    assert spike_fraction(spiky, land) >= 30 / h.size


def _grooves(n, angles, seed, count=300, depth=6.0):
    """Incised straight grooves (channel segments) with directions drawn from ``angles``."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    g = np.zeros((n, n))
    for _ in range(count):
        a = np.radians(rng.choice(angles))
        cx, cy, length = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(20, 80)
        dx, dy = np.cos(a), np.sin(a)
        t = np.clip((xx - cx) * dx + (yy - cy) * dy, -length / 2, length / 2)
        g = np.minimum(g, -depth * np.exp(-(np.hypot(xx - (cx + t * dx), yy - (cy + t * dy)) / 1.2) ** 2))
    return g


def test_grid_axis_artifacts():
    n = 256
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    land = np.ones((n, n), dtype=bool)
    base = _fbm(n, 3.2, 0) * 0.3 + 0.3 * xx                       # rough terrain on a regional tilt
    ax_a, dg_a = grid_alignment(base + _grooves(n, [0, 90], 0), 1.0, land)      # channels along the axes
    ax_d, dg_d = grid_alignment(base + _grooves(n, [45, 135], 0), 1.0, land)    # ... along the diagonals
    ax_r, dg_r = grid_alignment(base + _grooves(n, list(range(180)), 0), 1.0, land)   # any direction
    assert ax_a > 0.04 and dg_a == 0.0
    assert dg_d > 0.06 and ax_d == 0.0
    assert ax_r < 0.6 * ax_a and dg_r < 0.6 * dg_d


def test_natural_anisotropy_and_tilt_are_not_grid_artifacts():
    # parallel ridges (fold belts, dunes) are 2-fold, not 4-fold; a smooth tilted plane has none
    n = 256
    land = np.ones((n, n), dtype=bool)
    rng = np.random.default_rng(1)
    kx, ky = np.fft.fftfreq(n)[None, :], np.fft.fftfreq(n)[:, None]
    for ang in (0.0, 22.5, 45.0):
        a = np.radians(ang)
        ku, kv = kx * np.cos(a) + ky * np.sin(a), -kx * np.sin(a) + ky * np.cos(a)
        k = np.hypot(ku, 4 * kv)
        k[0, 0] = 1.0
        amp = k ** -1.6
        amp[0, 0] = 0.0
        z = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n)))))
        ax, dg = grid_alignment(z / z.std() * 50, 1.0, land)
        assert ax < 0.25 and dg < 0.25, ang
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    # no detail at all: not measurable, rather than round-off directions or blur edge effects
    assert grid_alignment(0.4 * xx + 0.1 * yy, 8.0, land) == (None, None)
    assert grid_alignment(0.002 * ((xx - 128) ** 2 + (yy - 128) ** 2), 8.0, land) == (None, None)
    ax_n, dg_n = grid_alignment(_fbm(n, 3.2, 2), 1.0, land)
    assert ax_n < 0.05 and dg_n < 0.05


@pytest.mark.parametrize("beta2", [2.4, 3.2, 4.8])
def test_spectral_slope_with_partial_land(beta2):
    # water raised to the shoreline: no coastline step that would pin the exponent near 2
    z = _fbm(512, beta2, 1)
    thr = np.quantile(z, 0.4)
    assert abs(spectral_slope(z, z > thr) - (beta2 - 1.0)) < 0.2


def test_spectral_slope_needs_land_and_relief():
    assert spectral_slope(np.full((129, 129), 50.0)) is None
    z = _fbm(128, 3.2, 0)
    assert spectral_slope(z, np.zeros_like(z, dtype=bool)) is None


def test_pits_ignore_lakes_and_count_spurious_bowls():
    n = 64
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    h = 100.0 - 0.5 * yy                                   # tilted plane draining south
    ocean = np.zeros((n, n), dtype=bool)
    ocean[:2] = True
    h[ocean] = -5.0
    bowl = np.hypot(xx - 32, yy - 40) < 6
    h2 = h - np.where(bowl, 8.0, 0.0)
    w = SimpleNamespace(height=h2, ocean=ocean, lake_mask=np.zeros_like(ocean), playa=np.zeros_like(ocean))
    frac = pit_fraction(w)
    assert frac == pytest.approx(bowl.sum() / (~ocean).sum(), rel=0.3)
    w.lake_mask = bowl.copy()
    assert pit_fraction(w) < frac / 5


def test_coastline_dimension_smooth_vs_fractal():
    n = 256
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    straight = coastline_dimension(xx < 128)
    disk = coastline_dimension(np.hypot(xx - 128, yy - 128) > 80)
    rough = coastline_dimension(_fbm(n, 2.4, 5) < 0)
    assert abs(straight - 1.0) < 0.05
    assert abs(disk - 1.0) < 0.1
    assert rough > 1.12
    assert coastline_dimension(np.zeros((n, n), dtype=bool)) is None


def test_band_score_and_overall():
    assert band_score(30, [20, 60]) == 10.0
    vals = [band_score(v, [20, 60]) for v in (60, 70, 80, 100)]
    assert vals == sorted(vals, reverse=True) and vals[0] == 10.0
    assert band_score(80, [20, 60]) == pytest.approx(5.0)        # one scale (min(width, edge)/2 = 20) above
    assert band_score(0.0, [0.4, 4.0]) < 1.0                     # edge near zero stays sensitive
    assert band_score(None, [0, 1]) is None and band_score(float("nan"), [0, 1]) is None
    assert band_score(5, [5, 5]) == 10.0 and band_score(4, [5, 5]) < 10.0
    overall, eff = weighted_overall({"a": 4.0, "b": 8.0, "c": None}, {"a": 1.0, "b": 3.0, "c": 5.0})
    assert overall == pytest.approx(7.0) and set(eff) == {"a", "b"}
    assert weighted_overall({"a": 4.0}, {"a": 0.0}) == (None, {})


def test_summary_statistics():
    assert t_quantile(0.975, 2) == pytest.approx(4.303, abs=1e-3)
    assert t_quantile(0.95, 30) == pytest.approx(1.697, abs=5e-3)
    s = summarize([1.0, 2.0, 3.0, 4.0], 0.9)
    assert s["ci_low"] < s["mean"] < s["ci_high"] and s["n"] == 4
    one = summarize([5.0])
    assert one["mean"] == 5.0 and one["ci_low"] is None and one["ci_high"] is None   # undefined, not zero-width
    assert summarize([])["mean"] is None


def test_analyze_generated_world(small_world):
    t0 = time.perf_counter()
    r = analyze_terrain(small_world)
    dt = time.perf_counter() - t0
    assert dt < 1.5
    json.dumps(r)
    assert 0.0 <= r["score"] <= 10.0
    assert set(r["sub_scores"]) >= {"slope", "spectrum", "pits", "spikes", "grid_artifacts"}
    assert isinstance(r["flags"], list)
    assert analyze_terrain(small_world) == r
    md = format_quality_markdown(r, "ko")
    assert md.startswith("## 지형 품질") and "종합" in md
    assert small_world.evaluation["terrain"]["score"] == r["score"]
