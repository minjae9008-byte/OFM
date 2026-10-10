"""Terrain quality analysis (geomorphometry).

Objective measurements that separate natural-looking terrain from procedural
artefacts, each scored against a configurable band (``cfg.quality``):

* slope distribution (median, quantiles, 5 degree histogram) - very low medians
  read as featureless, very high ones as unplayable;
* hypsometric integral (Strahler 1952): mean relative elevation of the land;
  young, dissected uplands ~0.5, old eroded plains ~0.2;
* drainage density (Horton 1945): km of river per km2 of land, and length
  weighted river sinuosity;
* spurious pits: closed depressions deeper than 0.5 m that hydrology did not
  turn into lakes or dry basins (priority-flood fill difference);
* spikes / holes on the export grid: cells whose high-pass residual is many
  times the local residual level;
* grid artefacts: fourth circular moment of the high-pass gradient directions.
  Its real part is +1 when detail aligns with the grid axes and -1 when it
  aligns with the diagonals (D8 lines); natural detail has no 4-fold preference;
* power-spectrum slope: radially averaged power of the detrended, windowed
  export heightfield fitted as P(k) ~ k^-beta2 over mid frequencies; reported as
  the profile-equivalent exponent beta = beta2 - 1 (natural terrain ~2-2.6;
  white noise 0, very smooth surfaces > 3.5);
* landform classes (flat, slope, ridge, valley, peak, pit) from slope and
  Laplacian curvature, summarised by their normalised Shannon entropy;
* coastline box-counting dimension (natural coasts ~1.1-1.3), cliff share and
  terrain ruggedness index (Riley et al. 1999).

Result (JSON serialisable, stored in ``world.evaluation["terrain"]``):
``{"score", "sub_scores", "metrics", "flags"}``.
"""
from __future__ import annotations

import math

import numpy as np

from ..grid import NEIGHBORS8, blur, box_blur, shifted, sliding_max, sliding_min, slope
from ..physics.hydrology import priority_flood
from ..psych.scoring import band_score, upper_limit_score, weighted_overall

LANDFORMS = ("flat", "slope", "ridge", "valley", "peak", "pit")


def _f(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, 5) if math.isfinite(x) else None


# -- individual measurements ------------------------------------------------------------
def slope_stats(h: np.ndarray, cell: float, land: np.ndarray) -> dict:
    s = np.degrees(np.arctan(slope(h, cell)))[land]
    if s.size == 0:
        return {"median": None, "quantiles": {}, "histogram_5deg": []}
    q = np.quantile(s, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    hist, _ = np.histogram(s, bins=np.arange(0, 95, 5))
    return {
        "median": _f(q[2]),
        "quantiles": {k: _f(v) for k, v in zip(("p10", "p25", "p50", "p75", "p90", "p99"), q)},
        "histogram_5deg": [_f(v) for v in hist / max(1, s.size)],
    }


def hypsometric_integral(h: np.ndarray, land: np.ndarray) -> float | None:
    v = h[land]
    if v.size < 2 or np.ptp(v) <= 0:
        return None
    return float((v.mean() - v.min()) / np.ptp(v))


def drainage(rivers: list[dict], land_km2: float) -> dict:
    total_km = sum(float(r.get("length_m", 0.0)) for r in rivers) / 1000.0
    sin_num = sin_den = 0.0
    for r in rivers:
        pts = np.asarray(r.get("points", []), dtype=np.float64)
        if len(pts) < 3:
            continue
        path = float(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum())
        chord = float(np.linalg.norm(pts[-1, :2] - pts[0, :2]))
        if chord > 1e-6 and path > 0:
            sin_num += path / chord * path
            sin_den += path
    return {
        "density_km_per_km2": total_km / land_km2 if land_km2 > 0 else None,
        "river_km": total_km,
        "sinuosity": sin_num / sin_den if sin_den > 0 else None,
    }


def pit_fraction(world) -> float | None:
    land = ~world.ocean
    if not land.any():
        return None
    filled, _, _ = priority_flood(world.height, world.ocean)
    pits = (filled - world.height > 0.5) & land
    if world.lake_mask is not None:
        pits &= ~world.lake_mask
    if world.playa is not None:
        pits &= ~world.playa
    return float(pits.sum() / land.sum())


def spike_fraction(h: np.ndarray, land: np.ndarray, k: float = 7.0) -> float | None:
    """Cells whose high-pass residual exceeds k times the local residual level."""
    if not land.any():
        return None
    resid = h - box_blur(h, 1)
    level = box_blur(np.abs(resid), 3)
    floor = max(float(np.median(np.abs(resid[land]))), 1e-3)
    spike = (np.abs(resid) > k * np.maximum(level, floor)) & land
    return float(spike.sum() / land.sum())


def grid_alignment(h: np.ndarray, spacing: float, land: np.ndarray) -> tuple[float | None, float | None]:
    """(axis, diagonal) alignment of high-pass detail from the 4th circular moment of gradient directions."""
    detail = h - blur(h, 2.0)
    gy, gx = np.gradient(detail, spacing)
    w = np.hypot(gx, gy) * land
    if w.sum() <= 1e-12:
        return None, None
    theta = np.arctan2(gy, gx)
    re = float((w * np.cos(4.0 * theta)).sum() / w.sum())
    return max(0.0, re), max(0.0, -re)


def spectral_slope(h: np.ndarray, land: np.ndarray | None = None) -> float | None:
    """Profile-equivalent power-spectrum exponent of the (detrended, Hann-windowed) surface."""
    n = min(h.shape)
    if n < 32:
        return None
    z = h[:n, :n].astype(np.float64)
    if land is not None and land[:n, :n].any():
        # water is flattened to its own mean so sea floors do not dominate the spectrum
        m = land[:n, :n]
        z = np.where(m, z, float(z[m].mean()))
    yy, xx = np.mgrid[0:n, 0:n] / n
    A = np.stack([np.ones(n * n), xx.ravel(), yy.ravel()], axis=1)
    coef, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    z = z - (A @ coef).reshape(n, n)
    win = np.outer(np.hanning(n), np.hanning(n))
    p = np.abs(np.fft.fftshift(np.fft.fft2(z * win))) ** 2
    c = n // 2
    ky, kx = np.mgrid[-c:n - c, -c:n - c]
    kr = np.hypot(kx, ky).astype(int)
    radial = np.bincount(kr.ravel(), p.ravel()) / np.maximum(np.bincount(kr.ravel()), 1)
    ks = np.arange(len(radial))
    sel = (ks >= 4) & (ks <= n // 4) & (radial > 0)
    if sel.sum() < 4:
        return None
    slope2, _ = np.polyfit(np.log(ks[sel]), np.log(radial[sel]), 1)
    return float(-slope2 - 1.0)


def landform_fractions(h: np.ndarray, cell: float, land: np.ndarray) -> dict[str, float]:
    hs = blur(h, 1.0)
    s_deg = np.degrees(np.arctan(slope(hs, cell)))
    lap = sum(shifted(hs, dy, dx) for dy, dx in NEIGHBORS8) / 8.0 - hs       # > 0: concave (valley)
    # curvature threshold relative to this terrain's own curvature spread: about a fifth of
    # the land is classed ridge and a fifth valley on typical relief
    t = 0.8 * float(np.std(lap[land])) if land.any() else 0.0
    r = 3
    peak = (hs >= sliding_max(hs, r) - 1e-9) & (s_deg < 12)
    pit = (hs <= sliding_min(hs, r) + 1e-9) & (s_deg < 12)
    code = {k: i for i, k in enumerate(LANDFORMS)}
    cls = np.full(h.shape, code["slope"], dtype=np.int8)
    cls[lap < -t] = code["ridge"]
    cls[lap > t] = code["valley"]
    cls[(s_deg < 3.0) & (np.abs(lap) <= t)] = code["flat"]
    cls[peak] = code["peak"]
    cls[pit] = code["pit"]
    counts = np.bincount(cls[land], minlength=len(LANDFORMS))
    total = max(1, int(counts.sum()))
    return {k: float(counts[code[k]] / total) for k in LANDFORMS}


def entropy01(fractions: dict[str, float]) -> float | None:
    p = np.array([v for v in fractions.values() if v > 0])
    if p.size == 0:
        return None
    return float(-(p * np.log(p)).sum() / math.log(len(fractions)))


def _box_dimension(ocean: np.ndarray) -> float | None:
    coast = ocean & ~(shifted(ocean, 1, 0) & shifted(ocean, -1, 0) & shifted(ocean, 0, 1) & shifted(ocean, 0, -1))
    n = ocean.shape[0]
    sizes, counts = [], []
    s = 2          # at 1 cell a raster staircase over-counts diagonal coasts
    while s <= n // 16:
        m = n // s
        c = coast[: m * s, : m * s].reshape(m, s, m, s).any(axis=(1, 3)).sum()
        if c > 0:
            sizes.append(s)
            counts.append(c)
        s *= 2
    if len(sizes) < 3:
        return None
    d, _ = np.polyfit(np.log(1.0 / np.array(sizes)), np.log(np.array(counts)), 1)
    return float(d)


def coastline_dimension(ocean: np.ndarray, smooth_sigma: float = 4.0) -> float | None:
    """Box-counting dimension of the land/ocean boundary, bias-corrected.

    Raw box counting on a raster underestimates the dimension of oblique or
    map-crossing coasts (constant per-box offsets dominate at coarse sizes), so
    the same estimator is applied to a heavily smoothed copy of the coastline
    and only the excess is kept: D = 1 + D(coast) - D(smoothed coast). A smooth
    coast scores ~1.0 at any orientation; natural coasts ~1.1-1.3.
    """
    if not ocean.any() or ocean.all():
        return None
    raw = _box_dimension(ocean)
    smooth = blur(ocean.astype(np.float64), smooth_sigma) > 0.5
    if raw is None or not smooth.any() or smooth.all():
        return raw
    ref = _box_dimension(smooth)
    return raw if ref is None else float(1.0 + raw - ref)


def ruggedness(h: np.ndarray, land: np.ndarray) -> float | None:
    if not land.any():
        return None
    tri = np.sqrt(sum((shifted(h, dy, dx) - h) ** 2 for dy, dx in NEIGHBORS8))
    return float(tri[land].mean())


# -- analysis -------------------------------------------------------------------------------
def analyze_terrain(world) -> dict:
    qc = world.cfg.quality
    soft = qc.band_softness
    h = world.height
    cell = world.cell
    land = ~world.water
    land_km2 = land.sum() * cell * cell / 1e6
    hires = world.hires if world.hires is not None else h
    spacing = world.hires_spacing or cell
    land_hi = None
    if world.hires is not None:
        n = hires.shape[0]
        idx = np.clip(((np.arange(n) * spacing) / cell).astype(int), 0, world.size - 1)
        land_hi = land[np.ix_(idx, idx)]

    sl = slope_stats(h, cell, land)
    hyps = hypsometric_integral(h, land)
    dr = drainage(world.rivers, land_km2)
    pits = pit_fraction(world)
    spikes = spike_fraction(hires, land_hi if land_hi is not None else land)
    axis, diag = grid_alignment(hires, spacing, land_hi if land_hi is not None else land)
    beta = spectral_slope(hires, land_hi)
    forms = landform_fractions(h, cell, land) if land.any() else {k: 0.0 for k in LANDFORMS}
    form_entropy = entropy01(forms)
    coast_d = coastline_dimension(world.ocean)
    cliff = float(world.cliff_mask[land].mean()) if world.cliff_mask is not None and land.any() else None
    relief = float(np.ptp(h[land])) if land.any() else 0.0

    metrics = {
        "slope_median_deg": sl["median"],
        "slope_quantiles_deg": sl["quantiles"],
        "slope_histogram_5deg": sl["histogram_5deg"],
        "hypsometric_integral": _f(hyps),
        "relief_m": _f(relief),
        "relief_ratio": _f(relief / world.extent),
        "drainage_density_km": _f(dr["density_km_per_km2"]),
        "river_km": _f(dr["river_km"]),
        "river_sinuosity": _f(dr["sinuosity"]),
        "pit_fraction": _f(pits),
        "spike_fraction": _f(spikes),
        "axis_artifact": _f(axis),
        "diagonal_artifact": _f(diag),
        "spectral_slope": _f(beta),
        "landforms": {k: _f(v) for k, v in forms.items()},
        "landform_entropy": _f(form_entropy),
        "coastline_dimension": _f(coast_d),
        "cliff_fraction": _f(cliff),
        "ruggedness_m": _f(ruggedness(h, land)),
        "land_km2": _f(land_km2),
    }

    grid_score = None
    if axis is not None:
        grid_score = min(upper_limit_score(axis, qc.max_axis_artifact, soft),
                         upper_limit_score(diag, qc.max_diagonal_artifact, soft))
    sub = {
        "slope": band_score(sl["median"], qc.slope_median_deg, soft),
        "hypsometry": band_score(hyps, qc.hypsometric_integral, soft),
        "drainage": band_score(dr["density_km_per_km2"], qc.drainage_density_km, soft),
        "spectrum": band_score(beta, qc.spectral_slope, soft),
        "pits": upper_limit_score(pits, qc.max_pit_fraction, soft),
        "spikes": upper_limit_score(spikes, qc.max_spike_fraction, soft),
        "grid_artifacts": grid_score,
        "landforms": band_score(form_entropy, qc.landform_entropy, soft),
        "coastline": band_score(coast_d, qc.coastline_dimension, soft),
    }
    overall, eff = weighted_overall(sub, qc.weights)

    flags = []

    def outside(name, value, band, unit=""):
        if value is not None and not (band[0] <= value <= band[1]):
            flags.append(f"{name} {value:.3g}{unit} outside [{band[0]:g}, {band[1]:g}]")

    outside("slope median", sl["median"], qc.slope_median_deg, " deg")
    outside("hypsometric integral", hyps, qc.hypsometric_integral)
    outside("drainage density", dr["density_km_per_km2"], qc.drainage_density_km, " km/km2")
    outside("spectral slope", beta, qc.spectral_slope)
    outside("coastline dimension", coast_d, qc.coastline_dimension)
    outside("landform entropy", form_entropy, qc.landform_entropy)
    if pits is not None and pits > qc.max_pit_fraction:
        flags.append(f"spurious pits on {pits * 100:.2f}% of land (max {qc.max_pit_fraction * 100:.2f}%)")
    if spikes is not None and spikes > qc.max_spike_fraction:
        flags.append(f"spikes/holes on {spikes * 100:.3f}% of export cells (max {qc.max_spike_fraction * 100:.3f}%)")
    if axis is not None and axis > qc.max_axis_artifact:
        flags.append(f"grid-axis aligned detail {axis:.2f} (max {qc.max_axis_artifact:g})")
    if diag is not None and diag > qc.max_diagonal_artifact:
        flags.append(f"grid-diagonal aligned detail {diag:.2f} (max {qc.max_diagonal_artifact:g})")

    return {
        "score": _f(overall),
        "sub_scores": {k: _f(v) for k, v in sub.items()},
        "weights": {k: _f(v) for k, v in eff.items()},
        "metrics": metrics,
        "flags": flags,
    }


SUB_LABELS_KO = {
    "slope": "경사 분포", "hypsometry": "고도분포적분", "drainage": "배수 밀도", "spectrum": "스펙트럼 기울기",
    "pits": "가짜 웅덩이", "spikes": "스파이크", "grid_artifacts": "격자 아티팩트", "landforms": "지형형태 다양성",
    "coastline": "해안선 복잡도",
}


def format_quality_markdown(result: dict, language: str = "ko") -> str:
    ko = language == "ko"
    title = "## 지형 품질" if ko else "## Terrain quality"
    score = result.get("score")
    lines = [title, ""]
    if score is None:
        lines.append("- (분석 없음)" if ko else "- (not analysed)")
        return "\n".join(lines)
    lines += [f"- {'종합' if ko else 'Overall'}: **{score:.2f}/10**", "",
              "| 항목 | 점수 | 측정값 |" if ko else "| Criterion | Score | Measured |", "|---|---|---|"]
    m = result.get("metrics", {})
    measured = {
        "slope": m.get("slope_median_deg"), "hypsometry": m.get("hypsometric_integral"),
        "drainage": m.get("drainage_density_km"), "spectrum": m.get("spectral_slope"),
        "pits": m.get("pit_fraction"), "spikes": m.get("spike_fraction"),
        "grid_artifacts": (m.get("axis_artifact"), m.get("diagonal_artifact")),
        "landforms": m.get("landform_entropy"), "coastline": m.get("coastline_dimension"),
    }
    for k, v in (result.get("sub_scores") or {}).items():
        label = SUB_LABELS_KO.get(k, k) if ko else k
        val = measured.get(k)
        if isinstance(val, tuple):
            val_txt = " / ".join("-" if x is None else f"{x:.3g}" for x in val)
        else:
            val_txt = "-" if val is None else f"{val:.3g}"
        lines.append(f"| {label} | {'-' if v is None else f'{v:.2f}'} | {val_txt} |")
    flags = result.get("flags") or []
    if flags:
        lines += ["", "**경고**" if ko else "**Flags**"] + [f"- {f}" for f in flags]
    return "\n".join(lines)
