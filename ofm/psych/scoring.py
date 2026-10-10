"""Scoring primitives shared by the evaluation and the terrain-quality analysis.

* ``band_score``: designer targets are bands [low, high]; a value inside its
  band scores 10, outside the score falls off smoothly (Gaussian in the
  distance to the band, scaled by the band width and a softness factor), so
  "a bit too long between events" costs little and "far off" costs a lot.
* ``weighted_overall``: weighted mean of 0-10 sub-scores, renormalised over the
  sub-scores that are actually available (e.g. no terrain-quality score).
* ``summarize``: mean, standard deviation and a confidence interval for the
  Monte Carlo repetitions of the agent play-tests (Student t quantile from a
  Cornish-Fisher expansion of the normal quantile; exact enough for n >= 2).
"""
from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from statistics import NormalDist
from typing import Iterable, Mapping

LN2 = math.log(2.0)


def band_score(value: float | None, band, softness: float = 1.0) -> float | None:
    """10 inside [low, high]; Gaussian falloff outside, reaching 5 at a distance of
    ``softness x scale`` from the violated edge.

    The scale is the smaller of half the band width and half the magnitude of
    the violated edge, so narrow bands (a coastline dimension) and edges close to
    zero (a minimum drainage density) are both sensitive; when one of the two is
    zero the other is used.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    lo, hi = float(band[0]), float(band[1])
    if lo > hi:
        lo, hi = hi, lo
    if lo <= v <= hi:
        return 10.0
    edge = lo if v < lo else hi
    candidates = [c for c in (0.5 * (hi - lo), 0.5 * abs(edge)) if c > 0]
    scale = (min(candidates) if candidates else 1e-9) * max(float(softness), 1e-6)
    d = (lo - v) if v < lo else (v - hi)
    return 10.0 * math.exp(-LN2 * (d / scale) ** 2)


def upper_limit_score(value: float | None, limit: float, softness: float = 1.0) -> float | None:
    """Band [0, limit] for "at most" metrics (pits, spikes, artifacts), falloff relative to the limit."""
    if value is None:
        return None
    return band_score(value, (0.0, limit), softness)


def _weights_dict(weights) -> dict[str, float]:
    if is_dataclass(weights):
        return {k: float(v) for k, v in asdict(weights).items()}
    return {k: float(v) for k, v in dict(weights).items()}


def weighted_overall(scores: Mapping[str, float | None], weights) -> tuple[float | None, dict[str, float]]:
    """Weighted mean over the sub-scores that exist and have a positive weight.

    Returns (overall, effective normalised weights)."""
    w = _weights_dict(weights)
    used = {k: wk for k, wk in w.items() if wk > 0 and scores.get(k) is not None}
    total = sum(used.values())
    if total <= 0:
        return None, {}
    eff = {k: wk / total for k, wk in used.items()}
    return sum(float(scores[k]) * eff[k] for k in eff), eff


def t_quantile(p: float, dof: int) -> float:
    """Student-t quantile: exact for 1 and 2 degrees of freedom, Cornish-Fisher expansion
    around the normal quantile otherwise (within 1 % for dof >= 3 at the 90-95 % levels,
    a few % at 99 %)."""
    if dof <= 0:
        return float("inf")
    if dof == 1:                                   # Cauchy: exact
        return math.tan(math.pi * (p - 0.5))
    if dof == 2:                                   # exact closed form
        return (2 * p - 1) / math.sqrt(2 * p * (1 - p))
    z = NormalDist().inv_cdf(p)
    v = float(dof)
    z3, z5, z7 = z ** 3, z ** 5, z ** 7
    return (z + (z3 + z) / (4 * v) + (5 * z5 + 16 * z3 + 3 * z) / (96 * v * v)
            + (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / (384 * v ** 3))


def summarize(values: Iterable[float], confidence: float = 0.9,
              bounds: tuple[float | None, float | None] = (None, None)) -> dict:
    """Mean, sample std and a two-sided confidence interval of the mean, clipped to
    ``bounds`` (e.g. (0, 1) for fractions)."""
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci_low": None, "ci_high": None}
    mean = sum(xs) / n
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    half = t_quantile(0.5 + confidence / 2.0, n - 1) * std / math.sqrt(n)
    lo, hi = mean - half, mean + half
    if bounds[0] is not None:
        lo = max(lo, bounds[0])
    if bounds[1] is not None:
        hi = min(hi, bounds[1])
    return {"n": n, "mean": mean, "std": std, "ci_low": lo, "ci_high": hi}


def rounded(d: dict, digits: int = 3) -> dict:
    """Round floats (numpy scalars included) to plain Python floats for JSON output."""
    return {k: (round(float(v), digits) if isinstance(v, (float, int)) and not isinstance(v, bool) else v)
            for k, v in d.items()}
