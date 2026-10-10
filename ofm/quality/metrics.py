"""Terrain quality analysis.

CONTRACT (used by pipeline.rebuild, stored in world.evaluation["terrain"]):

    analyze_terrain(world) -> dict with at least
        "score":      float 0..10 overall terrain quality (or None if not computed)
        "sub_scores": dict[str, float] 0..10 per criterion
        "metrics":    dict[str, float | int | list] raw measurements (JSON serialisable)
        "flags":      list[str] human-readable problems found (e.g. "spikes: 0.3% of cells")

The psychology scoring reads ``score`` (weight ``evaluation.weights.terrain_quality``).

PLACEHOLDER until implemented.
"""
from __future__ import annotations


def analyze_terrain(world) -> dict:
    return {"score": None, "sub_scores": {}, "metrics": {}, "flags": []}
