"""Ground materials on the export grid.

The simulation grid decides which materials a place can have (its biome, a river bed); the
export grid decides where exactly each lies: rock on the detail slopes (terrace risers,
canyon walls the 16 m grid smooths over), snow by the temperature at the detail elevation
and off steep faces, dirt on slopes. Biome boundaries are blurred over a cell and pushed
around by noise so they wander instead of tracing the simulation cells. Everything the
player sees uses this one field: the engines' splat maps, the browser viewer and the
volume meshes that fill terrain holes.
"""
from __future__ import annotations

import numpy as np

from ..grid import bilinear, blur, gradient
from ..noise import Noise, derive_seed
from ..physics.biome import _biome_terms, _normalize, _physical_terms


def detail_splat(world) -> np.ndarray:
    """Material weights (n, n, 8) at the detail vertices; cached until the detail grid changes."""
    cache = getattr(world, "_detail_splat_cache", None)
    if cache is not None and cache[0] is world.hires:
        return cache[1]
    h = world.hires
    n = h.shape[0]
    sp = world.hires_spacing
    c = np.arange(n) * sp / world.cell - 0.5
    U, V = np.meshgrid(c, c)
    noise = Noise(derive_seed(world.cfg.world.seed, "materials"))
    X, Y = np.arange(n)[None, :] * sp, np.arange(n)[:, None] * sp
    X, Y = np.broadcast_to(X, (n, n)), np.broadcast_to(Y, (n, n))
    # biome boundaries wander: the biome field is read through a smooth warp of about one
    # simulation cell (~90 m wavelength), so they no longer trace the cell grid; river beds
    # are not warped (their gravel stays in the channel)
    warp = 1.3
    Uw = U + warp * noise.fbm2(X / 90.0, Y / 90.0, 3) * 2.0
    Vw = V + warp * noise.fbm2(X / 90.0 + 41.7, Y / 90.0 - 23.9, 3) * 2.0
    bt = _biome_terms(world.biome, np.zeros_like(world.river_mask))
    w = np.stack([np.clip(bilinear(blur(bt[..., k], 0.8), Uw, Vw), 0.0, None) for k in range(bt.shape[-1])], axis=-1)
    w[..., 7] += bilinear(blur(world.river_mask.astype(np.float64) * 1.2, 0.8), U, V)
    # and each material's share varies a little within a biome (~40 m)
    for k in range(w.shape[-1]):
        if w[..., k].max() > 0:
            w[..., k] *= np.clip(1.0 + 0.9 * noise.fbm2(X / 42.0 + 17.3 * k, Y / 42.0 - 9.1 * k, 3), 0.15, None)
    gx, gy = gradient(h, sp)
    s_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    # temperature at the detail elevation (the lapse rate over the detail relief)
    t = bilinear(world.temperature, U, V) - world.cfg.climate.lapse_rate * (h - bilinear(world.height, U, V)) / 1000.0
    fold = bilinear(world.strata.fold, U, V)
    _physical_terms(w, s_deg, t, world.strata.hardness(h, fold))
    out = _normalize(w)
    world._detail_splat_cache = (world.hires, out)
    return out


def splat_at(world, x, y) -> np.ndarray:
    """Material weights (..., 8) at world positions, from the detail splat."""
    w = detail_splat(world)
    u, v = np.asarray(x) / world.hires_spacing, np.asarray(y) / world.hires_spacing
    return np.stack([bilinear(w[..., k], u, v) for k in range(w.shape[-1])], axis=-1)


def material_rgb(world, weights: np.ndarray, x, y, z) -> np.ndarray:
    """Colour of the ground material mix at world points: the material palette, with bare
    rock in the colour of the stratum exposed at that height (cliffs show their bands on the
    terrain and on the volume meshes alike)."""
    from ..physics.biome import MATERIALS
    pal = np.array([c for _, c in MATERIALS], dtype=np.float64)
    strata = world.strata
    rock = strata.color_table[strata.layer_index(np.asarray(z), strata.fold_at(x, y))].astype(np.float64)
    out = weights @ pal
    out += weights[..., 4:5] * (rock - pal[4])
    return out
