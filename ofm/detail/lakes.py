"""Lake water sheets on the export (detail) grid.

A lake's level and extent come from the simulation grid; its shoreline on the detail
grid lies wherever the detail ground rises past the level. ``lake_sheets`` finds the
quads a flat sheet at the level must cover, and every stage that asks "is this under a
lake?" (export meshes, vegetation, placement) uses the same answer.
"""
from __future__ import annotations

import numpy as np

from ..grid import dilate, label

# a sheet edge may stand this far above the ground it ends on (m)
SHORE_TOLERANCE = 0.5

_CORNERS = ((slice(None, -1), slice(None, -1)), (slice(None, -1), slice(1, None)),
            (slice(1, None), slice(None, -1)), (slice(1, None), slice(1, None)))


def _used_vertices(quads: np.ndarray) -> np.ndarray:
    v = np.zeros((quads.shape[0] + 1, quads.shape[1] + 1), dtype=bool)
    for s in _CORNERS:
        v[s] |= quads
    return v


def sheet_quads(h: np.ndarray, wet: np.ndarray, reach: np.ndarray, level: float,
                tol: float = SHORE_TOLERANCE) -> np.ndarray:
    """Quads of a flat water sheet at ``level`` over the vertex heights ``h``.

    The lake's own quads end on the boundary between two simulation cells, where the
    interpolated ground is about halfway between the bed and the shore, far below the
    water: the shoreline lies inside the first shore cell. The sheet therefore grows from
    every edge that would hang over lower ground, but only while the ground keeps rising
    (the basin wall); where it falls away (the outlet) it stops at the lip instead of
    hanging over the valley below. A last ring standing on ground at the level tucks the
    edge under the shore. ``reach`` bounds the growth."""
    low = h < level - tol
    g = [h[s] for s in _CORNERS]
    sheet = wet & reach
    for _ in range(4 * sum(sheet.shape)):
        vert = _used_vertices(sheet)
        hang = vert & low
        if not hang.any():
            break
        inner = [vert[s] for s in _CORNERS]
        touches = np.logical_or.reduce([hang[s] for s in _CORNERS])
        inner_min = np.min([np.where(i, gc, np.inf) for i, gc in zip(inner, g)], axis=0)
        grow = reach & ~sheet & touches
        for i, gc in zip(inner, g):
            grow &= i | (gc >= level - tol) | (gc >= inner_min)
        if not grow.any():
            break
        sheet |= grow
    vert = _used_vertices(sheet)
    touches = np.logical_or.reduce([vert[s] for s in _CORNERS])
    vmin = np.minimum(np.minimum(g[0], g[1]), np.minimum(g[2], g[3]))
    return sheet | (reach & touches & (vmin >= level - tol))


def lake_sheets(world) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    """(lake, quad rows, quad columns) of every lake's sheet (see ``sheet_quads``)."""
    out = []
    if world.lake_mask is None or not world.lake_mask.any() or world.hires is None:
        return out
    lab, n = label(world.lake_mask, diagonal=True)
    cell = world.cell
    h = world.hires
    sp = world.hires_spacing
    nq = h.shape[0] - 1
    qc = (np.arange(nq) + 0.5) * sp
    bi = np.clip((qc / cell).astype(int), 0, world.size - 1)
    levels = np.where(world.lake_mask, world.lake_level, np.nan)
    for lake in world.lakes:
        cx, cy = lake["center"]
        level = lake["level"]
        # find the component whose cells sit at this lake's level
        comp = None
        for k in range(1, n + 1):
            m = lab == k
            if abs(float(np.nanmean(levels[m])) - level) < 1e-3:
                jj, ii = np.nonzero(m)
                if np.hypot((ii.mean() + 0.5) * cell - cx, (jj.mean() + 0.5) * cell - cy) < 4 * cell + 1:
                    comp = m
                    break
        if comp is None:
            continue
        reach = dilate(comp, 2)[np.ix_(bi, bi)]
        rj, ri = np.nonzero(reach)
        j0, j1, i0, i1 = rj.min(), rj.max() + 1, ri.min(), ri.max() + 1
        quads = sheet_quads(h[j0:j1 + 1, i0:i1 + 1], comp[np.ix_(bi[j0:j1], bi[i0:i1])],
                            reach[j0:j1, i0:i1], level)
        qj, qi = np.nonzero(quads)
        if len(qi):
            out.append((lake, qj + j0, qi + i0))
    return out


def lake_surface(world) -> np.ndarray | None:
    """Water level over every detail-grid quad a lake sheet covers (NaN elsewhere)."""
    if world.hires is None:
        return None
    nq = world.hires.shape[0] - 1
    out = np.full((nq, nq), np.nan)
    for lake, qj, qi in lake_sheets(world):
        out[qj, qi] = np.fmax(out[qj, qi], lake["level"])
    return out


def under_lake(world, x, y, z, surface: np.ndarray | None = None, margin: float = 0.05) -> np.ndarray:
    """True where (x, y, z) lies below a lake sheet."""
    surface = lake_surface(world) if surface is None else surface
    if surface is None:
        return np.zeros(np.shape(x), dtype=bool)
    nq = surface.shape[0]
    qi = np.clip((np.asarray(x) / world.hires_spacing).astype(np.int64), 0, nq - 1)
    qj = np.clip((np.asarray(y) / world.hires_spacing).astype(np.int64), 0, nq - 1)
    level = surface[qj, qi]
    return np.isfinite(level) & (np.asarray(z) < np.where(np.isfinite(level), level, 0.0) + margin)
