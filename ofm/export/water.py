"""Water surface meshes: lakes (flat at their level) and river ribbons."""
from __future__ import annotations

import numpy as np

from ..grid import bilinear, dilate, label
from ..volume.mesher import Mesh


def lake_meshes(world) -> list[tuple[dict, Mesh, np.ndarray]]:
    """One flat mesh per lake on the exported (detail) grid.

    Quads over lake cells are water; a one-cell shore ring is added where the ground stays
    above the water (the sheet tucks under the shore) but not where it falls away (the
    outlet valley), where a grown sheet would hang in the air."""
    out = []
    if world.lake_mask is None or not world.lake_mask.any():
        return out
    lab, n = label(world.lake_mask, diagonal=True)
    cell = world.cell
    h = world.hires
    sp = world.hires_spacing
    nq = h.shape[0] - 1
    qc = (np.arange(nq) + 0.5) * sp
    bi = np.clip((qc / cell).astype(int), 0, world.size - 1)
    vmin = np.minimum(np.minimum(h[:-1, :-1], h[:-1, 1:]), np.minimum(h[1:, :-1], h[1:, 1:]))
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
        wet = comp[np.ix_(bi, bi)]
        ring = dilate(comp, 1)[np.ix_(bi, bi)] & ~wet & (vmin >= level - 0.5)
        qj, qi = np.nonzero(wet | ring)
        if not len(qi):
            continue
        corners = np.stack([np.stack([qi, qj], 1), np.stack([qi + 1, qj], 1),
                            np.stack([qi + 1, qj + 1], 1), np.stack([qi, qj + 1], 1)], axis=1)
        uniq, inv = np.unique(corners.reshape(-1, 2), axis=0, return_inverse=True)
        inv = inv.reshape(-1, 4)
        verts = np.column_stack([uniq[:, 0] * sp, uniq[:, 1] * sp, np.full(len(uniq), level)])
        tris = np.vstack([inv[:, [0, 1, 2]], inv[:, [0, 2, 3]]]).astype(np.uint32)
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(verts), 1))
        colors = np.tile(np.array([[52, 104, 150, 200]], dtype=np.uint8), (len(verts), 1))
        mesh = Mesh(positions=verts, normals=normals, indices=tris, colors=colors)
        origin = np.array([verts[:, 0].min(), verts[:, 1].min(), level])
        uvs = (verts[:, :2] - origin[:2]) / 32.0
        out.append((lake, mesh, uvs))
    return out


def ribbon_half_width(world, width) -> np.ndarray:
    """Half-width of a river's water sheet: the channel the detail pass carved (at least
    1.25 export cells so it does not bead along diagonals, see ``hires.river_field``) plus
    half a cell, so the edges tuck under the banks instead of floating over the channel."""
    sp = world.hires_spacing or world.cell
    return np.maximum(0.5 * np.maximum(np.asarray(width, dtype=np.float64), 2.0), 1.25 * sp) + 0.5 * sp


def river_meshes(world) -> list[tuple[dict, Mesh, np.ndarray]]:
    """Ribbon along each river polyline at water-surface height; uv.y runs downstream."""
    out = []
    for river in world.rivers:
        pts = np.array(river["points"], dtype=np.float64)
        if len(pts) < 2:
            continue
        half = ribbon_half_width(world, river["width"])
        d = np.gradient(pts[:, :2], axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        side = np.stack([-d[:, 1], d[:, 0]], axis=1)
        left = pts[:, :2] + side * half[:, None]
        right = pts[:, :2] - side * half[:, None]
        z = pts[:, 2]
        verts = np.empty((2 * len(pts), 3))
        verts[0::2, :2] = left
        verts[1::2, :2] = right
        verts[0::2, 2] = z
        verts[1::2, 2] = z
        if world.hires is not None:
            # on steep reaches the banks fall away faster than a flat cross-section: drape the
            # edges onto the bank (white water) instead of leaving them hanging in the air
            sp = world.hires_spacing
            g = bilinear(world.hires, verts[:, 0] / sp, verts[:, 1] / sp)
            verts[:, 2] = np.minimum(verts[:, 2], g + 0.1)
        n = len(pts)
        a = np.arange(n - 1) * 2
        # (l_k, r_k, r_k+1), (l_k, r_k+1, l_k+1) is counter-clockwise seen from above
        tris = np.vstack([np.stack([a, a + 1, a + 3], 1), np.stack([a, a + 3, a + 2], 1)]).astype(np.uint32)
        along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))]
        uvs = np.empty((2 * n, 2))
        uvs[0::2, 0] = 0.0
        uvs[1::2, 0] = 1.0
        uvs[0::2, 1] = along / 20.0
        uvs[1::2, 1] = along / 20.0
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(verts), 1))
        colors = np.tile(np.array([[70, 130, 190, 210]], dtype=np.uint8), (len(verts), 1))
        mesh = Mesh(positions=verts, normals=normals, indices=tris, colors=colors)
        out.append((river, mesh, uvs))
    return out
