"""Water surface meshes: lakes (flat at their level) and river ribbons."""
from __future__ import annotations

import numpy as np

from ..detail.lakes import lake_sheets
from ..grid import bilinear
from ..volume.mesher import Mesh


def lake_meshes(world) -> list[tuple[dict, Mesh, np.ndarray]]:
    """One flat mesh per lake on the exported (detail) grid (see ``detail.lakes``)."""
    out = []
    sp = world.hires_spacing
    for lake, qj, qi in lake_sheets(world):
        level = lake["level"]
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


CREEK_LIFT = 0.15      # creek water above the detail ground (m): visible, never buried


def creek_centrelines(world, step: float | None = None) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    """(creek, xy samples every ``step`` m, half-widths) along every creek."""
    sp = world.hires_spacing or world.cell
    step = step or sp * 0.25
    out = []
    for creek in getattr(world, "creeks", None) or []:
        p = np.asarray(creek["points"], dtype=np.float64)[:, :2]
        if len(p) < 2:
            continue
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        if s[-1] <= 0:
            continue
        t = np.linspace(0.0, s[-1], int(np.ceil(s[-1] / step)) + 1)
        xy = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)
        # also where the path crosses a detail-grid line: the bilinear ground creases there,
        # and a strip vertex on each crease keeps the water from cutting under it
        extra = []
        for ax in (0, 1):
            a, b = xy[:-1, ax] / sp, xy[1:, ax] / sp
            k = np.ceil(np.minimum(a, b))
            hit = (k <= np.maximum(a, b)) & (np.abs(b - a) > 1e-9)
            f = (k[hit] - a[hit]) / (b[hit] - a[hit])
            extra.append(t[:-1][hit] + f * (t[1:][hit] - t[:-1][hit]))
        t = np.unique(np.concatenate([t] + extra))
        xy = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)
        half = np.maximum(0.5 * np.interp(t, s, np.asarray(creek["width"], dtype=np.float64)), 0.6)
        out.append((creek, xy, half))
    return out


def creek_meshes(world) -> list[tuple[dict, Mesh, np.ndarray]]:
    """Thin water strips draped on the detail ground along every creek: the export grid
    cannot carve a channel a few metres wide, so the water follows the ground just above it
    (its edges cross the same ground, so it never floats and is never buried)."""
    out = []
    if world.hires is None:
        return out
    sp = world.hires_spacing
    for creek, xy, half in creek_centrelines(world):
        d = np.gradient(xy, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        side = np.stack([-d[:, 1], d[:, 0]], axis=1)
        n = len(xy)
        # left, centre, right: across a convex patch the centre would sink under a two-vertex strip
        verts = np.empty((3 * n, 3))
        for k, f in enumerate((1.0, 0.0, -1.0)):
            verts[k::3, :2] = xy + side * (f * half)[:, None]
        u, v = verts[:, 0] / sp, verts[:, 1] / sp
        g = bilinear(world.hires, u, v)
        # on steep ground the strip crosses creases between its vertices: lift it a little more
        tan = np.hypot(bilinear(world.hires, u + 0.5, v) - bilinear(world.hires, u - 0.5, v),
                       bilinear(world.hires, u, v + 0.5) - bilinear(world.hires, u, v - 0.5)) / sp
        verts[:, 2] = g + CREEK_LIFT + np.minimum(0.25 * tan, 0.35)
        a = np.arange(n - 1) * 3
        tris = np.vstack([np.stack([a, a + 1, a + 4], 1), np.stack([a, a + 4, a + 3], 1),
                          np.stack([a + 1, a + 2, a + 5], 1), np.stack([a + 1, a + 5, a + 4], 1)]).astype(np.uint32)
        along = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        uvs = np.empty((3 * n, 2))
        uvs[0::3, 0], uvs[1::3, 0], uvs[2::3, 0] = 0.0, 0.5, 1.0
        for k in range(3):
            uvs[k::3, 1] = along / 10.0
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(verts), 1))
        colors = np.tile(np.array([[80, 140, 196, 200]], dtype=np.uint8), (len(verts), 1))
        out.append((creek, Mesh(positions=verts, normals=normals, indices=tris, colors=colors), uvs))
    return out


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
