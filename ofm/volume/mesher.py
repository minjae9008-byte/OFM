"""Naive Surface Nets isosurface extraction (vectorised).

Density convention: D > 0 is solid rock, D < 0 is air. One vertex is placed
per grid cell that the surface crosses (the mean of its edge crossings) and
one quad is emitted per crossing grid edge. Unlike marching cubes this needs
no lookup tables and produces smooth, well-shaped quads suitable for rocky
terrain, overhangs and caves.

Chunks sample one extra layer of grid points around their owned range, and
each grid edge is owned by exactly one chunk, so neighbouring chunks share
vertex positions exactly and meshes are seamless without duplicate faces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_CORNERS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
_EDGES = [
    ((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1)),
]


@dataclass
class Mesh:
    positions: np.ndarray                 # (n, 3) float64 world metres
    normals: np.ndarray                   # (n, 3) float32
    indices: np.ndarray                   # (m, 3) uint32
    colors: np.ndarray | None = None      # (n, 4) uint8
    meta: dict = field(default_factory=dict)

    @property
    def triangle_count(self) -> int:
        return int(len(self.indices))

    def is_empty(self) -> bool:
        return len(self.indices) == 0

    def compact(self) -> "Mesh":
        used = np.unique(self.indices)
        remap = np.full(len(self.positions), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return Mesh(
            positions=self.positions[used], normals=self.normals[used],
            indices=remap[self.indices].astype(np.uint32),
            colors=None if self.colors is None else self.colors[used], meta=dict(self.meta),
        )

    def select_faces(self, keep: np.ndarray) -> "Mesh":
        m = Mesh(self.positions, self.normals, self.indices[keep], self.colors, dict(self.meta))
        return m.compact() if len(m.indices) else Mesh(
            np.zeros((0, 3)), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32),
            None if self.colors is None else np.zeros((0, 4), np.uint8), dict(self.meta))


def empty_mesh() -> Mesh:
    return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32))


def surface_nets(D: np.ndarray, origin: np.ndarray, voxel: float,
                 owned: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None) -> Mesh:
    """Extract the D = 0 surface.

    D is indexed [i, j, k] = (x, y, z); grid point (i, j, k) is at
    origin + (i, j, k) * voxel. ``owned`` gives, per axis, the half-open
    range of grid-point indices whose outgoing edges this call emits.
    """
    D = np.asarray(D, dtype=np.float64)
    nx, ny, nz = D.shape
    if min(nx, ny, nz) < 2:
        return empty_mesh()
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    solid = D > 0

    sums = np.zeros((cx, cy, cz, 3))
    count = np.zeros((cx, cy, cz), dtype=np.int32)
    for p, q in _EDGES:
        dp = D[p[0]:p[0] + cx, p[1]:p[1] + cy, p[2]:p[2] + cz]
        dq = D[q[0]:q[0] + cx, q[1]:q[1] + cy, q[2]:q[2] + cz]
        cross = (dp > 0) != (dq > 0)
        if not cross.any():
            continue
        denom = np.where(cross, dp - dq, 1.0)
        t = np.where(cross, dp / denom, 0.0)
        for axis in range(3):
            coord = p[axis] + t * (q[axis] - p[axis])
            sums[..., axis] += np.where(cross, coord, 0.0)
        count += cross
    has = count > 0
    if not has.any():
        return empty_mesh()
    vidx = np.full((cx, cy, cz), -1, dtype=np.int64)
    n_verts = int(has.sum())
    vidx[has] = np.arange(n_verts)
    ci, cj, ck = np.nonzero(has)
    local = sums[has] / count[has][:, None]
    positions = np.asarray(origin, dtype=np.float64) + (np.stack([ci, cj, ck], axis=1) + local) * voxel

    if owned is None:
        owned = ((0, nx), (0, ny), (0, nz))
    quads = []
    for axis in range(3):
        u_ax, v_ax = [(1, 2), (2, 0), (0, 1)][axis]
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, -1)
        sl_hi[axis] = slice(1, None)
        change = solid[tuple(sl_lo)] != solid[tuple(sl_hi)]
        lo = [max(owned[a][0], 0) for a in range(3)]
        hi = [min(owned[a][1], change.shape[a]) for a in range(3)]
        # the two neighbouring axes need cells at (index - 1) and (index)
        for a in (u_ax, v_ax):
            lo[a] = max(lo[a], 1)
            hi[a] = min(hi[a], (nx, ny, nz)[a] - 1)
        if any(lo[a] >= hi[a] for a in range(3)):
            continue
        sub = change[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        e = np.nonzero(sub)
        if len(e[0]) == 0:
            continue
        idx = [e[a] + lo[a] for a in range(3)]

        def cell(du, dv):
            c = list(idx)
            c[u_ax] = c[u_ax] - du
            c[v_ax] = c[v_ax] - dv
            return vidx[c[0], c[1], c[2]]

        v0 = cell(1, 1)
        v1 = cell(0, 1)
        v2 = cell(0, 0)
        v3 = cell(1, 0)
        outward_pos = solid[idx[0], idx[1], idx[2]]       # solid below, air above along +axis
        q = np.stack([v0, v1, v2, v3], axis=1)
        q[~outward_pos] = q[~outward_pos][:, ::-1]
        quads.append(q)
    if not quads:
        return empty_mesh()
    q = np.vstack(quads)
    q = q[(q >= 0).all(axis=1)]
    # split each quad along its shorter diagonal
    p = positions
    d02 = np.linalg.norm(p[q[:, 0]] - p[q[:, 2]], axis=1)
    d13 = np.linalg.norm(p[q[:, 1]] - p[q[:, 3]], axis=1)
    use02 = d02 <= d13
    tris_a = np.where(use02[:, None], q[:, [0, 1, 2]], q[:, [0, 1, 3]])
    tris_b = np.where(use02[:, None], q[:, [0, 2, 3]], q[:, [1, 2, 3]])
    tris = np.vstack([tris_a, tris_b])

    normals = face_normals_to_vertex(positions, tris)
    mesh = Mesh(positions=positions, normals=normals, indices=tris.astype(np.uint32))
    return mesh.compact()


def face_normals_to_vertex(positions: np.ndarray, tris: np.ndarray) -> np.ndarray:
    a = positions[tris[:, 0]]
    b = positions[tris[:, 1]]
    c = positions[tris[:, 2]]
    fn = np.cross(b - a, c - a)
    vn = np.zeros_like(positions)
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.where(norm > 1e-12, vn / np.maximum(norm, 1e-12), np.array([0.0, 0.0, 1.0]))
    return vn.astype(np.float32)


def merge_meshes(meshes: list[Mesh]) -> Mesh:
    meshes = [m for m in meshes if not m.is_empty()]
    if not meshes:
        return empty_mesh()
    offs = np.cumsum([0] + [len(m.positions) for m in meshes[:-1]])
    colors = None
    if all(m.colors is not None for m in meshes):
        colors = np.vstack([m.colors for m in meshes])
    return Mesh(
        positions=np.vstack([m.positions for m in meshes]),
        normals=np.vstack([m.normals for m in meshes]),
        indices=np.vstack([m.indices.astype(np.int64) + o for m, o in zip(meshes, offs)]).astype(np.uint32),
        colors=colors,
    )
