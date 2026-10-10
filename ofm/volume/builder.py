"""Chunked volumetric meshing and engine-terrain hole cutting.

``hybrid`` mode keeps the engine's native terrain (Unity Terrain / Unreal
Landscape) for ordinary ground and only builds volumetric meshes where the
world cannot be a heightfield: caves, overhangs, sea notches, arches and
spires. Wherever the volume surface deviates from the heightfield near the
ground, a terrain hole is cut and the mesh supplies the surface instead.

``full`` mode meshes the whole world (useful for Nanite static meshes or
engines without terrain holes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..grid import bilinear, dilate, label
from ..physics.biome import BIOME_COLORS
from .density import DensityField
from .mesher import Mesh, surface_nets


@dataclass
class Chunk:
    key: tuple[int, int]
    origin: np.ndarray              # world position of the chunk's min corner
    mesh: Mesh
    kinds: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"chunk_{self.key[0]:03d}_{self.key[1]:03d}"


@dataclass
class VolumeResult:
    chunks: list[Chunk]
    holes: np.ndarray               # bool, (N-1, N-1) export cells, True = hole
    stats: dict


def _active_columns(world, field_: DensityField, n_cols: int, col_size: float, mode: str) -> dict:
    act: dict[tuple[int, int], set[str]] = {}

    def mark(lo, hi, tag):
        i0 = int(np.clip(np.floor(lo[0] / col_size), 0, n_cols - 1))
        i1 = int(np.clip(np.floor(hi[0] / col_size), 0, n_cols - 1))
        j0 = int(np.clip(np.floor(lo[1] / col_size), 0, n_cols - 1))
        j1 = int(np.clip(np.floor(hi[1] / col_size), 0, n_cols - 1))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                act.setdefault((i, j), set()).add(tag)

    if mode == "full":
        for i in range(n_cols):
            for j in range(n_cols):
                act.setdefault((i, j), set()).add("terrain")
    sp = field_.spacing
    min_cells = max(2, int((col_size / sp) ** 2 * 0.02))
    for tag, wmap in (("cliff", field_.cliff_w), ("coast", field_.coast_w)):
        jj, ii = np.nonzero(wmap > 0.5)
        if len(ii):
            ci = np.clip((ii * sp / col_size).astype(int), 0, n_cols - 1)
            cj = np.clip((jj * sp / col_size).astype(int), 0, n_cols - 1)
            counts = np.bincount(ci * n_cols + cj, minlength=n_cols * n_cols)
            for c in np.flatnonzero(counts >= min_cells).tolist():
                act.setdefault((c // n_cols, c % n_cols), set()).add(tag)
    for cave in world.caves:
        for seg in cave.segments:
            r = max(seg[6], seg[7]) + 4.0
            mark(np.minimum(seg[0:3], seg[3:6]) - r, np.maximum(seg[0:3], seg[3:6]) + r, "cave")
        for ch in cave.chambers:
            mark(ch[0:3] - ch[3:6] - 4.0, ch[0:3] + ch[3:6] + 4.0, "cave")
    for feat in world.features:
        lo, hi = feat.bbox()
        mark(lo, hi, "feature")
    return act


def _remove_floating(D: np.ndarray) -> np.ndarray:
    solid = D > 0
    lab, n = label(solid, diagonal=False)
    if n <= 1:
        return D
    keep = np.zeros(n + 1, dtype=bool)
    # Components within 3 voxels of a side may continue into the neighbouring chunk,
    # which only ever samples 2 voxels into this one, so both chunks agree.
    m = 3
    for face in (lab[:m], lab[-m:], lab[:, :m], lab[:, -m:], lab[:, :, :1]):
        keep[np.unique(face)] = True
    keep[0] = False
    floating = (lab > 0) & ~keep[lab]
    if floating.any():
        D = D.copy()
        D[floating] = -0.5
    return D


def _vertex_colors(world, field_: DensityField, mesh: Mesh) -> np.ndarray:
    p = mesh.positions
    strata = world.strata
    fold = strata.fold_at(p[:, 0], p[:, 1])
    rock = strata.color_table[strata.layer_index(p[:, 2], fold)]
    bi = np.clip((p[:, 0] / world.cell).astype(int), 0, world.size - 1)
    bj = np.clip((p[:, 1] / world.cell).astype(int), 0, world.size - 1)
    ground = BIOME_COLORS[world.biome[bj, bi]].astype(np.float64)
    d0 = field_.base_density(p[:, 0], p[:, 1], p[:, 2])
    up = np.clip((mesh.normals[:, 2] - 0.55) / 0.3, 0.0, 1.0)
    surface = np.clip(1.0 - np.abs(d0) / 3.0, 0.0, 1.0)
    w = (up * surface)[:, None]
    col = rock * (1 - w) + ground * w
    underground = np.clip((d0 - 6.0) / 10.0, 0.0, 1.0)[:, None]
    col = col * (1.0 - 0.35 * underground)
    out = np.empty((len(p), 4), dtype=np.uint8)
    out[:, :3] = np.clip(col, 0, 255).astype(np.uint8)
    out[:, 3] = np.clip(255 * (1.0 - underground[:, 0] * 0.5), 0, 255).astype(np.uint8)  # alpha = sky exposure
    return out


def build_volume(world, mode: str | None = None) -> VolumeResult:
    cfg = world.cfg
    vc = cfg.volume
    mode = mode or vc.mesh_mode
    voxel = vc.voxel_size
    C = vc.chunk_voxels
    field_ = DensityField(world)
    col_size = C * voxel
    n_cols = int(np.ceil(world.extent / col_size))
    active = _active_columns(world, field_, n_cols, col_size, mode)

    n_hi = world.hires.shape[0]
    sp = world.hires_spacing
    holes = np.zeros((n_hi - 1, n_hi - 1), dtype=bool)
    raw_chunks = []
    for (ci, cj), kinds in sorted(active.items()):
        gi = np.arange(-1, C + 2) + ci * C          # global voxel indices incl. 1-voxel apron
        gj = np.arange(-1, C + 2) + cj * C
        xs = gi * voxel
        ys = gj * voxel
        xx, yy = np.meshgrid(np.clip(xs, 0, world.extent), np.clip(ys, 0, world.extent), indexing="ij")
        hcol = field_.surface(xx, yy)
        zmin = float(hcol.min()) - 3 * voxel
        zmax = float(hcol.max()) + 3 * voxel
        x0, y0 = ci * col_size, cj * col_size
        for cave in world.caves:
            lo, hi = cave.bbox(pad=4.0)
            if hi[0] >= x0 and lo[0] <= x0 + col_size and hi[1] >= y0 and lo[1] <= y0 + col_size:
                zmin = min(zmin, float(lo[2]) - 2 * voxel)
        for feat in world.features:
            lo, hi = feat.bbox()
            if hi[0] >= x0 and lo[0] <= x0 + col_size and hi[1] >= y0 and lo[1] <= y0 + col_size:
                zmax = max(zmax, float(hi[2]) + 2 * voxel)
                zmin = min(zmin, float(lo[2]))
        if mode == "hybrid" and kinds <= {"cliff", "coast"}:
            zmin = max(zmin, float(hcol.min()) - vc.surface_band - 4 * voxel)
        k0 = int(np.floor(zmin / voxel)) - 1
        k1 = int(np.ceil(zmax / voxel)) + 1
        zs = np.arange(k0, k1 + 1) * voxel
        X, Y, Z = np.meshgrid(np.clip(xs, 0, world.extent), np.clip(ys, 0, world.extent), zs, indexing="ij")
        D, D0 = field_.sample(X, Y, Z)
        D[:, :, 0] = np.maximum(D[:, :, 0], 1.0)     # close the bottom
        D[:, :, -1] = np.minimum(D[:, :, -1], -1.0)  # open sky on top
        if vc.remove_floating:
            D = _remove_floating(D)
        origin = np.array([xs[0], ys[0], zs[0]], dtype=np.float64)
        owned = ((1, C + 1), (1, C + 1), (0, len(zs)))
        mesh = surface_nets(D, origin, voxel, owned)
        if mode == "hybrid":
            # Rock where the heightfield has air (features) or air close under the ground
            # (cave mouths, overhang recesses) means the engine terrain must be cut.
            solid = D > 0
            base_solid = D0 > 0
            near = np.abs(D0) < vc.surface_band
            mism = (solid != base_solid) & near
            mism |= solid & ~base_solid
            cols = mism[1:C + 1, 1:C + 1].any(axis=2)
            if cols.any():
                ii, jj = np.nonzero(cols)
                hx = np.clip(((ii + ci * C) * voxel / sp).astype(int), 0, n_hi - 2)
                hy = np.clip(((jj + cj * C) * voxel / sp).astype(int), 0, n_hi - 2)
                holes[hy, hx] = True
        raw_chunks.append(((ci, cj), origin, mesh, sorted(kinds)))

    if mode == "hybrid":
        holes = dilate(holes, 1)
        keep_mask = dilate(holes, 1)
    chunks = []
    tris = 0
    for key, origin, mesh, kinds in raw_chunks:
        if mesh.is_empty():
            continue
        if mode == "hybrid":
            p = mesh.positions
            cen = (p[mesh.indices[:, 0]] + p[mesh.indices[:, 1]] + p[mesh.indices[:, 2]]) / 3.0
            hx = np.clip((cen[:, 0] / sp).astype(int), 0, n_hi - 2)
            hy = np.clip((cen[:, 1] / sp).astype(int), 0, n_hi - 2)
            d0 = field_.base_density(cen[:, 0], cen[:, 1], cen[:, 2])
            keep = keep_mask[hy, hx] | (np.abs(d0) > 1.5 * voxel)
            mesh = mesh.select_faces(keep)
            if mesh.is_empty():
                continue
        mesh.colors = _vertex_colors(world, field_, mesh)
        mesh.meta = {"kinds": kinds}
        tris += mesh.triangle_count
        chunks.append(Chunk(key=key, origin=origin, mesh=mesh, kinds=kinds))
    stats = {
        "mode": mode,
        "voxel_size": voxel,
        "chunk_size_m": col_size,
        "active_columns": len(active),
        "mesh_chunks": len(chunks),
        "triangles": int(tris),
        "hole_cells": int(holes.sum()),
    }
    return VolumeResult(chunks=chunks, holes=holes, stats=stats)


def sample_terrain_height(world, x, y):
    """Height of the export terrain at world positions (bilinear on the hi-res field)."""
    return bilinear(world.hires, np.asarray(x) / world.hires_spacing, np.asarray(y) / world.hires_spacing)
