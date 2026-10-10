"""Perception model: isovists from a grid of observers.

For each observer (eye height above the ground) rays are marched across the
"visual height" field (terrain + volumetric landmarks + water surface). A
sample is visible when its elevation angle is at least the running maximum
of all nearer samples (classic horizon test). From the isovists we derive
environmental-psychology measures:

* prospect  - how far and how much one can see (Appleton's prospect)
* refuge    - how sheltered the spot is: nearby walls/overhangs, canopy, caves
* mystery   - partially hidden information: terrain disappears behind a
              crest and re-appears further away (Kaplan & Kaplan's
              "deflected vista" / promise of more information)
* complexity, coherence from local land-cover statistics.

The observer -> navigation-cell visibility matrix is reused by the
exploration agents.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import bilinear, box_blur, slope, smoothstep
from ..physics.biome import BIOME_VEG, BIOMES


@dataclass
class Perception:
    grid: int
    cell: float                      # observer spacing (m)
    ox: np.ndarray                   # (P,) observer x
    oy: np.ndarray
    oz: np.ndarray                   # eye elevation
    land: np.ndarray                 # (G, G) observer on walkable land
    visible: np.ndarray              # (P, P) observer -> nav cell visibility
    view_area: np.ndarray            # (G, G) m2
    prospect: np.ndarray             # (G, G) 0..1
    refuge: np.ndarray               # (G, G) 0..1
    mystery: np.ndarray              # (G, G) 0..1
    complexity: np.ndarray           # (G, G) 0..1
    coherence: np.ndarray            # (G, G) 0..1
    vista: np.ndarray                # (G, G) 0..1 (top view-area percentile)


def compute_perception(world, batch: int = 384) -> Perception:
    pc = world.cfg.psych
    G = pc.grid
    extent = world.extent
    cs = extent / G
    vh = world.visual_height()
    cell = world.cell

    gi, gj = np.meshgrid(np.arange(G), np.arange(G))
    ox = ((gi + 0.5) * cs).ravel()
    oy = ((gj + 0.5) * cs).ravel()
    ground = bilinear(vh, ox / cell - 0.5, oy / cell - 0.5)
    oz = ground + pc.eye_height
    land_full = ~world.water
    land = bilinear(land_full.astype(float), ox / cell - 0.5, oy / cell - 0.5).reshape(G, G) > 0.5

    K, S = pc.rays, pc.steps
    theta = np.arange(K) * (2 * np.pi / K)
    d0 = max(cell * 0.75, 6.0)
    dist = d0 * (pc.view_distance / d0) ** (np.arange(S) / (S - 1))
    dd = np.gradient(dist)
    ring_w = dist * dd * (2 * np.pi / K)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    P = G * G
    visible = np.zeros((P, P), dtype=bool)
    view_area = np.zeros(P)
    depth_far = np.zeros((P, K))
    near_block = np.zeros(P)
    mystery = np.zeros(P)
    for b0 in range(0, P, batch):
        b1 = min(P, b0 + batch)
        px = ox[b0:b1, None, None] + dist[None, None, :] * cos_t[None, :, None]
        py = oy[b0:b1, None, None] + dist[None, None, :] * sin_t[None, :, None]
        inside = (px >= 0) & (px <= extent) & (py >= 0) & (py <= extent)
        H = bilinear(vh, px / cell - 0.5, py / cell - 0.5)
        tan = (H - oz[b0:b1, None, None]) / dist[None, None, :]
        cmax = np.maximum.accumulate(tan, axis=-1)
        prev = np.concatenate([np.full(tan.shape[:2] + (1,), -np.inf), cmax[..., :-1]], axis=-1)
        vis = (tan >= prev - 1e-9) & inside
        view_area[b0:b1] = (vis * ring_w[None, None, :]).sum(axis=(1, 2))
        far_idx = np.where(vis, np.arange(S)[None, None, :], -1).max(axis=-1)
        depth_far[b0:b1] = np.where(far_idx >= 0, dist[np.clip(far_idx, 0, S - 1)], 0.0)
        # refuge: steep horizon close by (walls, cliffs, overhang lips)
        close = dist < 70.0
        near_block[b0:b1] = (cmax[..., close].max(axis=-1) > np.tan(np.radians(22))).mean(axis=-1) \
            if close.any() else 0.0
        # mystery: hidden stretch at 40..600 m followed by visible ground further out
        mid = (dist > 40.0) & (dist < 600.0)
        hidden_mid = (~vis[..., mid] & inside[..., mid]).any(axis=-1)
        last_mid = np.flatnonzero(mid).max()
        reveal = vis[..., last_mid + 1:].any(axis=-1) if last_mid + 1 < S else np.zeros_like(hidden_mid)
        mystery[b0:b1] = (hidden_mid & reveal).mean(axis=-1)
        # observer -> nav cell visibility
        ci = np.clip((px / cs).astype(int), 0, G - 1)
        cj = np.clip((py / cs).astype(int), 0, G - 1)
        idx = cj * G + ci
        rows = np.broadcast_to(np.arange(b0, b1)[:, None, None], idx.shape)
        visible[rows[vis], idx[vis]] = True
    visible[np.arange(P), np.arange(P)] = True

    land_f = land.ravel()
    va = view_area.reshape(G, G)
    ref_area = np.quantile(view_area[land_f], 0.9) if land_f.any() else view_area.max()
    # prospect: best quarter of view directions, normalised
    best = np.sort(depth_far, axis=1)[:, -max(1, K // 4):].mean(axis=1) / pc.view_distance
    prospect = np.clip(0.5 * best + 0.5 * view_area / max(ref_area, 1.0), 0, 1).reshape(G, G)

    # refuge also comes from canopy and nearby cave mouths
    veg = bilinear(BIOME_VEG[world.biome], ox / cell - 0.5, oy / cell - 0.5)
    cave_near = np.zeros(P)
    for cave in world.caves:
        for e in cave.entrances:
            cave_near = np.maximum(cave_near, np.exp(-np.hypot(ox - e["x"], oy - e["y"]) / 120.0))
    refuge = np.clip(0.55 * near_block + 0.3 * veg + 0.35 * cave_near, 0, 1).reshape(G, G)

    # complexity & coherence from land-cover statistics at observer scale
    n_b = len(BIOMES)
    r_c = max(1, int(round(260.0 / cell)))
    r_h = max(1, int(round(110.0 / cell)))
    shares = np.stack([box_blur((world.biome == b).astype(float), r_c) for b in range(n_b)])
    p = np.clip(shares, 1e-9, 1)
    ent = -(p * np.log(p)).sum(axis=0) / np.log(6.0)
    local = np.stack([box_blur((world.biome == b).astype(float), r_h) for b in range(n_b)]).max(axis=0)
    rough = np.sqrt(np.maximum(box_blur(world.height ** 2, r_c) - box_blur(world.height, r_c) ** 2, 0))
    s = slope(world.height, cell)
    feat = box_blur((world.river_mask | world.cliff_mask).astype(float) + world.water.astype(float) * 0.5, r_c)
    complexity_f = np.clip(0.45 * np.clip(ent, 0, 1) + 0.35 * smoothstep(5.0, 90.0, rough) + 0.4 * np.clip(feat * 4, 0, 1), 0, 1)
    jag = box_blur(np.abs(s - box_blur(s, r_h)), r_h)
    coherence_f = np.clip(0.6 * local + 0.4 * (1.0 - smoothstep(0.03, 0.25, jag)), 0, 1)
    complexity = bilinear(complexity_f, ox / cell - 0.5, oy / cell - 0.5).reshape(G, G)
    coherence = bilinear(coherence_f, ox / cell - 0.5, oy / cell - 0.5).reshape(G, G)

    vista = smoothstep(np.quantile(va[land], 0.75) if land.any() else 0, np.quantile(va[land], 0.97) if land.any() else 1, va)
    return Perception(
        grid=G, cell=cs, ox=ox, oy=oy, oz=oz, land=land, visible=visible, view_area=va,
        prospect=prospect, refuge=refuge, mystery=mystery.reshape(G, G),
        complexity=complexity, coherence=coherence, vista=vista,
    )
