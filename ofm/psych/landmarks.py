"""Landmark salience (after Lynch, *The Image of the City*, and wayfinding research).

A good landmark is visible from far away, distinct from its surroundings and
unique. Candidates are summits, volumetric rock formations, waterfalls and
large lakes. Visibility is measured with line-of-sight tests from every
observer to the top of each candidate; salience combines visibility,
prominence and rarity. The chosen landmarks drive legibility (can a player
orient themselves?) and attract exploring agents ("weenies").
"""
from __future__ import annotations

import numpy as np

from ..grid import bilinear, sliding_max, sliding_min
from ..world import Landmark
from .perception import Perception


def _candidates(world) -> list[Landmark]:
    cell = world.cell
    h = world.ground()                  # summits of the exported terrain
    out: list[Landmark] = []
    r = max(3, int(300.0 / cell))
    peak = (h >= sliding_max(h, r) - 1e-6) & ~world.water
    prom = h - sliding_min(h, r * 2)
    jj, ii = np.nonzero(peak & (prom > 40.0))
    for j, i in zip(jj, ii):
        out.append(Landmark(id="", kind="summit", x=(i + 0.5) * cell, y=(j + 0.5) * cell,
                            z=float(h[j, i]), height=float(prom[j, i])))
    for f in world.features:
        out.append(Landmark(id="", kind=f.kind, x=f.x, y=f.y, z=f.z, height=f.height))
    for wf in world.waterfalls:
        out.append(Landmark(id="", kind="waterfall", x=wf["x"], y=wf["y"], z=wf["bottom"],
                            height=wf["drop"]))
    for lake in world.lakes:
        if lake["area_m2"] > 40000:
            out.append(Landmark(id="", kind="lake", x=lake["center"][0], y=lake["center"][1],
                                z=lake["level"], height=1.0))
    return out


def line_of_sight(vh: np.ndarray, cell: float, ox, oy, oz, tx, ty, tz, steps: int = 32) -> np.ndarray:
    """(P,) observers x (L,) targets -> bool visibility matrix (P, L)."""
    t = np.linspace(0.05, 0.94, steps)
    px = ox[:, None, None] + (tx[None, :, None] - ox[:, None, None]) * t
    py = oy[:, None, None] + (ty[None, :, None] - oy[:, None, None]) * t
    pz = oz[:, None, None] + (tz[None, :, None] - oz[:, None, None]) * t
    H = bilinear(vh, px / cell - 0.5, py / cell - 0.5)
    return (H < pz + 0.5).all(axis=-1)


def select_landmarks(world, per: Perception) -> tuple[list[Landmark], np.ndarray]:
    pc = world.cfg.psych
    cands = _candidates(world)
    G = per.grid
    if not cands:
        return [], np.zeros((G * G, 0), dtype=bool)
    vh = world.visual_height()
    tx = np.array([c.x for c in cands])
    ty = np.array([c.y for c in cands])
    top = np.array([c.z + (c.height if c.kind not in ("summit", "lake") else 0.0) for c in cands])
    tz = top + np.array([2.0 if c.kind != "lake" else 0.5 for c in cands])
    land = per.land.ravel()
    vis = np.zeros((G * G, len(cands)), dtype=bool)
    step = 512
    for b in range(0, G * G, step):
        sl = slice(b, min(G * G, b + step))
        v = line_of_sight(vh, world.cell, per.ox[sl], per.oy[sl], per.oz[sl], tx, ty, tz)
        d = np.hypot(per.ox[sl, None] - tx[None], per.oy[sl, None] - ty[None])
        size_term = np.array([max(c.height, 8.0) for c in cands])
        reach = np.minimum(pc.view_distance, 120.0 * np.sqrt(size_term) + 250.0)
        vis[sl] = v & (d < reach[None, :])
    vis_frac = vis[land].mean(axis=0) if land.any() else vis.mean(axis=0)

    kinds = [c.kind for c in cands]
    rarity = np.array([1.0 / kinds.count(k) ** 0.5 for k in kinds])
    bonus = {"spire": 1.0, "arch": 1.0, "sea_stack": 0.9, "butte": 0.8, "monolith": 0.8,
             "waterfall": 0.7, "summit": 0.6, "lake": 0.5}
    shape_bonus = np.array([bonus.get(k, 0.5) for k in kinds])
    prom = np.array([np.clip(c.height / 250.0, 0, 1) for c in cands])
    sal = np.sqrt(vis_frac) * (0.4 + 0.6 * shape_bonus) * (0.6 + 0.4 * np.maximum(prom, 0.3)) * (0.7 + 0.3 * rarity)
    for c, s, v in zip(cands, sal, vis_frac):
        c.salience = float(s)
        c.visibility = float(v)

    order = np.argsort(-sal)
    chosen: list[int] = []
    for k in order:
        if len(chosen) >= pc.landmarks:
            break
        if sal[k] <= 0.02:
            break
        if any(np.hypot(tx[k] - tx[j], ty[k] - ty[j]) < 350.0 for j in chosen):
            continue
        chosen.append(int(k))
    landmarks = []
    for n, k in enumerate(chosen):
        c = cands[k]
        c.id = f"landmark_{n}"
        landmarks.append(c)
    return landmarks, vis[:, chosen]
