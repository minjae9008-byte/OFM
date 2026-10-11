"""Landform processes on the export-resolution heightfield.

The simulation grid (16 m) cannot hold the small landforms a player walks
over. After Catmull-Rom upsampling and geology-aware detail, these processes
run on the export grid, in the order material moves through a landscape:

  * Fine droplet erosion: the droplet model of ``physics.erosion`` re-run at
    export resolution with a narrow brush, so rills and small gullies form on
    soft slopes (erosion scales with 1 - hardness of the exposed stratum).
  * Scree (talus) aprons: rockfall from steep hard faces piles up at the
    angle of repose (~34 deg; Statham 1976, Carson 1977). Fall sorting sends
    coarse blocks further, so aprons are concave: steep near the face and
    flattening into the valley floor. The apron is the upper envelope of
    three cones of decreasing slope seeded on the face, its height limited
    by the face relief and the reach.
  * Floodplains: overbank deposition builds a flat alluvial surface at about
    bankfull stage, ~0.5-1.5 m above mean water (Leopold & Wolman 1957), on
    low-gradient reaches only; the channel stays incised below it.
  * Beaches: soft, low coasts relax to the equilibrium shoreface profile
    h = A x^(2/3) (Bruun 1954, Dean 1991; A ~ 0.1 m^(1/3) for medium sand)
    below water and a swash berm at the foreshore slope above it.
  * Shore platforms: hard sea cliffs retreat behind a gently sloping
    wave-cut platform just below sea level (Sunamura 1992, Trenhaile 1987);
    the cliff itself is left untouched.

Hard constraints are enforced last: lake shores are not cut below their
lake's level, river channels stay open below their water surface, the sea
floor stays below sea level (lake beds are clamped by ``amplify``). Every
process can be switched off in ``detail.hires``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from ..grid import NEIGHBORS8, bilinear, blur, chamfer_distance, gradient, shifted, sliding_max, smoothstep
from ..noise import derive_seed
from ..physics.erosion import hydraulic_erosion

# 16-neighbourhood (8 + knight moves) for nearly isotropic cone propagation.
_OFFS16 = NEIGHBORS8 + [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


# -- rivers on the export grid -------------------------------------------------
@dataclass
class RiverField:
    """Attributes of the nearest river centreline point for every export cell."""
    dist: np.ndarray        # metres to the centreline (inf beyond reach)
    water: np.ndarray       # water-surface elevation (m)
    half_width: np.ndarray  # channel half-width (m)
    channel_hw: np.ndarray  # resolvable channel half-width (m, >= one export cell)
    depth: np.ndarray       # water depth at the thalweg (m)
    gradient: np.ndarray    # along-channel water-surface gradient (m/m)


def _river_samples(world, step: float, glen: float, extend: float = 0.0,
                   vertices: bool = False) -> np.ndarray | None:
    """Dense samples (x, y, z_water, width, depth, gradient) along all river polylines
    (``vertices``: the polyline vertices themselves are samples too)."""
    hc = world.cfg.hydrology
    rows = []
    for r in world.rivers:
        p = np.asarray(r["points"], dtype=np.float64)
        wid = np.asarray(r["width"], dtype=np.float64)
        q = np.asarray(r["discharge"], dtype=np.float64)
        if len(p) < 2:
            continue
        # run the mouth on (into the receiving water or river, or past the map edge) so
        # the channel does not end in a closed pit
        d = p[-1, :2] - p[-2, :2]
        if extend > 0.0 and np.hypot(*d) > 1e-9:
            ext = p[-1].copy()
            ext[:2] += d / np.hypot(*d) * extend
            p, wid, q = np.vstack([p, ext]), np.append(wid, wid[-1]), np.append(q, q[-1])
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])))])
        if s[-1] <= 0.0:
            continue
        t = np.linspace(0.0, s[-1], int(np.ceil(s[-1] / step)) + 1)
        if vertices:
            t = np.union1d(t, s)
        z = np.minimum.accumulate(np.interp(t, s, p[:, 2]))       # never uphill
        wid = np.interp(t, s, wid)
        q = np.interp(t, s, q)
        # water surface sits 0.8 x channel depth above the bed (see hydrology._trace_rivers)
        depth = 0.8 * hc.channel_depth_k * np.maximum(q, 0.0) ** 0.4
        grad = (np.interp(t - glen, s, p[:, 2]) - np.interp(t + glen, s, p[:, 2])) / (2.0 * glen)
        rows.append(np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1]), z, wid, depth,
                              np.maximum(grad, 0.0)], axis=1))
    return np.concatenate(rows) if rows else None


def river_field(world, n: int, spacing: float, reach: float) -> RiverField | None:
    """Nearest-centreline attributes by jump flooding (Rong & Tan 2006) over dense
    polyline samples; only a box around the rivers is processed."""
    smp = _river_samples(world, spacing * 0.25, spacing * 3.0, 2.0 * spacing)
    if smp is None:
        return None
    sx, sy = smp[:, 0], smp[:, 1]
    # seed the four nodes around every sample (each keeps its closest sample), so nodes
    # beside a diagonal reach see the samples between their neighbours
    fi = np.clip(np.floor(sx / spacing).astype(np.int64), 0, n - 2)
    fj = np.clip(np.floor(sy / spacing).astype(np.int64), 0, n - 2)
    k = np.arange(len(sx))
    ci = np.concatenate([fi, fi + 1, fi, fi + 1])
    cj = np.concatenate([fj, fj, fj + 1, fj + 1])
    kk = np.concatenate([k, k, k, k])
    d0 = np.hypot(sx[kk] - ci * spacing, sy[kk] - cj * spacing)
    cell = cj * n + ci
    order = np.lexsort((d0, cell))                       # per node, the closest sample first
    pick = order[np.unique(cell[order], return_index=True)[1]]
    ci, cj, first = ci[pick], cj[pick], kk[pick]
    r = int(np.ceil(reach / spacing)) + 1
    j0, j1 = max(int(cj.min()) - r, 0), min(int(cj.max()) + r + 1, n)
    i0, i1 = max(int(ci.min()) - r, 0), min(int(ci.max()) + r + 1, n)
    seed = np.full((j1 - j0, i1 - i0), -1, dtype=np.int64)
    seed[cj - j0, ci - i0] = first
    yy, xx = np.mgrid[j0:j1, i0:i1].astype(np.float64) * spacing
    sxp = np.append(sx, np.inf)                          # index -1 -> infinitely far
    syp = np.append(sy, np.inf)
    best = np.hypot(xx - sxp[seed], yy - syp[seed])
    step = 1 << max(int(np.ceil(np.log2(max(r, 1)))) - 1, 0)
    steps = []
    while step >= 1:
        steps.append(step)
        step //= 2
    for k in steps + [1]:                                # JFA+1: one extra unit pass
        for dy, dx in NEIGHBORS8:
            cand = shifted(seed, dy * k, dx * k, fill=-1)
            d = np.hypot(xx - sxp[cand], yy - syp[cand])
            better = d < best
            seed = np.where(better, cand, seed)
            best = np.where(better, d, best)
    near = best <= reach

    def full(col: np.ndarray, fill: float) -> np.ndarray:
        out = np.full((n, n), fill)
        out[j0:j1, i0:i1] = np.where(near, col[np.maximum(seed, 0)], fill)
        return out

    dist = np.full((n, n), np.inf)
    dist[j0:j1, i0:i1] = np.where(near, best, np.inf)
    hw = np.maximum(full(smp[:, 3] * 0.5, 0.0), 0.5)
    return RiverField(dist=dist, water=full(smp[:, 2], 0.0), half_width=hw,
                      channel_hw=np.maximum(hw, 1.25 * spacing), depth=np.maximum(full(smp[:, 4], 0.0), 0.3),
                      gradient=full(smp[:, 5], 0.0))


def floodplain_half_width(rf: RiverField, hc, spacing: float) -> np.ndarray:
    """Floodplain half-width: floodplain_width_k channel widths beyond each bank (at
    least 1.5 export cells beyond the resolvable channel)."""
    return np.maximum(rf.half_width * (1.0 + 2.0 * hc.floodplain_width_k), rf.channel_hw + 1.5 * spacing)


def river_protection(rf: RiverField | None, spacing: float, shape) -> np.ndarray:
    """1 on channels and their banks, fading out over two cells (no erosion fill / scree there)."""
    if rf is None:
        return np.zeros(shape)
    inner = rf.channel_hw + spacing
    return smoothstep(inner + 2.0 * spacing, inner, rf.dist)


# -- 1. fine droplet erosion -----------------------------------------------------
def fine_erosion(h: np.ndarray, fold: np.ndarray, strata, protect: np.ndarray, spacing: float,
                 cfg) -> np.ndarray:
    """Rills and gullies from droplets on the export grid (hardness of the exposed stratum
    at each cell), blended in by ``erosion_strength`` except where ``protect`` is 1."""
    hc = cfg.detail.hires
    if hc.droplets_per_cell <= 0.0 or hc.erosion_strength <= 0.0:
        return h
    ec = dataclasses.replace(cfg.erosion, droplets_per_cell=hc.droplets_per_cell, brush_radius=1,
                             max_steps=hc.erosion_steps, batch=8192, erode_speed=0.08,
                             deposit_speed=0.1, capacity=2.0, inertia=0.1)
    rng = np.random.default_rng(derive_seed(cfg.world.seed, "hires-erosion"))
    eroded = hydraulic_erosion(h, lambda z: strata.hardness(z, fold), ec, rng)
    dh = eroded - h
    # keep only what the simulation grid cannot hold: the high-passed change (rills,
    # gullies, small fans); the broad redistribution was already done at 16 m
    dh = dh - blur(dh, 2.5)
    dh = 5.0 * np.tanh(dh / 5.0)                         # soft limit (m)
    # rills need runoff speed: near-flat ground only gets faint sheet-wash texture
    gx, gy = gradient(h, spacing)
    w = 0.3 + 0.7 * smoothstep(0.05, 0.25, np.hypot(gx, gy))
    return h + hc.erosion_strength * w * (1.0 - protect) * dh


# -- 2. scree aprons -----------------------------------------------------------------
def _cone_envelope(apex: np.ndarray, base: np.ndarray, tan_slope: float, spacing: float,
                   iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Upper envelope of truncated cones: max over seeds s of apex(s) - tan_slope * |x - s|
    while that stays above base(s) (chamfer metric, 16 neighbours). Returns the envelope
    and the base of the winning cone."""
    out, fl = apex, base
    drops = [tan_slope * spacing * np.hypot(dy, dx) for dy, dx in _OFFS16]
    for _ in range(iters):
        new, nfl = out.copy(), fl.copy()
        for (dy, dx), drop in zip(_OFFS16, drops):
            cand = shifted(out, dy, dx, fill=-np.inf) - drop
            better = cand > new
            new = np.where(better, cand, new)
            nfl = np.where(better, shifted(fl, dy, dx, fill=np.inf), nfl)
        new = np.where(new >= nfl, new, -np.inf)          # each cone ends at its base
        if np.array_equal(new, out):
            break
        out, fl = new, nfl
    return out, fl


def _face_foot(h: np.ndarray, face: np.ndarray, spacing: float, max_steps: int) -> np.ndarray:
    """Elevation of the first non-face cell reached by steepest descent from each face
    cell (pointer jumping along D8 receivers); h itself elsewhere."""
    ny, nx = h.shape
    idx = np.arange(h.size).reshape(h.shape)
    rec = idx.copy()
    best = np.zeros(h.shape)
    for dy, dx in NEIGHBORS8:
        drop = (h - shifted(h, dy, dx, fill=np.inf)) / (spacing * np.hypot(dy, dx))
        better = drop > best
        best = np.where(better, drop, best)
        rec = np.where(better, idx + dy * nx + dx, rec)
    rec = np.where(face, rec, idx).reshape(-1)
    for _ in range(int(np.ceil(np.log2(max(max_steps, 2))))):
        rec = rec[rec]
    return h.reshape(-1)[rec].reshape(h.shape)


def scree_aprons(h: np.ndarray, hard: np.ndarray, protect: np.ndarray, spacing: float, hc) -> np.ndarray:
    """Concave talus aprons below steep hard faces (only raises ground; not where
    ``protect`` is 1)."""
    gx, gy = gradient(h, spacing)
    s_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    steep = s_deg > hc.scree_face_deg
    face = steep & (hard > 0.45)                         # hard free faces shed blocks
    if not face.any() or hc.scree_strength <= 0.0:
        return h
    t1 = np.tan(np.radians(hc.scree_repose_deg))
    reach = max(hc.scree_reach_m, spacing)
    r = int(np.ceil(reach / spacing))
    jj, ii = np.nonzero(face)
    j0, j1 = max(jj.min() - 2 * r, 0), min(jj.max() + 2 * r + 1, h.shape[0])
    i0, i1 = max(ii.min() - 2 * r, 0), min(ii.max() + 2 * r + 1, h.shape[1])
    hb, fb, sb = h[j0:j1, i0:i1], face[j0:j1, i0:i1], steep[j0:j1, i0:i1]
    # debris lands at the base of the face: the apron top sits a share of the face
    # relief above its foot, capped so that the concave apron (run ~1.8 H /
    # tan(repose)) ends inside the reach
    foot = _face_foot(hb, sb, spacing, 8 * r)
    relief = sliding_max(hb, r) - foot
    H = np.minimum(np.minimum(hb - foot, 0.5 * hc.scree_strength * relief), reach * t1 / 1.9)
    H = np.where(fb, np.maximum(H, 0.0), 0.0)
    apex = np.where(fb, foot + H, -np.inf)
    foot = np.where(fb, foot, np.inf)
    T = np.zeros_like(hb)
    for drop, tan_k in ((0.0, t1), (0.25, 0.55 * t1), (0.55, 0.25 * t1)):
        S, base = _cone_envelope(apex - drop * H, foot, tan_k, spacing, r + 1)
        # a cone only covers ground at or above its own foot (debris below it fell further)
        T = np.maximum(T, np.where(hb >= base - 0.5, S - hb, 0.0))
    T = T * smoothstep(0.0, 1.2, T)                      # feather the toe into the floor
    out = h.copy()
    out[j0:j1, i0:i1] = hb + T * (1.0 - protect[j0:j1, i0:i1])
    return out


# -- 3. floodplains and channels --------------------------------------------------
def floodplains(h: np.ndarray, rf: RiverField, water_w: np.ndarray, sea: float, spacing: float,
                hc) -> np.ndarray:
    """Pull the low valley floor beside low-gradient reaches towards bankfull level and
    shape the banks; ``water_w`` (lakes / sea) is left alone."""
    near = np.isfinite(rf.dist)
    if not near.any():
        return h
    W = floodplain_half_width(rf, hc, spacing)
    hw = rf.channel_hw
    base = np.maximum(rf.water, sea)
    rise = np.clip(0.5 + 0.5 * rf.depth, 0.5, 1.5)          # bankfull stage above mean water
    level = base + rise
    wd = smoothstep(W, hw + 0.5 * (W - hw), rf.dist)       # full near the channel, 0 at the edge
    g = hc.floodplain_max_gradient
    wg = smoothstep(g, 0.3 * g, rf.gradient)               # only low-gradient reaches build floodplains
    hmax = hc.floodplain_max_height_m
    wh = smoothstep(hmax, 0.5 * hmax, h - base)            # valley walls are not floodplain
    w = np.clip(hc.floodplain_flatten, 0.0, 1.0) * wd * wg * wh * (1.0 - water_w)
    out = h + np.where(near, w, 0.0) * (level - h)
    # banks: rise from the waterline to the floodplain over one cell beyond the channel
    # (alluvial reaches only; steep reaches, falls and valley walls keep their banks)
    bank = rf.water + (level - rf.water) * smoothstep(hw, hw + spacing, rf.dist)
    in_bank = near & (rf.dist < hw + spacing) & (water_w < 0.5)
    return np.where(in_bank, out - wg * wh * np.maximum(out - bank, 0.0), out)


def open_channels(h: np.ndarray, rf: RiverField | None, water_w: np.ndarray, spacing: float) -> np.ndarray:
    """Channel below the water surface with a flat thalweg core and smooth sides.

    Nodes take the bed of their nearest centreline point (the water surface is locally
    planar, so the bilinear terrain follows it); on land the core (inner half of the
    channel) is set to the bed itself, so the thalweg descends with the water surface
    and leaves no pits, and the sides are clamped to [bed - depth, bed]."""
    if rf is None:
        return h
    hw = rf.channel_hw
    inside = rf.dist < hw
    if not inside.any():
        return h
    u = np.clip(rf.dist / hw, 0.0, 1.0)
    bed = rf.water - 0.15 - rf.depth * (1.0 - smoothstep(0.5, 1.0, u))
    land = water_w < 0.5
    floor = np.where(land, np.minimum(rf.water - 2.0 * rf.depth, bed), -np.inf)
    floor = np.where(land & (u <= 0.5), bed, floor)
    out = np.where(inside, np.clip(h, floor, bed), h)
    # a channel narrower than ~2 cells beads along diagonals: smooth the corridor once
    corridor = smoothstep(hw + 1.5 * spacing, hw, rf.dist) * land
    out = out + corridor * (blur(out, 0.8) - out)
    return np.where(inside, np.clip(out, floor, bed), out)


def clear_ribbons(h: np.ndarray, samples: np.ndarray | None, spacing: float,
                  clearance: float = 0.1, step: float = 0.0, iterations: int = 8,
                  frozen: np.ndarray | None = None) -> np.ndarray:
    """Lower the four nodes around any centreline sample whose bilinear terrain would
    reach the water surface, so the river ribbon is never blocked.

    Between two samples ``step`` apart the water falls by up to gradient * step, so the
    clearance grows with the local water gradient (column 5 of the samples); lowering
    nodes can lift nothing, but neighbouring samples share nodes, so it iterates. Nodes marked
    ``frozen`` (protected terrain) are never lowered."""
    if samples is None or not len(samples):
        return h
    n = h.shape[0]
    out = h.copy()
    u = np.clip(samples[:, 0] / spacing, 0.0, n - 1.0)
    v = np.clip(samples[:, 1] / spacing, 0.0, n - 1.0)
    i0 = np.minimum(np.floor(u).astype(np.int64), n - 2)
    j0 = np.minimum(np.floor(v).astype(np.int64), n - 2)
    clear = clearance
    if step > 0.0 and samples.shape[1] > 5:
        clear = clearance + 0.5 * step * np.maximum(samples[:, 5], 0.0)
    for _ in range(max(1, int(iterations))):
        excess = bilinear(out, u, v) - (samples[:, 2] - clear)
        hit = excess > 0.0
        if not hit.any():
            break
        for dj in (0, 1):
            for di in (0, 1):
                jj, ii = j0[hit] + dj, i0[hit] + di
                low = out[jj, ii] - excess[hit]
                if frozen is not None:
                    low = np.where(frozen[jj, ii], out[jj, ii], low)
                np.minimum.at(out, (jj, ii), low)
    return out


# -- 4. coasts ------------------------------------------------------------------
def _spread(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: smooth extension of values known on ``mask``."""
    m = mask.astype(np.float64)
    num = blur(values * m, sigma)
    den = blur(m, sigma)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def shape_coasts(h: np.ndarray, ocean: np.ndarray, cliff_w: np.ndarray, hard: np.ndarray,
                 sea: float, spacing: float, hc) -> np.ndarray:
    """Beach / shoreface profiles on soft, low coasts and wave-cut platforms seaward of
    sea cliffs (``cliff_w`` 0..1 on land). Land stays above and sea below ``sea``; cliff
    land cells are never changed."""
    if not ocean.any() or ocean.all():
        return h
    W = max(hc.beach_width_m, spacing)
    Wp = max(hc.platform_width_m, spacing)
    rmax = int(np.ceil(max(W, Wp) / spacing)) + 1
    land = ~ocean
    d_on = chamfer_distance(ocean, max_dist=rmax)
    d_off = chamfer_distance(land, max_dist=rmax)
    y = np.maximum(d_on - 0.5, 0.0) * spacing                # onshore distance (land cells)
    x = np.maximum(d_off - 0.5, 0.0) * spacing               # offshore distance (ocean cells)
    shore = land & (d_on <= 1.5)
    sig = W / spacing * 0.6
    # coast type, measured along the shoreline and spread smoothly to both sides
    relief = sliding_max(h, max(1, int(round(W / spacing)))) - sea
    berm = hc.beach_berm_m
    soft = smoothstep(0.8, 0.5, hard)
    low = smoothstep(4.0 * berm + 20.0, berm + 6.0, relief)
    beach_w = _spread(soft * low * (1.0 - cliff_w), shore, sig) if hc.beaches else 0.0
    plat_w = _spread(cliff_w * (0.4 + 0.6 * smoothstep(0.3, 0.6, hard)), shore, sig) if hc.wave_platform else 0.0
    out = h.copy()
    if hc.beaches:
        # swash berm on land: sorted sand is smooth (no rills / soil texture), and the
        # foreshore is cut down (never built up) to sea + tan(beta) y, capped at the berm
        land_t = sea + np.minimum(np.tan(np.radians(hc.beach_slope_deg)) * y, berm)
        wy = beach_w * smoothstep(W, 0.4 * W, y)
        smooth = np.maximum(blur(out, 1.2), sea + 0.05)
        out = np.where(land, out + wy * (smooth - out), out)
        out = np.where(land, out - wy * np.maximum(out - land_t, 0.0), out)
        # Dean profile below water, A fitted to the local shoreface (so the profile meets
        # the original sea floor), behind a planar swash zone at the beach slope where the
        # Dean slope would be steeper (x < x0; Bodge 1992)
        band = ocean & (x <= W)
        a0 = (sea - h) / np.maximum(x, spacing * 0.5) ** (2.0 / 3.0)
        a_loc = np.clip(_spread(a0, band, sig), hc.beach_dean_a, 4.0 * hc.beach_dean_a)
        tb = np.tan(np.radians(hc.beach_slope_deg))
        x0 = (2.0 * a_loc / (3.0 * tb)) ** 3
        xs = np.maximum(x, spacing * 0.25)
        depth = np.where(xs <= x0, tb * xs, tb * x0 + a_loc * (xs ** (2.0 / 3.0) - x0 ** (2.0 / 3.0)))
        sea_t = sea - depth
        wx = beach_w * smoothstep(W, 0.4 * W, x)
        out = np.where(ocean, out + wx * (sea_t - out), out)
    if hc.wave_platform:
        plat_t = sea - hc.platform_depth_m - np.tan(np.radians(hc.platform_slope_deg)) * x
        wx = plat_w * smoothstep(Wp, 0.55 * Wp, x)
        out = np.where(ocean, out + wx * (plat_t - out), out)
    return out


# -- driver -----------------------------------------------------------------------
def lake_shore_floor(world, base: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray | None:
    """Lowest the export grid may go around lakes: the interpolated simulation surface,
    capped at the lake level. A lake's shore is its dam; rills, beaches or detail cutting
    it below the level would drain the lake on the detail grid (the water sheet would hang
    over the breach). -inf away from lakes and on the lake beds themselves."""
    if world.lake_mask is None or not world.lake_mask.any():
        return None
    lv = np.where(world.lake_mask, world.lake_level, -np.inf)
    rim = lv.copy()
    for dy, dx in NEIGHBORS8:
        rim = np.maximum(rim, shifted(lv, dy, dx, -np.inf))
    size = world.size
    ci = np.clip(np.floor(U + 0.5).astype(np.int64), 0, size - 1)
    cj = np.clip(np.floor(V + 0.5).astype(np.int64), 0, size - 1)
    bed = bilinear(world.lake_mask.astype(np.float64), U, V) > 0.5
    return np.where(bed, -np.inf, np.minimum(base, rim[cj, ci]))


def refine(world, h: np.ndarray, fold: np.ndarray, U: np.ndarray, V: np.ndarray,
           spacing: float, base: np.ndarray | None = None) -> np.ndarray:
    """Run the enabled export-resolution processes on ``h`` and enforce water constraints
    (``base``: the interpolated simulation surface, which keeps lake shores closed)."""
    cfg = world.cfg
    hc = cfg.detail.hires
    sea = cfg.world.sea_level
    n = h.shape[0]

    ocean_soft = bilinear(world.ocean.astype(np.float64), U, V) if world.ocean is not None else np.zeros_like(h)
    water_soft = bilinear(world.water.astype(np.float64), U, V) if world.ocean is not None else ocean_soft
    ocean = ((h < sea) & (ocean_soft > 0.02)) | (ocean_soft > 0.5)
    dry = ~ocean & (h >= sea)                    # land must not sink below the sea plane
    floor_dry = np.minimum(h, sea + 0.05)
    water_w = smoothstep(0.05, 0.5, water_soft)

    rf = None
    if world.rivers:
        hw_max = max([0.5 * max(r["width"]) for r in world.rivers if len(r["width"])] or [0.5])
        reach = max(hw_max, spacing) + 3.5 * spacing             # channel + protected banks
        if hc.floodplains:
            reach = max(reach, hw_max * (1.0 + 2.0 * hc.floodplain_width_k) + spacing)
        rf = river_field(world, n, spacing, reach)
    protect = np.maximum(river_protection(rf, spacing, h.shape), water_w)

    if hc.erosion:
        h = fine_erosion(h, fold, world.strata, protect, spacing, cfg)
    if hc.scree:
        hard = world.strata.hardness(h, fold)
        coast_zone = smoothstep(4.0 * spacing, 2.0 * spacing,
                                chamfer_distance(ocean, max_dist=5) * spacing) if ocean.any() else 0.0
        h = scree_aprons(h, hard, np.maximum(protect, coast_zone), spacing, hc)
    if hc.floodplains and rf is not None:
        h = floodplains(h, rf, water_w, sea, spacing, hc)
    if (hc.beaches or hc.wave_platform) and ocean.any():
        cliff = np.zeros_like(h)
        if world.coast_cliff is not None:
            cliff = smoothstep(0.1, 0.5, bilinear(world.coast_cliff.astype(np.float64), U, V))
        h = shape_coasts(h, ocean, cliff, world.strata.hardness(h, fold), sea, spacing, hc)

    h = np.where(dry, np.maximum(h, floor_dry), h)
    shore = lake_shore_floor(world, h if base is None else base, U, V)
    if shore is not None:
        h = np.where(ocean, h, np.maximum(h, shore))
    h = open_channels(h, rf, water_w, spacing)
    if rf is not None:
        h = clear_ribbons(h, _river_samples(world, spacing * 0.25, spacing, vertices=True), spacing,
                          step=spacing * 0.25)
    return np.where(ocean, np.minimum(h, sea - 0.05), h)
