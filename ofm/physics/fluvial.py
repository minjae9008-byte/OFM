"""Stream-power landscape evolution (fluvial erosion).

Model
-----
Large-scale relief is shaped by rivers: bedrock incision follows the
stream-power law (Howard 1994; Whipple & Tucker 1999)

    dh/dt = U - K A^m S^n + G / A * Qs + kappa * laplacian(h)

with uplift ``U``, drainage area ``A``, slope towards the receiver ``S``,
erodibility ``K`` (lowered by rock hardness, so hard layers stand out as
ridges, spurs and benches), sediment flux ``Qs`` and hillslope diffusivity
``kappa``. Each time step

1. routes flow with Priority-Flood (Barnes et al. 2014, from
   ``hydrology.priority_flood``; sinks = ocean connected to the border) and a
   single downslope receiver per cell on the filled surface. The receiver is
   drawn among the strictly lower neighbours with probability ~ slope^p
   (stochastic D8, Fairfield & Leymarie 1991): averaged over time steps this
   approximates multiple-flow-direction routing on planar hillslopes, removes
   the straight D8 grid lines and lets channels wander and merge like
   Scheidegger's random networks, while incised channels (one much steeper
   exit) route deterministically. Only strictly lower cells qualify, so the
   flood (pop) order stays a valid stack: every receiver precedes its donors;
2. accumulates drainage area over the reversed stack;
3. solves the stream-power term implicitly along the stack from the outlets
   upwards (Braun & Willett 2013): for n = 1
   ``h_i <- (h_i + dt U_i + F_i h_r) / (1 + F_i)``, ``F_i = K_i A_i^m dt / d_i``,
   which is unconditionally stable; n != 1 uses a per-node Newton iteration.
   Nodes are grouped by their depth in the receiver tree, so every group is
   independent and solved with one vectorised update;
4. adds sediment deposition (Yuan et al. 2019): ``D = G Qs / A`` with ``Qs``
   the net sediment flux (upstream integral of incision minus deposition). It
   is solved exactly for the step's incision in one donor-first pass over the
   receiver tree and fed into the next implicit solve as a source term (so it
   cannot oscillate, whatever ``G``). Locally D is about ``G/(1+G)`` of the
   incision, which lowers net incision everywhere; where the flux from
   upstream exceeds the local incision (plains below a mountain front, wide
   low-gradient valley floors, lake inlets) the surface aggrades, while fast-
   incising trunk channels keep a smaller share of their incision. Channel migration and avulsion spread
   the alluvium over the valley floor (a mass-conserving scatter onto cells
   within a few metres of the local low point);
5. applies explicit hillslope diffusion with stable sub-steps, then threshold
   hillslopes (landsliding, Burbank et al. 1996; Montgomery & Brandon 2002):
   slopes steeper than a critical gradient that grows with rock hardness are
   relaxed towards it (never below the input's own slope, so tectonic relief
   survives); the debris is assumed to leave with the rivers;
6. returns a fraction of the regionally averaged unloading as flexural-
   isostatic rebound (Molnar & England 1990): valleys deepen while summits
   keep their height and the mean land elevation stays roughly stable.

Ocean cells are base level (held at sea level, never eroded), so coastlines
and the land fraction placed by tectonics are preserved. Most time steps run
on a grid ``fluvial.resolution_divisor`` times coarser (cubic resampling) and
only the *change* is added back to the full-resolution input, so fine detail
survives; the last ``fluvial.fine_iterations`` steps run on the full grid to
sharpen channels and start first-order tributaries. The cumulative change is
capped by ``max_change`` and blended with ``strength``. Uplift is
``uplift_rate`` times the normalised orogenic uplift of the tectonics stage.

CONTRACT (used by pipeline.generate_base):

    stream_power_erosion(height, uplift, strata, cfg) -> np.ndarray

* ``height``: (N, N) float64 metres after tectonics (sea level = cfg.world.sea_level).
* ``uplift``: (N, N) metres of orogenic uplift from tectonics (>= 0); scales the uplift rate.
* ``strata``: geology.strata.Strata; ``strata.hardness(z)`` gives rock hardness 0..1 at
  elevations z on the simulation grid (use it to scale erodibility).
* ``cfg``: OFMConfig; parameters in ``cfg.fluvial`` (FluvialConfig), cell size in
  ``cfg.world.cell_size``.
Returns a new array of the same shape; must be deterministic for a given input.
"""
from __future__ import annotations

import numpy as np

from ..grid import NEIGHBORS8, blur, border_mask, resample, sliding_min
from ..noise import derive_seed
from .hydrology import accumulate, ocean_mask, priority_flood

MIN_LAND = 0.5          # metres: eroded land never sinks below sea level + this (coastline stays put)
FLOODPLAIN_RELIEF = 4.0  # metres: alluvium spreads onto cells this close to the local valley floor
INPUT_SLOPE_KEEP = 0.9   # landslide threshold >= this fraction of the input slope (keeps tectonic relief)
_OFFS_DIST = np.array([np.hypot(dy, dx) for dy, dx in NEIGHBORS8])


# -- flow routing ------------------------------------------------------------
def _receivers(filled: np.ndarray, flood_rec: np.ndarray, spacing: float,
               rng: np.random.Generator | None = None, exponent: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Downslope receiver on the filled surface and the distance to it.

    Without ``rng`` (or for a very large ``exponent``) this is steepest descent
    (D8). Otherwise the receiver is drawn among the strictly lower neighbours
    with probability ~ slope^exponent. Flats (filled depressions) keep the
    Priority-Flood receiver; seeds (sinks, border) stay outlets (-1).
    """
    hgt, wid = filled.shape
    pad = np.pad(filled, 1, mode="constant", constant_values=np.inf)
    w = np.empty((8,) + filled.shape)
    for k, (dy, dx) in enumerate(NEIGHBORS8):
        nb = pad[1 + dy:1 + dy + hgt, 1 + dx:1 + dx + wid]
        w[k] = np.maximum((filled - nb) / (spacing * _OFFS_DIST[k]), 0.0)
    has = w.max(axis=0) > 0.0
    if rng is None or exponent >= 50.0:
        k_sel = np.argmax(w, axis=0)
    else:
        cum = np.cumsum(w ** exponent, axis=0)
        u = rng.random(filled.shape) * cum[-1]
        k_sel = np.minimum((cum < u[None]).sum(axis=0), 7)
        # Guard against round-off picking a zero-weight (non-lower) neighbour.
        k_sel = np.where(np.take_along_axis(w, k_sel[None], 0)[0] > 0.0, k_sel, np.argmax(w, axis=0))
    flat_rec = flood_rec.reshape(-1)
    idx = np.arange(filled.size)
    offs = np.array([dy * wid + dx for dy, dx in NEIGHBORS8])
    k_flat = k_sel.reshape(-1)
    rec = np.where(has.reshape(-1), idx + offs[k_flat], flat_rec)
    rec = np.where(flat_rec < 0, -1, rec)        # sinks and border stay outlets
    r = np.maximum(rec, 0)
    dist = spacing * np.hypot(r // wid - idx // wid, r % wid - idx % wid)
    return rec, np.where(rec >= 0, np.maximum(dist, spacing), spacing)


def flow_routing(height: np.ndarray, sinks: np.ndarray, spacing: float,
                 rng: np.random.Generator | None = None, exponent: float = 1.0):
    """Receivers, stack order, drainage area (m^2) and receiver distance (m).

    Returns ``(receiver, order, area, dist)`` with flat arrays; ``order`` lists
    outlets first, so every receiver precedes its donors.
    """
    filled, flood_rec, order = priority_flood(height, sinks)
    rec, dist = _receivers(filled, flood_rec, spacing, rng, exponent)
    area = accumulate(order, rec, np.full(height.size, spacing * spacing))
    return rec, order, area, dist


def _levels(order: np.ndarray, rec: np.ndarray) -> list[np.ndarray]:
    """Group nodes by depth in the receiver tree (depth >= 1), outlets excluded."""
    r = rec.tolist()
    depth = [0] * len(r)
    for c in order.tolist():
        p = r[c]
        if p >= 0:
            depth[c] = depth[p] + 1
    depth_a = np.asarray(depth, dtype=np.int64)
    idx = np.argsort(depth_a, kind="stable")
    bounds = np.cumsum(np.bincount(depth_a))
    return [idx[bounds[k - 1]:bounds[k]] for k in range(1, len(bounds))]


# -- implicit stream-power solve ---------------------------------------------
def _solve_implicit(h_src: np.ndarray, rec: np.ndarray, levels: list[np.ndarray], F: np.ndarray,
                    n: float) -> np.ndarray:
    """Braun & Willett (2013) solve of h - h_src + F (h - h_rec)^n = 0 in stack order.

    ``h_src`` already contains the explicit sources (old height + uplift +
    deposition). Nodes below their receiver (pits) receive no incision.
    """
    h = h_src.copy()
    linear = abs(n - 1.0) < 1e-9
    for g in levels:
        hr = h[rec[g]]
        hs = h_src[g]
        x0 = hs - hr
        Fg = F[g]
        if linear:
            x = x0 / (1.0 + Fg)
        else:
            xp = np.maximum(x0, 1e-12)
            x = xp.copy()
            for _ in range(30):     # Newton on x = h - h_rec, bracketed in (0, x0]
                f = x - xp + Fg * x ** n
                fp = 1.0 + n * Fg * x ** (n - 1.0)
                nx = np.clip(x - f / fp, 1e-12 * xp, xp)
                done = np.max(np.abs(nx - x), initial=0.0) < 1e-4
                x = nx
                if done:
                    break
        h[g] = np.where(x0 > 0, hr + x, hs)
    return h


def _deposition(ero: np.ndarray, rec: np.ndarray, levels: list[np.ndarray], area: np.ndarray,
                cell_area: float, G: float) -> np.ndarray:
    """Deposition rate D (m/yr) of the Yuan et al. (2019) model for incision rates ``ero``.

    ``D_i = G / A_i * (sum over the donors' outgoing flux + E_i a - D_i a)``: the outgoing
    flux of each node (its incoming flux plus its own incision minus what it deposits)
    feeds its receiver. Processing the receiver tree from the deepest level up makes each
    level one vectorised update, and solving for ``D_i`` keeps it below the available
    flux, so the outgoing flux is never negative."""
    D = np.zeros_like(ero)
    inflow = np.zeros_like(ero)
    for g in reversed(levels):
        tot = inflow[g] + ero[g] * cell_area
        D[g] = G * tot / (area[g] + G * cell_area)
        np.add.at(inflow, rec[g], tot - D[g] * cell_area)
    return D


def _diffuse(h: np.ndarray, fixed: np.ndarray, alpha: float) -> np.ndarray:
    """Explicit linear hillslope diffusion; ``alpha = kappa dt / dx^2`` split into stable sub-steps."""
    if alpha <= 0:
        return h
    steps = int(np.ceil(alpha / 0.2))
    a = alpha / steps
    free = ~fixed
    for _ in range(steps):
        p = np.pad(h, 1, mode="edge")
        lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * h
        h = np.where(free, h + a * lap, h)
    return h


def _evolve(h: np.ndarray, ocean: np.ndarray, uplift: np.ndarray, fold, strata, fc,
            spacing: float, steps: int, lo: np.ndarray, hi: np.ndarray, floor: np.ndarray,
            rng: np.random.Generator, s_input: np.ndarray) -> np.ndarray:
    """Run ``steps`` landscape-evolution time steps on one grid (see module docstring).

    ``uplift`` is the uplift rate (m/yr) and ``lo``/``hi``/``floor`` bound the
    surface. ``s_input`` is the steepest-descent gradient of the uneroded
    input: the landslide threshold never drops much below it, so incision
    cannot steepen valley walls past the critical slope while the tectonic
    relief is kept.
    """
    shape = h.shape
    fixed = ocean | border_mask(shape)
    fixed_f = fixed.reshape(-1)
    land = ~ocean
    movable = ~fixed
    dt = float(fc.dt_years)
    cell_area = spacing * spacing
    G = max(0.0, float(fc.deposition))
    alpha = max(0.0, float(fc.diffusion)) * dt / cell_area
    iso = float(np.clip(fc.isostasy, 0.0, 1.0))
    flex = max(0.0, float(fc.flexure_m)) / spacing
    land_w = np.maximum(blur(land.astype(np.float64), flex), 1e-3)
    U = np.where(movable, uplift, 0.0).reshape(-1)  # uplift rate (m/yr)
    exponent = float(fc.routing_exponent)
    spread = max(0.0, float(fc.floodplain_m)) / spacing
    crit = max(0.0, float(fc.critical_slope))
    dep = np.zeros(h.size)           # deposition rate from the previous step's sediment flux (m/yr)
    open_edge = border_mask(shape) & land
    edge_top = h.copy()
    for _ in range(int(steps)):
        prev = h
        flat = h.reshape(-1)
        rec, order, area, dist = flow_routing(h, ocean, spacing, rng, exponent)
        levels = _levels(order, rec)
        hard = strata.hardness(h, fold).reshape(-1) if strata is not None else np.zeros(h.size)
        K = fc.k_erosion * np.clip(1.0 - fc.hardness_effect * hard, 0.02, 1.0)
        F = K * area ** fc.m * dt / dist ** fc.n
        src = flat + dt * (U + dep)
        new = _solve_implicit(src, rec, levels, F, fc.n)
        new = np.where(fixed_f, flat, new)
        if G > 0:
            # Yuan et al. (2019): D = G Qs / A with Qs the net sediment flux, i.e. the upstream
            # integral of (E - D). Solved exactly for this step's incision E in one donor-first
            # pass; the previous code fed back the lagged D (factor -G per step), which
            # oscillated with period 2 for G >= 1.
            ero = np.where(fixed_f, 0.0, np.maximum(src - new, 0.0) / dt)
            d = np.where(fixed_f, 0.0, _deposition(ero, rec, levels, area, cell_area, G)).reshape(shape)
            if spread > 0.3:
                # Channel migration and avulsion spread the alluvium over the valley
                # floor: a mass-conserving scatter onto the cells within a few metres
                # of the local low point, so valley fills and plains widen as they aggrade.
                hn = new.reshape(shape)
                w = np.clip(1.0 - (hn - sliding_min(hn, int(np.ceil(spread)))) / FLOODPLAIN_RELIEF, 0.0, 1.0)
                w = np.where(fixed, 0.0, w)
                bw = blur(w, spread)
                ok = bw > 1e-3
                d = w * blur(np.where(ok, d / np.maximum(bw, 1e-3), 0.0), spread) + np.where(ok, 0.0, d)
            dep = d.reshape(-1)
        h = _diffuse(new.reshape(shape), fixed, alpha)
        if crit > 0:
            # Threshold hillslopes: landslides keep slopes below a critical gradient
            # that grows with rock strength; the debris is carried off by the rivers.
            sc = np.maximum(crit * (0.75 + 0.5 * hard.reshape(shape)), INPUT_SLOPE_KEEP * s_input)
            h = _limit_slopes(h, fixed, sc, spacing)
        if iso > 0:
            loss = np.where(land, flat.reshape(shape) + dt * U.reshape(shape) - h, 0.0)
            h = h + np.where(movable, iso * blur(loss, flex) / land_w, 0.0)
        if open_edge.any():
            h = _open_border(h, prev, open_edge, edge_top)
        h = np.maximum(np.clip(h, lo, hi), floor)
    return h


def _open_border(h: np.ndarray, prev: np.ndarray, edge: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Land border cells follow the change of their inward neighbour (corners the diagonal),
    but never rise above their height at the start (``top``).

    The border is the outlet of every river leaving the map, so it is held fixed in the
    solve; without this it would never erode and would stand as a one-cell wall above the
    incised ring next to it. An outlet that kept rising with uplift or rebound would lift
    the base level and the whole region would climb, hence the cap."""
    d = h - prev
    inner = np.empty_like(d)
    inner[1:-1, 1:-1] = 0.0
    inner[0, :], inner[-1, :] = d[1, :], d[-2, :]
    inner[:, 0], inner[:, -1] = d[:, 1], d[:, -2]
    inner[0, 0], inner[0, -1], inner[-1, 0], inner[-1, -1] = d[1, 1], d[1, -2], d[-2, 1], d[-2, -2]
    return np.where(edge, np.minimum(prev + inner, top), h)


def _steepest(h: np.ndarray, spacing: float) -> np.ndarray:
    """Steepest-descent gradient (rise/run) to any of the 8 neighbours, >= 0."""
    hgt, wid = h.shape
    pad = np.pad(h, 1, mode="edge")
    best = np.zeros(h.shape)
    for k, (dy, dx) in enumerate(NEIGHBORS8):
        best = np.maximum(best, (h - pad[1 + dy:1 + dy + hgt, 1 + dx:1 + dx + wid]) / (spacing * _OFFS_DIST[k]))
    return best


def _limit_slopes(h: np.ndarray, fixed: np.ndarray, sc: np.ndarray, spacing: float,
                  relax: float = 0.5) -> np.ndarray:
    """Move cells standing more than ``sc * distance`` above a neighbour part of the way down.

    One partial relaxation per time step: the envelope converges over the run
    while short-wavelength texture is not flattened into planar facets.
    """
    hgt, wid = h.shape
    pad = np.pad(h, 1, mode="constant", constant_values=np.inf)
    env = h
    for k, (dy, dx) in enumerate(NEIGHBORS8):
        env = np.minimum(env, pad[1 + dy:1 + dy + hgt, 1 + dx:1 + dx + wid] + sc * spacing * _OFFS_DIST[k])
    return np.where(fixed, h, h - relax * (h - env))


def _binomial(a: np.ndarray) -> np.ndarray:
    """Separable 1-2-1 filter (removes single-cell, Nyquist-frequency features)."""
    p = np.pad(a, 1, mode="edge")
    a = (p[:-2, 1:-1] + 2.0 * p[1:-1, 1:-1] + p[2:, 1:-1]) / 4.0
    p = np.pad(a, 1, mode="edge")
    return (p[1:-1, :-2] + 2.0 * p[1:-1, 1:-1] + p[1:-1, 2:]) / 4.0


# -- public entry point ------------------------------------------------------
def stream_power_erosion(height: np.ndarray, uplift: np.ndarray, strata, cfg) -> np.ndarray:
    fc, wc = cfg.fluvial, cfg.world
    h0 = np.asarray(height, dtype=np.float64)
    if (not fc.enabled or fc.strength <= 0 or h0.ndim != 2
            or (fc.iterations <= 0 and fc.fine_iterations <= 0)):
        return h0.copy()
    sea = wc.sea_level
    ocean0 = ocean_mask(h0, sea)
    if ocean0.all():
        return h0.copy()
    rng = np.random.default_rng(derive_seed(wc.seed, "fluvial-routing"))
    up = np.zeros(h0.shape) if uplift is None else np.clip(np.asarray(uplift, dtype=np.float64), 0.0, None)
    u_max = float(up.max())
    U0 = fc.uplift_rate * up / u_max if u_max > 0 else np.zeros(h0.shape)
    cap = max(0.0, float(fc.max_change))

    # Coarse simulation grid (corner-aligned resampling, as ofm.grid.resample).
    div = int(fc.resolution_divisor) if fc.resolution_divisor in (1, 2, 4) else 1
    shape = tuple(max(16, s // div) for s in h0.shape)
    coarse = shape != h0.shape
    coarse_steps = int(fc.iterations) + (0 if coarse else max(0, int(fc.fine_iterations)))
    fine_steps = max(0, int(fc.fine_iterations)) if coarse else 0

    h = h0.copy()
    if coarse_steps > 0:
        if coarse:
            hc = resample(blur(h0, 0.4 * div), shape, cubic=True)
            spacing = wc.cell_size * (h0.shape[1] - 1) / (shape[1] - 1)
            Uc = resample(U0, shape, cubic=False)
            fold_c = resample(strata.fold, shape, cubic=True) if strata is not None else None
        else:
            hc, spacing, Uc = h0, wc.cell_size, U0
            fold_c = strata.fold if strata is not None else None
        ocean = ocean_mask(hc, sea)
        land = ~ocean
        # Base level: the sea surface. Ocean cells are held there and never change.
        start = np.where(ocean, sea, hc)
        # A little white-noise roughness breaks the symmetry of smooth slopes so
        # channels nucleate and compete; it is removed again from the change.
        noise_rng = np.random.default_rng(derive_seed(wc.seed, "fluvial-roughness"))
        pert = np.where(land & ~border_mask(shape),
                        noise_rng.uniform(-1.0, 1.0, shape) * max(0.0, float(fc.roughness)), 0.0)
        floor = np.where(land, np.minimum(start, sea + MIN_LAND), -np.inf)
        hc_end = _evolve(start + pert, ocean, np.where(land, Uc, 0.0), fold_c, strata, fc, spacing,
                         coarse_steps, start - cap, start + cap, floor, rng, _steepest(start, spacing))
        delta = hc_end - start - pert
        if coarse:
            # The 1-2-1 pre-filter removes single-coarse-cell channels that would
            # alias into diagonal moire when upsampled.
            delta = resample(np.where(ocean, 0.0, _binomial(delta)), h0.shape, cubic=True)
        h = h0 + np.where(ocean0, 0.0, np.clip(delta, -cap, cap))

    floor0 = np.where(ocean0, -np.inf, np.minimum(h0, sea + MIN_LAND))
    if fine_steps > 0:
        # Final steps on the full grid sharpen channels and start first-order tributaries.
        h = np.maximum(h, floor0)
        start = np.where(ocean0, sea, h)
        fine = _evolve(start, ocean0, np.where(ocean0, 0.0, U0), strata.fold if strata is not None else None,
                       strata, fc, wc.cell_size, fine_steps, h0 - cap, h0 + cap, floor0, rng,
                       _steepest(np.where(ocean0, sea, h0), wc.cell_size))
        h = np.where(ocean0, h0, fine)

    delta = np.clip(h - h0, -cap, cap) * float(np.clip(fc.strength, 0.0, 1.0))
    out = h0 + np.where(ocean0, 0.0, delta)
    return np.where(ocean0, out, np.maximum(out, floor0))
