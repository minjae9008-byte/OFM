"""Export-resolution landforms (ofm/detail/hires.py) and the amplified heightfield."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from ofm.config import HiresConfig, OFMConfig
from ofm.detail.amplify import amplify
from ofm.detail.hires import floodplain_half_width, floodplains, river_field, scree_aprons, shape_coasts
from ofm.grid import bilinear, gradient
from ofm.pipeline import generate_base, rebuild

from .conftest import small_config

STEPS = ("erosion", "scree", "beaches", "wave_platform", "floodplains")


def _with_steps(world, on):
    cfg = world.cfg.copy()
    for s in STEPS:
        setattr(cfg.detail.hires, s, s in on)
    return dataclasses.replace(world, cfg=cfg)


def _hires_uv(world):
    n = world.hires.shape[0]
    c = np.arange(n) * world.hires_spacing
    X, Y = np.meshgrid(c, c)
    return X / world.cell - 0.5, Y / world.cell - 0.5


def _slope_deg(h, spacing):
    gx, gy = gradient(h, spacing)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


@pytest.fixture(scope="module")
def river_world():
    """Small world with several rivers (the default small world has none)."""
    return rebuild(generate_base(small_config(**{"hydrology.river_min_discharge": 0.1, "world.seed": 5})),
                   psych=False)


# -- pipeline level ------------------------------------------------------------------
def test_shape_and_spacing(small_world):
    w = small_world
    n = w.size * w.cfg.detail.upscale + 1
    assert w.hires.shape == (n, n)
    assert w.hires_spacing == pytest.approx(w.extent / (n - 1))
    assert np.isfinite(w.hires).all()
    # sample k sits at x = k * spacing: node values agree with the sim grid at cell centres
    j = i = w.size // 2
    x, y = w.cell_center(i, j)
    assert abs(float(bilinear(w.hires, x / w.hires_spacing, y / w.hires_spacing)) - w.height[j, i]) < 25.0


def _check_water(w, h):
    U, V = _hires_uv(w)
    sea = w.cfg.world.sea_level
    ocean = bilinear(w.ocean.astype(np.float64), U, V) > 0.5
    assert ocean.any()
    assert (h[ocean] < sea).all(), "sea floor above sea level"
    if w.lake_mask is not None and w.lake_mask.any():
        lake = bilinear(w.lake_mask.astype(np.float64), U, V) > 0.5
        level = bilinear(np.where(w.lake_mask, w.lake_level, w.height), U, V)
        assert (h[lake] <= level[lake] - 0.4 + 1e-6).all(), "lake bed above lake level"
    sp = w.hires_spacing
    for r in w.rivers:
        p = np.asarray(r["points"])
        hz = bilinear(h, p[:, 0] / sp, p[:, 1] / sp)
        assert (hz <= p[:, 2] + 1e-6).all(), f"{r['id']} blocked by terrain"


def test_water_constraints(small_world):
    _check_water(small_world, small_world.hires)


def test_river_channels_stay_open(river_world):
    assert river_world.rivers
    _check_water(river_world, river_world.hires)


def test_deterministic(small_world):
    a, sa = amplify(small_world)
    b, sb = amplify(small_world)
    assert sa == sb
    assert np.array_equal(a, b)


def test_all_steps_off_still_valid(river_world):
    off = _with_steps(river_world, ())
    h, sp = amplify(off)
    assert h.shape == river_world.hires.shape and sp == river_world.hires_spacing
    assert np.isfinite(h).all()
    _check_water(off, h)
    on, _ = amplify(_with_steps(river_world, STEPS))
    assert np.abs(on - h).max() > 0.1          # the processes actually do something


# -- scree -------------------------------------------------------------------------------
def test_scree_raises_cliff_foot_at_repose():
    hc = HiresConfig()
    n, sp = 48, 8.0
    x = np.arange(n) * sp
    h = np.tile(np.where(x >= 200.0, 60.0, 0.0), (n, 1))      # 60 m hard free face
    protect = np.zeros((n, n))
    protect[:8] = 1.0                                          # e.g. a river corridor
    out = scree_aprons(h, np.full((n, n), 0.8), protect, sp, hc)
    foot = int(200.0 / sp) - 1
    assert out[24, foot] > h[24, foot] + 3.0                    # debris piled against the face
    assert (out >= h - 1e-9).all()
    assert np.array_equal(out[:8], h[:8])                       # protected cells untouched
    assert np.array_equal(out[:, foot + 1:], h[:, foot + 1:])   # the face itself stands
    row = out[24, :foot + 1]
    assert (np.diff(row) >= -1e-9).all()                        # rises towards the face
    apron = out - h
    s = _slope_deg(out, sp)
    interior = (apron > 0.3)
    interior[:, foot:] = False                                  # exclude cells touching the face
    assert interior.sum() > 20
    assert s[interior].max() <= hc.scree_repose_deg + 3.0
    reach = int(np.ceil(hc.scree_reach_m / sp))
    assert np.allclose(out[12:, :foot - reach - 1], 0.0)        # nothing beyond the reach


# -- coasts ----------------------------------------------------------------------------
def _coast(n=80, sp=8.0, land_deg=9.0, sea_deg=9.0, shore_x=240.0, seed=0):
    x = np.arange(n) * sp
    rng = np.random.default_rng(seed)
    d = x - shore_x
    h1 = np.where(d >= 0, d * np.tan(np.radians(land_deg)), d * np.tan(np.radians(sea_deg)))
    h = np.tile(h1, (n, 1)) + rng.normal(0.0, 0.4, (n, n))
    h = np.where(np.tile(d, (n, 1)) >= 4.0, np.maximum(h, 0.3), np.minimum(h, -0.3))
    return h, h < 0.0


def test_beach_profile_is_gentle_on_soft_low_coast():
    hc = HiresConfig()
    sp = 8.0
    h, ocean = _coast(land_deg=9.0)
    out = shape_coasts(h, ocean, np.zeros_like(h), np.full(h.shape, 0.2), 0.0, sp, hc)
    assert (out[ocean] < 0.0).all() and (out[~ocean] > 0.0).all()   # shoreline does not move
    col = int(240.0 / sp)
    rows = slice(10, 70)
    near = slice(col, col + int(0.4 * hc.beach_width_m / sp))
    s_before = _slope_deg(h, sp)[rows, near]
    s_after = _slope_deg(out, sp)[rows, near]
    assert s_after.mean() < s_before.mean() - 2.0
    assert np.median(s_after) <= hc.beach_slope_deg + 2.0
    assert out[rows, near].max() <= hc.beach_berm_m + 0.5           # berm stays low
    # underwater: smooth shoreface (Dean profile), no deeper than the original trend
    off = slice(col - int(0.4 * hc.beach_width_m / sp), col)
    assert _slope_deg(out, sp)[rows, off].mean() < _slope_deg(h, sp)[rows, off].mean()
    # far inland untouched
    assert np.array_equal(out[:, col + 12:], h[:, col + 12:])


def test_hard_cliff_gets_wave_platform_and_stays_untouched():
    hc = HiresConfig(beaches=False)
    n, sp = 80, 8.0
    x = np.arange(n) * sp
    h = np.tile(np.where(x >= 240.0, 45.0 + 0.05 * (x - 240.0), -25.0), (n, 1))
    ocean = h < 0.0
    out = shape_coasts(h, ocean, np.ones_like(h), np.full(h.shape, 0.9), 0.0, sp, hc)
    assert np.array_equal(out[~ocean], h[~ocean])                     # cliff untouched
    col = int(240.0 / sp)
    plat = out[10:70, col - 3:col]                                    # first 3 cells seaward
    assert (plat < 0.0).all()
    assert (plat > -hc.platform_depth_m - 2.5).all()                  # just below sea level
    drop = out[10:70, col - 1] - out[10:70, col - 3]                  # falls gently seaward
    assert (drop >= -1e-6).all() and drop.mean() < 2.0
    assert np.allclose(out[:, :col - 2 * int(hc.platform_width_m / sp)], -25.0)


# -- floodplains --------------------------------------------------------------------------
def _valley(n=80, sp=8.0, seed=1):
    """Alluvial valley: 1 % down-valley gradient, gentle bumpy floor, 16 m wide river."""
    x = np.arange(n) * sp
    X, Y = np.meshgrid(x, x)
    xc = 316.0
    water = 10.0 + 0.01 * Y
    rng = np.random.default_rng(seed)
    h = water + 1.0 + 0.04 * np.abs(X - xc) + rng.normal(0.0, 1.0, X.shape)
    pts = [[xc, y, 10.0 + 0.01 * y] for y in np.arange(620.0, 0.0, -8.0)]
    world = SimpleNamespace(cfg=OFMConfig(), rivers=[{
        "id": "river_0", "points": pts, "width": [16.0] * len(pts), "discharge": [44.0] * len(pts)}])
    return h, world


def test_floodplain_flattens_valley_floor_and_keeps_channel_low():
    hc = HiresConfig()
    sp = 8.0
    h, world = _valley()
    n = h.shape[0]
    rf = river_field(world, n, sp, 80.0)
    assert np.isfinite(rf.dist).any()
    out = floodplains(h, rf, np.zeros_like(h), -100.0, sp, hc)
    W = floodplain_half_width(rf, hc, sp)
    zone = (rf.dist >= rf.channel_hw + 1.5 * sp) & (rf.dist <= 0.7 * W)
    zone[:6] = zone[-6:] = False
    assert zone.sum() > 200
    lvl = rf.water + np.clip(0.5 + 0.5 * rf.depth, 0.5, 1.5)
    assert np.abs(out - lvl)[zone].mean() < 0.5 * np.abs(h - lvl)[zone].mean()   # towards bankfull level
    g0 = np.hypot(*gradient(h, sp))
    g1 = np.hypot(*gradient(out, sp))
    assert g1[zone].mean() < 0.7 * g0[zone].mean()                              # flatter
    assert (out - lvl)[zone].std() < 0.6 * (h - lvl)[zone].std()                # smoother
    # channel lower than its floodplain
    centre = rf.dist < 0.5 * rf.channel_hw
    for j in range(10, n - 10):
        if centre[j].any() and zone[j].any():
            assert out[j, centre[j]].max() < out[j, zone[j]].min()
    far = rf.dist > W + sp
    assert np.array_equal(out[far], h[far])


def test_floodplain_skips_steep_reaches():
    hc = HiresConfig()
    sp = 8.0
    h, world = _valley()
    for r in world.rivers:                      # a 30 % mountain stream: no alluvial floor
        r["points"] = [[x, y, 10.0 + 0.3 * y] for x, y, _ in r["points"]]
    rf = river_field(world, h.shape[0], sp, 80.0)
    out = floodplains(h, rf, np.zeros_like(h), -100.0, sp, hc)
    assert np.abs(out - h).max() < 1e-9


@pytest.mark.parametrize("shape,seed", [("coast", 2), ("inland", 3), ("inland", 8)])
def test_steep_rivers_stay_open_between_vertices(shape, seed):
    from ofm.pipeline import generate_base, rebuild
    w = rebuild(generate_base(small_config(**{"world.size": 64, "world.shape": shape, "world.seed": seed,
                                              "hydrology.river_min_discharge": 0.1})), psych=False)
    sp = w.hires_spacing
    assert w.rivers
    for r in w.rivers:
        p = np.asarray(r["points"])
        t = np.linspace(0.0, 1.0, 9)[:, None]
        for a, b in zip(p[:-1], p[1:]):
            q = a + t * (b - a)
            hz = bilinear(w.hires, q[:, 0] / sp, q[:, 1] / sp)
            assert (hz <= q[:, 2] + 1e-6).all(), f"{r['id']} blocked between vertices"
