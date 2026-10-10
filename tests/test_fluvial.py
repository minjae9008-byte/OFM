"""Stream-power landscape evolution (ofm.physics.fluvial) and basin carving."""
import time

import numpy as np
import pytest

from ofm.geology.strata import Strata
from ofm.geology.tectonics import carve_basins, simulate_tectonics
from ofm.physics.fluvial import flow_routing, stream_power_erosion
from ofm.physics.hydrology import ocean_mask, priority_flood
from tests.conftest import small_config


@pytest.fixture(scope="module")
def tect():
    cfg = small_config(**{"world.size": 128})
    tec = simulate_tectonics(cfg)
    strata = Strata(cfg, tec.uplift, cfg.world.cell_size)
    return cfg, tec, strata


@pytest.fixture(scope="module")
def evolved(tect):
    cfg, tec, strata = tect
    t0 = time.perf_counter()
    out = stream_power_erosion(tec.height, tec.uplift, strata, cfg)
    return out, time.perf_counter() - t0


def _plane(size=64, cell=16.0, seed=3):
    """A coast-facing slope: sea along the south edge, land rising 0.15 m/m northwards."""
    rng = np.random.default_rng(seed)
    yy, _ = np.mgrid[0:size, 0:size].astype(np.float64)
    h = (yy - 6.0) * cell * 0.15 + rng.uniform(-0.2, 0.2, (size, size))
    return np.where(yy < 6, -10.0, h)


def _area_cells(h, cfg):
    ocean = ocean_mask(h, cfg.world.sea_level)
    _, _, area, _ = flow_routing(h, ocean, cfg.world.cell_size)
    return area.reshape(h.shape) / cfg.world.cell_size ** 2, ocean


def test_deterministic_and_finite(tect, evolved):
    cfg, tec, strata = tect
    out, _ = evolved
    again = stream_power_erosion(tec.height, tec.uplift, strata, cfg)
    assert np.array_equal(out, again)
    assert out.shape == tec.height.shape
    assert np.isfinite(out).all()
    assert not np.array_equal(out, tec.height)


def test_sea_floor_unchanged_and_coast_kept(tect, evolved):
    cfg, tec, _ = tect
    out, _ = evolved
    sea = cfg.world.sea_level
    ocean = ocean_mask(tec.height, sea)
    assert np.array_equal(out[ocean], tec.height[ocean])
    land = ~ocean
    assert (out[land] > sea).all()                      # no new sea: the coastline stays put
    assert np.array_equal(ocean_mask(out, sea), ocean)


def test_channels_incised_more_than_hillslopes(tect, evolved):
    cfg, tec, _ = tect
    out, _ = evolved
    area, ocean = _area_cells(out, cfg)       # the evolved drainage network
    land = ~ocean & (out > cfg.world.sea_level + 5.0)
    lowered = tec.height - out
    channels = land & (area >= np.quantile(area[land], 0.95))
    hillslopes = land & (area <= 2.0)
    assert channels.sum() > 20 and hillslopes.sum() > 100
    assert lowered[channels].mean() > lowered[hillslopes].mean() + 10.0


def test_disabled_or_zero_strength_returns_input(tect):
    cfg, tec, strata = tect
    for key, value in (("fluvial.enabled", False), ("fluvial.strength", 0.0)):
        c = small_config(**{"world.size": 128, key: value})
        out = stream_power_erosion(tec.height, tec.uplift, strata, c)
        assert np.array_equal(out, tec.height)
        assert out is not tec.height
    c = small_config(**{"world.size": 128, "fluvial.iterations": 0, "fluvial.fine_iterations": 0})
    assert np.array_equal(stream_power_erosion(tec.height, tec.uplift, strata, c), tec.height)


def test_change_is_capped_and_blended(tect):
    cfg, tec, strata = tect
    c = small_config(**{"world.size": 128, "fluvial.max_change": 12.0})
    out = stream_power_erosion(tec.height, tec.uplift, strata, c)
    assert np.abs(out - tec.height).max() <= 12.0 + 1e-9
    full = stream_power_erosion(tec.height, tec.uplift, strata, cfg)
    half = stream_power_erosion(tec.height, tec.uplift, strata,
                                small_config(**{"world.size": 128, "fluvial.strength": 0.5}))
    land = tec.height > cfg.world.sea_level + 5.0
    np.testing.assert_allclose((half - tec.height)[land], 0.5 * (full - tec.height)[land], atol=1e-6)


def test_runtime_on_128_grid(evolved):
    _, seconds = evolved
    assert seconds < 4.0, f"fluvial stage took {seconds:.2f} s on a 128 grid"


@pytest.mark.parametrize("divisor", [1, 2])
def test_tilted_plane_develops_drainage_network(divisor):
    cfg = small_config(**{"world.size": 64, "fluvial.resolution_divisor": divisor,
                          "fluvial.iterations": 30, "fluvial.uplift_rate": 1e-4})
    h0 = _plane()
    out = stream_power_erosion(h0, np.ones_like(h0), None, cfg)
    a0, ocean = _area_cells(h0, cfg)
    a1, _ = _area_cells(out, cfg)
    land = ~ocean
    # On the bare plane, flow runs in parallel lines; the evolved surface gathers it
    # into a few trunk channels with much larger catchments.
    assert a1[land].max() > 3.0 * a0[land].max()
    big0 = (a0[land] > 60).sum()
    big1 = (a1[land] > 60).sum()
    assert big1 > big0
    # Valleys: trunk channels sit well below the mean land elevation of their rows.
    rows = land.any(axis=1)
    row_mean = np.where(land, out, 0.0).sum(axis=1, keepdims=True) / np.maximum(land.sum(axis=1, keepdims=True), 1)
    assert rows.sum() > 40
    trunk = land & (a1 > 60)
    assert (out - row_mean)[trunk].mean() < -3.0


def test_nonlinear_slope_exponent_is_stable(tect):
    _, tec, strata = tect
    c = small_config(**{"world.size": 128, "fluvial.n": 1.5, "fluvial.k_erosion": 1e-4})
    out = stream_power_erosion(tec.height, tec.uplift, strata, c)
    assert np.isfinite(out).all()
    assert (tec.height - out).max() > 1.0


def test_deposition_raises_lowland_valley_floors():
    base = {"world.size": 64, "fluvial.resolution_divisor": 1, "fluvial.iterations": 30,
            "fluvial.uplift_rate": 1e-4, "fluvial.isostasy": 0.0}
    h0 = _plane()
    up = np.ones_like(h0)
    no_dep = stream_power_erosion(h0, up, None, small_config(**dict(base, **{"fluvial.deposition": 0.0})))
    dep = stream_power_erosion(h0, up, None, small_config(**dict(base, **{"fluvial.deposition": 1.5})))
    land = h0 > 0
    assert dep[land].sum() > no_dep[land].sum()           # alluvium kept on land


def test_basins_are_closed_depressions(tect, evolved):
    cfg, _, _ = tect
    out, _ = evolved
    carved = carve_basins(out, cfg)
    assert np.isfinite(carved).all()
    ocean = ocean_mask(carved, cfg.world.sea_level)
    filled, _, _ = priority_flood(carved, ocean)
    filled0, _, _ = priority_flood(out, ocean_mask(out, cfg.world.sea_level))
    # Carving adds depressions deep enough for hydrology to hold a lake.
    assert ((filled - carved) > 2.0).sum() > ((filled0 - out) > 2.0).sum()
    assert np.array_equal(carve_basins(out, cfg), carved)
