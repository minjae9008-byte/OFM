"""Climate: temperature from latitude and lapse rate, rainfall from moisture advection.

Moisture is evaporated over the ocean and carried by the prevailing wind with
semi-Lagrangian advection. Air forced uphill (wind . grad h > 0) cools and
rains out its moisture (orographic precipitation), so mountain ranges create
wet windward slopes and dry rain shadows on their lee side.
"""
from __future__ import annotations

import numpy as np

from ..config import OFMConfig
from ..grid import bilinear, blur, gradient
from ..noise import Noise, derive_seed


def temperature_field(cfg: OFMConfig, height: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    cc, wc = cfg.climate, cfg.world
    size = height.shape[0]
    lat = np.linspace(cc.latitude_south, cc.latitude_north, size)[:, None] * np.ones((1, size))
    coslat = np.cos(np.radians(np.abs(lat))) ** 1.2
    t_sea = cc.sea_level_temp_pole + (cc.sea_level_temp_equator - cc.sea_level_temp_pole) * coslat
    elev = np.maximum(height - wc.sea_level, 0.0)
    t = t_sea - cc.lapse_rate * elev / 1000.0
    # Oceans moderate coastal climate.
    ocean_f = blur(ocean.astype(np.float64), size * 0.03)
    t = t * (1 - 0.25 * ocean_f) + t_sea * 0.25 * ocean_f
    noise = Noise(derive_seed(wc.seed, "temperature"))
    yy, xx = np.mgrid[0:size, 0:size] / size
    t += noise.fbm2(xx * 4, yy * 4, 3) * 1.2
    return t


def precipitation_field(cfg: OFMConfig, height: np.ndarray, ocean: np.ndarray,
                        temperature: np.ndarray) -> np.ndarray:
    cc, wc = cfg.climate, cfg.world
    size = height.shape[0]
    cell = wc.cell_size
    rng_noise = Noise(derive_seed(wc.seed, "wind"))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)

    # Wind blows FROM wind_from_deg (0 = north, 90 = east), i.e. towards the opposite side.
    to = np.radians(cc.wind_from_deg + 180.0)
    wx0, wy0 = np.sin(to), np.cos(to)
    swirl = rng_noise.fbm2(xx / size * 2.5, yy / size * 2.5, 3) * 0.6
    wx = wx0 * np.cos(swirl) - wy0 * np.sin(swirl)
    wy = wx0 * np.sin(swirl) + wy0 * np.cos(swirl)

    hs = blur(np.maximum(height, wc.sea_level), 1.5)
    gx, gy = gradient(hs, cell)
    upslope = np.maximum(wx * gx + wy * gy, 0.0)          # m rise per m travelled
    # Saturation capacity falls with temperature (Clausius-Clapeyron-ish).
    capacity = np.clip(np.exp(0.06 * (temperature - 20.0)), 0.1, 1.2)

    moisture = np.where(ocean, 1.0, cc.boundary_moisture)
    precip = np.zeros_like(moisture)
    step = 1.0  # cells per iteration
    step_km = step * cell / 1000.0
    iterations = int(size * 1.6)
    src_x = xx - wx * step
    src_y = yy - wy * step
    upwind_out = (src_x < 0) | (src_x > size - 1) | (src_y < 0) | (src_y > size - 1)
    for _ in range(iterations):
        m = bilinear(moisture, src_x, src_y)
        m = np.where(upwind_out, cc.boundary_moisture, m)
        m = np.where(ocean, np.minimum(1.0, m + 0.08), m)
        rain = m * (cc.base_rain * step_km + cc.orographic * upslope * step_km)
        rain += np.maximum(m - capacity, 0.0) * cc.condensation * step_km
        rain = np.minimum(m, rain)
        rain = np.where(ocean, rain * 0.5, rain)
        moisture = m - np.where(ocean, 0.0, rain)
        precip += rain
    precip /= iterations
    precip = blur(precip, 1.2)
    # Normalise to annual millimetres.
    land = ~ocean
    ref = np.quantile(precip[land], 0.98) if land.any() else precip.max()
    mm = np.clip(precip / max(ref, 1e-9), 0.0, 1.3) * cc.precip_scale + cc.background_mm
    return np.maximum(mm, 60.0)


def evaporation_mm(temperature: np.ndarray) -> np.ndarray:
    """Open-water evaporation estimate (mm/year)."""
    return np.clip(380.0 + 55.0 * temperature, 150.0, 2400.0)


def runoff_fraction(temperature: np.ndarray, precipitation: np.ndarray) -> np.ndarray:
    et = np.clip(0.25 + 0.022 * temperature, 0.15, 0.85)
    dry = np.clip(1.0 - precipitation / 900.0, 0.0, 1.0) * 0.25
    return np.clip(1.0 - et - dry, 0.05, 0.85)
