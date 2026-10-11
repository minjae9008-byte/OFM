"""Stage orchestration.

    tectonics -> strata -> fluvial (stream power) -> droplet + thermal erosion -> basins
                                                                (world.height_base)
    climate -> hydrology -> biomes -> cliff masks
    -> detail heightfield (hires erosion, scree, beaches, floodplains) = the exported ground
    -> caves/features (on that ground) -> POIs/scatter
    -> terrain quality analysis -> psychology
    [AI director loop edits height_base / features / POIs and rebuilds]
    -> lore -> volume meshing -> export

``rebuild`` re-runs everything downstream of ``height_base`` so the AI
director can edit terrain and immediately re-measure the result.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable

import numpy as np

from .config import OFMConfig
from .geology.strata import Strata
from .geology.tectonics import carve_basins, simulate_tectonics
from .noise import derive_seed
from .physics.biome import classify, splat_weights
from .physics.climate import precipitation_field, temperature_field
from .physics.erosion import hydraulic_erosion, thermal_erosion
from .physics.hydrology import ocean_mask, simulate_hydrology
from .world import World

Progress = Callable[[str], None]


@contextmanager
def _stage(world: World, name: str, progress: Progress | None):
    if progress:
        progress(name)
    t0 = time.perf_counter()
    yield
    world.timings[name] = round(world.timings.get(name, 0.0) + time.perf_counter() - t0, 3)


def generate_base(cfg: OFMConfig, progress: Progress | None = None) -> World:
    wc = cfg.world
    world = World(cfg=cfg, size=wc.size, cell=wc.cell_size)
    with _stage(world, "tectonics", progress):
        tec = simulate_tectonics(cfg)
        world.uplift = tec.uplift
        world.plate_id = tec.plate_id
    with _stage(world, "strata", progress):
        world.strata = Strata(cfg, tec.uplift, wc.cell_size)
    h = tec.height
    if cfg.fluvial.enabled:
        from .physics.fluvial import stream_power_erosion
        with _stage(world, "fluvial", progress):
            h = stream_power_erosion(h, tec.uplift, world.strata, cfg)
    with _stage(world, "erosion", progress):
        rng = np.random.default_rng(derive_seed(wc.seed, "erosion"))
        h = hydraulic_erosion(h, world.strata.hardness, cfg.erosion, rng)
        h = thermal_erosion(h, world.strata.hardness, cfg.erosion, wc.cell_size)
        h = carve_basins(h, cfg)
        world.height_base = h
    if not np.isfinite(world.height_base).all():
        from .config import ConfigError
        raise ConfigError("the terrain simulation produced non-finite heights; check the tectonics, "
                          "fluvial and erosion parameters")
    return world


def rebuild(world: World, progress: Progress | None = None, psych: bool = True,
            protect: Callable[[World, str], None] | None = None) -> World:
    """Recompute every stage downstream of ``world.height_base``.

    ``protect(world, stage)`` is called after the detail heightfield (``"terrain"``), after
    caves and features (``"surface"``) and after vegetation and POIs (``"objects"``), before
    any evaluation: the AI director uses it to put protected content back
    (``ScopePolicy.protect_rebuild``) and to abort the rebuild by raising when protected terrain
    or water changed anyway."""
    from .detail.amplify import amplify
    from .detail.pois import place_pois
    from .detail.scatter import scatter_vegetation
    from .volume.caves import simulate_caves
    from .volume.cliffs import cliff_masks, place_procedural_features

    cfg = world.cfg
    wc = cfg.world
    h = world.height_base.copy()
    with _stage(world, "climate", progress):
        ocean = ocean_mask(h, wc.sea_level)
        world.temperature = temperature_field(cfg, h, ocean)
        world.precipitation = precipitation_field(cfg, h, ocean, world.temperature)
    with _stage(world, "hydrology", progress):
        hy = simulate_hydrology(cfg, h, world.temperature, world.precipitation)
        world.height = hy["height"]
        world.ocean = hy["ocean"]
        world.lake_mask = hy["lake_mask"]
        world.lake_level = hy["lake_level"]
        world.lakes = hy["lakes"]
        world.playa = hy["playa"]
        world.discharge = hy["discharge"]
        world.receiver = hy["receiver"]
        world.river_mask = hy["river_mask"]
        world.rivers = hy["rivers"]
        world.waterfalls = hy["waterfalls"]
        world.water_table = hy["water_table"]
        # temperature follows the carved terrain
        world.temperature = temperature_field(cfg, world.height, world.ocean)
    with _stage(world, "biomes", progress):
        world.hardness = world.strata.hardness(world.height)
        world.biome = classify(world.height, world.temperature, world.precipitation, world.ocean,
                               world.lake_mask, world.river_mask, world.playa, world.cell,
                               wc.sea_level, wc.seed)
        world.splat = splat_weights(world.height, world.biome, world.temperature, world.cell,
                                    world.hardness, world.river_mask)
    with _stage(world, "caves_cliffs", progress):
        world.cliff_mask, world.coast_cliff = cliff_masks(world)
    # The exported terrain comes first: caves, features, vegetation and POIs are all placed
    # on it (``world.ground_at``), so cave mouths open at the visible ground, passages keep
    # their cover under the detail surface and river channels, and features stand on it.
    with _stage(world, "detail", progress):
        world.hires, world.hires_spacing = amplify(world)
        if protect is not None:
            protect(world, "terrain")
    with _stage(world, "caves_cliffs", progress):
        world.caves = simulate_caves(world)
        for cave in world.forced_caves:          # director-added grottos survive rebuilds
            for e in cave.entrances:
                e["z"] = round(float(world.ground_at(e["x"], e["y"])), 2)
        world.caves = world.caves + list(world.forced_caves)
        user_feats = [f for f in world.features if f.source != "procedural"]
        for f in user_feats:  # re-seat edited landmarks on the new ground
            f.z = float(world.ground_at(f.x, f.y)) if f.kind != "sea_stack" else f.z
        world.features = place_procedural_features(world) + user_feats
        for k, f in enumerate(world.features):
            f.id = f"feature_{k}"
        if protect is not None:
            protect(world, "surface")
    with _stage(world, "detail", progress):
        world.scatter = scatter_vegetation(world)
        world.pois = place_pois(world)
    if protect is not None:
        protect(world, "objects")
    if cfg.quality.enabled:
        from .quality.metrics import analyze_terrain
        with _stage(world, "quality", progress):
            world.evaluation["terrain"] = analyze_terrain(world)
    if psych:
        from .psych.interest import evaluate_psychology
        with _stage(world, "psychology", progress):
            evaluate_psychology(world)
    return world


def generate(cfg: OFMConfig, progress: Progress | None = None, psych: bool = True) -> World:
    world = generate_base(cfg, progress)
    return rebuild(world, progress, psych=psych)
