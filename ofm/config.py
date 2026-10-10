"""Configuration tree for OFM.

Every stage reads its parameters from a nested dataclass. Configs are loaded
from TOML files and can be overridden with dotted paths
(``caves.density=1.5``), which is also how the AI director changes global
parameters.
"""
from __future__ import annotations

import copy
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class WorldConfig:
    name: str = "ofm_world"
    seed: int = 1337
    size: int = 256                 # simulation grid cells per side
    cell_size: float = 16.0         # metres per simulation cell
    sea_level: float = 0.0          # metres
    shape: str = "continent"        # continent | coast | archipelago | inland
    land_fraction: float = 0.62     # target fraction of cells above sea level


@dataclass
class TectonicsConfig:
    plates: int = 7
    mountain_height: float = 680.0
    rift_depth: float = 140.0
    land_base: float = 60.0
    ocean_depth: float = -220.0
    hill_amplitude: float = 85.0
    ridge_wavelength: float = 2600.0  # metres between major ridges
    hill_wavelength: float = 820.0
    warp: float = 0.12
    basins_per_km2: float = 0.35
    basin_radius: float = 260.0     # metres
    basin_depth: float = 38.0       # metres
    belt_width: float = 0.11        # fraction of map size


@dataclass
class RockLayerConfig:
    name: str = "sandstone"
    thickness: float = 40.0
    hardness: float = 0.7
    solubility: float = 0.05
    color: list[int] = field(default_factory=lambda: [180, 150, 110])


def _default_layers() -> list[RockLayerConfig]:
    return [
        RockLayerConfig("sandstone", 38.0, 0.78, 0.05, [186, 148, 104]),
        RockLayerConfig("shale", 22.0, 0.22, 0.05, [112, 104, 98]),
        RockLayerConfig("limestone", 55.0, 0.62, 0.92, [196, 190, 172]),
        RockLayerConfig("mudstone", 18.0, 0.18, 0.12, [138, 112, 92]),
        RockLayerConfig("dolomite", 30.0, 0.7, 0.6, [170, 166, 150]),
    ]


@dataclass
class StrataConfig:
    layers: list[RockLayerConfig] = field(default_factory=_default_layers)
    basement_name: str = "granite"
    basement_hardness: float = 0.95
    basement_color: list[int] = field(default_factory=lambda: [128, 122, 120])
    basement_level: float = -320.0  # reference-frame elevation of the basement top
    fold_factor: float = 1.25       # how strongly strata follow tectonic uplift
    dip: float = 0.015              # regional dip (m per m)
    undulation: float = 18.0        # metres of noise on layer boundaries


@dataclass
class ErosionConfig:
    droplets_per_cell: float = 0.9
    batch: int = 4096
    max_steps: int = 64
    inertia: float = 0.05
    capacity: float = 4.0
    min_capacity: float = 0.01
    deposit_speed: float = 0.3
    erode_speed: float = 0.3
    evaporate: float = 0.015
    gravity: float = 4.0
    brush_radius: int = 2
    hardness_effect: float = 0.85
    thermal_iterations: int = 40
    soft_talus_deg: float = 33.0
    hard_talus_deg: float = 72.0


@dataclass
class ClimateConfig:
    latitude_south: float = 36.0
    latitude_north: float = 48.0
    wind_from_deg: float = 250.0    # meteorological: direction the wind blows FROM
    sea_level_temp_equator: float = 28.0
    sea_level_temp_pole: float = -16.0
    lapse_rate: float = 6.5         # degC per km
    boundary_moisture: float = 0.55
    orographic: float = 0.55        # fraction of moisture rained out per km of forced ascent
    base_rain: float = 0.07         # fraction rained out per km travelled over land
    background_mm: float = 320.0    # convective/frontal rain independent of advection
    condensation: float = 0.6       # rain-out rate of supersaturated air per km
    precip_scale: float = 2300.0    # advected mm/year at the wettest 2% of land


@dataclass
class HydrologyConfig:
    flow_scale: float = 400.0            # game-scale exaggeration of catchment discharge
    river_min_discharge: float = 1.0     # m3/s (after flow_scale) for a cell to count as river
    min_lake_cells: int = 6
    min_lake_depth: float = 0.6
    channel_width_k: float = 2.4         # width = k * Q^0.5
    channel_depth_k: float = 0.45        # depth = k * Q^0.4
    valley_carve: float = 1.0
    waterfall_min_drop: float = 7.0


@dataclass
class CaveConfig:
    density: float = 1.0           # systems per 16 km2 of karst
    min_systems: int = 2
    max_systems: int = 14
    step: float = 6.0
    tunnel_radius: float = 4.5
    shaft_radius: float = 3.5
    min_cover: float = 14.0
    branch_chance: float = 0.035
    chamber_chance: float = 0.03
    max_length: float = 1800.0
    wall_noise: float = 1.4
    search_radius: float = 1600.0


@dataclass
class CliffConfig:
    cliff_slope_deg: float = 50.0
    min_cliff_cells: int = 24       # export-grid cells in a face before it becomes volumetric
    undercut_depth: float = 7.0
    terrace_strength: float = 0.75
    notch_depth: float = 6.0
    notch_height: float = 3.5
    rock_noise: float = 1.6


@dataclass
class VolumeConfig:
    voxel_size: float = 4.0
    chunk_voxels: int = 32
    mesh_mode: str = "hybrid"       # hybrid (terrain + volumetric meshes) | full (meshes only)
    surface_band: float = 10.0      # metres: near-surface band that cuts terrain holes
    remove_floating: bool = True


@dataclass
class DetailConfig:
    upscale: int = 2                # export heightmap = size*upscale + 1
    rock_detail: float = 0.32       # x cell size
    soil_detail: float = 0.05
    scatter_max: int = 60000
    poi_count: int = 26
    poi_spacing: float = 280.0


@dataclass
class PsychConfig:
    grid: int = 64                  # observer / navigation grid
    rays: int = 48
    steps: int = 48
    eye_height: float = 1.7
    view_distance: float = 2600.0
    landmarks: int = 14
    agents: int = 8
    agent_minutes: float = 18.0
    run_speed: float = 5.5          # m/s on flat ground
    boredom_seconds: float = 75.0
    max_walk_slope_deg: float = 42.0


@dataclass
class AIConfig:
    provider: str = "auto"          # auto | ollama | openai | none
    base_url: str = ""              # empty -> provider default
    model: str = "llama3.1:8b"
    vision: bool = False
    temperature: float = 0.4
    timeout: float = 120.0
    iterations: int = 3
    max_actions: int = 5
    language: str = "ko"
    llm_score_weight: float = 0.3
    use_system_proxy: bool = False


@dataclass
class ExportConfig:
    engines: list[str] = field(default_factory=lambda: ["unity", "unreal"])
    previews: bool = True
    glb: bool = True
    scatter_json_limit: int = 60000


@dataclass
class OFMConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    tectonics: TectonicsConfig = field(default_factory=TectonicsConfig)
    strata: StrataConfig = field(default_factory=StrataConfig)
    erosion: ErosionConfig = field(default_factory=ErosionConfig)
    climate: ClimateConfig = field(default_factory=ClimateConfig)
    hydrology: HydrologyConfig = field(default_factory=HydrologyConfig)
    caves: CaveConfig = field(default_factory=CaveConfig)
    cliffs: CliffConfig = field(default_factory=CliffConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    detail: DetailConfig = field(default_factory=DetailConfig)
    psych: PsychConfig = field(default_factory=PsychConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def copy(self) -> "OFMConfig":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigError(ValueError):
    pass


def _coerce(value: Any, current: Any, path: str) -> Any:
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, str):
        return str(value)
    if isinstance(current, list):
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list")
        return value
    return value


def _merge(obj: Any, data: dict[str, Any], path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        full = f"{path}.{key}" if path else key
        if key not in known:
            raise ConfigError(f"unknown config key '{full}'")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, full)
        elif key == "layers" and isinstance(value, list):
            setattr(obj, key, [RockLayerConfig(**layer) for layer in value])
        else:
            setattr(obj, key, _coerce(value, current, full))


def config_from_dict(data: dict[str, Any]) -> OFMConfig:
    cfg = OFMConfig()
    _merge(cfg, data)
    return cfg


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> OFMConfig:
    cfg = OFMConfig()
    if path:
        with open(path, "rb") as fh:
            _merge(cfg, tomllib.load(fh))
    for dotted, value in (overrides or {}).items():
        set_by_path(cfg, dotted, value)
    validate(cfg)
    return cfg


def get_by_path(cfg: OFMConfig, dotted: str) -> Any:
    obj: Any = cfg
    for part in dotted.split("."):
        if not hasattr(obj, part):
            raise ConfigError(f"unknown config key '{dotted}'")
        obj = getattr(obj, part)
    return obj


def set_by_path(cfg: OFMConfig, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        if not hasattr(obj, part):
            raise ConfigError(f"unknown config key '{dotted}'")
        obj = getattr(obj, part)
    leaf = parts[-1]
    if not is_dataclass(obj) or leaf not in {f.name for f in fields(obj)}:
        raise ConfigError(f"unknown config key '{dotted}'")
    setattr(obj, leaf, _coerce(value, getattr(obj, leaf), dotted))


def validate(cfg: OFMConfig) -> None:
    w = cfg.world
    if w.size < 32 or w.size & (w.size - 1):
        raise ConfigError("world.size must be a power of two >= 32")
    if w.shape not in ("continent", "coast", "archipelago", "inland"):
        raise ConfigError(f"world.shape '{w.shape}' is not supported")
    if cfg.volume.mesh_mode not in ("hybrid", "full"):
        raise ConfigError("volume.mesh_mode must be 'hybrid' or 'full'")
    if cfg.detail.upscale not in (1, 2, 4):
        raise ConfigError("detail.upscale must be 1, 2 or 4")
    if cfg.volume.voxel_size <= 0 or cfg.volume.chunk_voxels < 8:
        raise ConfigError("volume.voxel_size must be > 0 and chunk_voxels >= 8")
    if not cfg.strata.layers:
        raise ConfigError("strata.layers must contain at least one layer")
    for engine in cfg.export.engines:
        if engine not in ("unity", "unreal"):
            raise ConfigError(f"unknown export engine '{engine}'")
    if cfg.ai.provider not in ("auto", "ollama", "openai", "none"):
        raise ConfigError("ai.provider must be auto, ollama, openai or none")
