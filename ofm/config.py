"""Configuration tree for OFM.

Every stage reads its parameters from a nested dataclass. Configs are loaded
from TOML files and can be overridden with dotted paths
(``caves.density=1.5``), which is also how the AI director changes global
parameters.
"""
from __future__ import annotations

import copy
import tomllib
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# Names of the actions the AI director may request (kept here so config validation
# does not need to import the AI package).
AI_ACTIONS = ("add_landmark", "add_cave", "add_lake", "add_poi", "raise_relief", "smooth_terrain", "set_param")


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
    basin_radius: float = 340.0     # metres
    basin_depth: float = 38.0       # metres
    coast_roughness: float = 400.0  # metres of fractal horizontal shoreline displacement (bays, headlands); 0 = off
    coast_wavelength: float = 300.0 # metres: base wavelength of the shoreline warp
    coast_band: float = 80.0        # metres above/below sea level over which the warp fades out
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
    droplets_per_cell: float = 0.4  # the fluvial stage carves the valleys; droplets add gullies and fans
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
class FluvialConfig:
    """Stream-power landscape evolution (dE/dt = U - K A^m S^n + diffusion), solved
    implicitly along the drainage stack (Braun & Willett 2013), with optional
    sediment deposition (Yuan et al. 2019). Carves dendritic valley networks
    before the droplet erosion adds gullies."""
    enabled: bool = True
    iterations: int = 24
    resolution_divisor: int = 2     # solve on a coarser grid, add the change back (1 = full res)
    dt_years: float = 20000.0
    k_erosion: float = 2.0e-5       # erodibility for soft rock (scaled by 1 - hardness_effect * hardness)
    m: float = 0.45                 # drainage-area exponent
    n: float = 1.0                  # slope exponent (implicit solver is exact for n = 1)
    uplift_rate: float = 6.0e-5     # m/yr at full orogenic uplift
    diffusion: float = 0.01         # hillslope diffusivity (m^2/yr)
    deposition: float = 0.6         # dimensionless deposition coefficient G (0 = detachment-limited)
    hardness_effect: float = 0.8
    max_change: float = 220.0       # metres: cap on cumulative change (keeps game-scale relief)
    strength: float = 1.0           # blend of the evolved terrain into the input (0..1)
    roughness: float = 1.5          # metres of initial white-noise perturbation (seeds channel initiation)
    isostasy: float = 0.75          # fraction of regionally averaged erosion returned as isostatic rebound
    flexure_m: float = 900.0        # metres: smoothing length of the isostatic (flexural) response
    routing_exponent: float = 1.5   # stochastic D8: receiver drawn with p ~ slope^x (>= 50 = plain steepest descent)
    fine_iterations: int = 4        # extra final time steps on the full-resolution grid (sharpens channels)
    floodplain_m: float = 60.0      # metres: lateral spreading of deposited alluvium across valley floors
    critical_slope: float = 0.5     # threshold hillslope gradient (x0.75 soft .. x1.25 hard rock); 0 = no landslides


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
class HiresConfig:
    """Processes applied on the export-resolution heightfield (detail amplification)."""
    erosion: bool = True            # droplet erosion at export resolution (fine gullies)
    droplets_per_cell: float = 0.25
    erosion_strength: float = 0.5
    scree: bool = True              # talus aprons accumulate below cliffs
    scree_strength: float = 0.6
    scree_reach_m: float = 40.0
    beaches: bool = True            # gentle beach / shoreface profile on soft, low coasts
    beach_width_m: float = 45.0
    beach_slope_deg: float = 3.5
    wave_platform: bool = True      # wave-cut platform at the foot of hard sea cliffs
    floodplains: bool = True        # flat alluvial floors along rivers
    floodplain_width_k: float = 3.0 # x channel width
    floodplain_flatten: float = 0.7
    erosion_steps: int = 40         # droplet lifetime (export cells travelled)
    scree_repose_deg: float = 34.0  # angle of repose of the upper apron
    scree_face_deg: float = 48.0    # faces steeper than this shed debris
    beach_berm_m: float = 2.5       # berm crest height above sea level
    beach_dean_a: float = 0.1       # Dean shape factor A (m^1/3), ~0.1 for medium sand
    platform_width_m: float = 40.0  # wave-cut platform width seaward of the cliff foot
    platform_depth_m: float = 0.6   # platform depth at the cliff foot (m below sea level)
    platform_slope_deg: float = 1.0 # seaward slope of the platform
    floodplain_max_gradient: float = 0.12  # river gradient (m/m) above which no floodplain forms
    floodplain_max_height_m: float = 6.0   # ground higher above the water is valley wall


@dataclass
class DetailConfig:
    upscale: int = 2                # export heightmap = size*upscale + 1
    rock_detail: float = 0.32       # x cell size
    soil_detail: float = 0.05
    scatter_max: int = 60000
    poi_count: int = 26
    poi_spacing: float = 280.0
    hires: HiresConfig = field(default_factory=HiresConfig)


@dataclass
class QualityWeights:
    """Weights of the terrain-quality sub-scores (normalised by their sum)."""
    slope: float = 1.0
    hypsometry: float = 0.6
    drainage: float = 1.0
    spectrum: float = 1.0
    pits: float = 1.2
    spikes: float = 1.2
    grid_artifacts: float = 1.2
    landforms: float = 0.8
    coastline: float = 0.6


@dataclass
class QualityConfig:
    """Terrain quality analysis (geomorphometry). Bands are [low, high] ranges that
    score 10; values outside fall off smoothly."""
    enabled: bool = True
    slope_median_deg: list[float] = field(default_factory=lambda: [8.0, 26.0])
    hypsometric_integral: list[float] = field(default_factory=lambda: [0.2, 0.6])
    drainage_density_km: list[float] = field(default_factory=lambda: [0.4, 4.0])   # km of river per km2 (game-scale rivers)
    spectral_slope: list[float] = field(default_factory=lambda: [1.8, 3.4])        # radial power-spectrum exponent
    max_pit_fraction: float = 0.002     # closed depressions not explained by lakes
    max_spike_fraction: float = 0.001   # single-cell spikes / holes
    max_axis_artifact: float = 0.25     # anisotropy aligned with the grid axes
    max_diagonal_artifact: float = 0.3  # anisotropy aligned with the grid diagonals (D8 lines)
    coastline_dimension: list[float] = field(default_factory=lambda: [1.08, 1.35])  # box-counting dimension (natural ~1.1-1.3)
    landform_entropy: list[float] = field(default_factory=lambda: [0.55, 1.0])      # normalised diversity of landforms
    band_softness: float = 1.0          # falloff outside a band, in band widths (score 5 at one softness)
    weights: QualityWeights = field(default_factory=QualityWeights)


@dataclass
class PsychConfig:
    grid: int = 64                  # observer / navigation grid
    rays: int = 48
    steps: int = 48
    eye_height: float = 1.7
    view_distance: float = 2600.0
    landmarks: int = 14
    agents: int = 0                 # 0 = use evaluation.profiles counts; >0 = cycle profiles
    agent_minutes: float = 18.0
    run_speed: float = 5.5          # m/s on flat ground
    boredom_seconds: float = 75.0
    max_walk_slope_deg: float = 42.0


@dataclass
class ScoreWeights:
    """Weights of the 0-10 sub-scores in the overall map score (normalised by their sum)."""
    scenic: float = 0.15
    variety: float = 0.13
    legibility: float = 0.13
    mystery: float = 0.13
    pacing: float = 0.17
    exploration: float = 0.11
    prospect_refuge: float = 0.1
    orientation: float = 0.04
    terrain_quality: float = 0.04


@dataclass
class PreferenceWeights:
    """Per-location landscape preference (Kaplan matrix + Appleton)."""
    coherence: float = 0.24
    complexity: float = 0.18
    legibility: float = 0.2
    mystery: float = 0.26
    prospect_refuge: float = 0.12


@dataclass
class EventValues:
    """Experience value of agent events; events >= strong_threshold reset boredom."""
    landmark_seen: float = 0.8
    landmark_reached: float = 1.0
    poi_discovered: float = 0.1         # + poi_discovered_importance * importance
    poi_discovered_importance: float = 0.5
    poi_reached: float = 0.6            # + poi_reached_importance * importance
    poi_reached_importance: float = 0.6
    new_biome: float = 0.4
    vista: float = 0.5
    vista_cooldown_s: float = 180.0
    water_seen: float = 0.35
    cave_entered: float = 0.9
    strong_threshold: float = 0.35


@dataclass
class TargetBands:
    """Designer targets. A metric inside its band scores 10 (see psych/scoring.py)."""
    event_gap_s: list[float] = field(default_factory=lambda: [20.0, 60.0])
    bored_fraction: list[float] = field(default_factory=lambda: [0.0, 0.15])
    discoveries_per_hour: list[float] = field(default_factory=lambda: [20.0, 45.0])
    coverage_seen: list[float] = field(default_factory=lambda: [0.5, 0.95])
    landmark_coverage: list[float] = field(default_factory=lambda: [0.6, 1.0])
    lost_fraction: list[float] = field(default_factory=lambda: [0.0, 0.2])
    backtrack_ratio: list[float] = field(default_factory=lambda: [0.0, 0.35])


@dataclass
class AgentProfile:
    """A play-tester personality. Weights scale the utility of goal types."""
    name: str = "explorer"
    count: int = 2
    poi: float = 0.6
    landmark: float = 0.7
    frontier: float = 1.4
    speed: float = 1.0              # multiplier on psych.run_speed
    view: float = 1.0               # multiplier on discovery distance
    risk_aversion: float = 0.3      # extra cost on steep edges (0..1)
    stamina_s: float = 240.0        # seconds of climbing before fatigue sets in
    fatigue_slowdown: float = 0.35  # speed loss when fatigued
    randomness: float = 0.25        # Gumbel noise on goal choice
    boredom_s: float = 0.0          # 0 = psych.boredom_seconds


def _default_profiles() -> list[AgentProfile]:
    return [
        AgentProfile("explorer", 2, poi=0.6, landmark=0.7, frontier=1.4, risk_aversion=0.2),
        AgentProfile("achiever", 2, poi=1.5, landmark=0.9, frontier=0.5, risk_aversion=0.4),
        AgentProfile("wanderer", 2, poi=0.8, landmark=1.0, frontier=1.0, randomness=0.6),
        AgentProfile("sightseer", 2, poi=0.7, landmark=1.6, frontier=0.7, speed=0.85, risk_aversion=0.6),
    ]


@dataclass
class EvaluationConfig:
    runs: int = 2                   # Monte Carlo repetitions with different agent seeds
    confidence: float = 0.9         # confidence level of reported intervals
    region_divs: int = 4            # region grid (A1..D4 for 4)
    orientation_lost_s: float = 90.0  # seconds with no landmark in view before an agent is lost
    band_softness: float = 1.0      # falloff outside a target band, in band widths (score 5 at one softness)
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    preference: PreferenceWeights = field(default_factory=PreferenceWeights)
    events: EventValues = field(default_factory=EventValues)
    targets: TargetBands = field(default_factory=TargetBands)
    profiles: list[AgentProfile] = field(default_factory=_default_profiles)


@dataclass
class ActionLimits:
    """Maximum number of times the AI may apply each action over a whole run."""
    add_landmark: int = 4
    add_cave: int = 3
    add_lake: int = 2
    add_poi: int = 12
    raise_relief: int = 2
    smooth_terrain: int = 2
    set_param: int = 3


@dataclass
class ParamRange:
    path: str = "caves.density"
    min: float = 0.2
    max: float = 3.0


def _default_param_ranges() -> list[ParamRange]:
    return [
        ParamRange("caves.density", 0.2, 3.0),
        ParamRange("caves.branch_chance", 0.0, 0.1),
        ParamRange("caves.chamber_chance", 0.0, 0.1),
        ParamRange("cliffs.undercut_depth", 0.0, 14.0),
        ParamRange("cliffs.terrace_strength", 0.0, 1.0),
        ParamRange("hydrology.river_min_discharge", 0.3, 6.0),
        ParamRange("detail.poi_count", 8, 60),
        ParamRange("detail.poi_spacing", 120.0, 600.0),
        ParamRange("climate.orographic", 0.1, 1.5),
    ]


@dataclass
class ProtectedZone:
    """Circle the AI must not touch (e.g. a hand-authored town)."""
    x: float = 0.0
    y: float = 0.0
    radius: float = 200.0
    reason: str = ""


@dataclass
class AIScopeConfig:
    """The sandbox the local model operates in. Everything outside is rejected and logged."""
    allowed_actions: list[str] = field(default_factory=lambda: list(AI_ACTIONS))
    limits: ActionLimits = field(default_factory=ActionLimits)
    max_actions_per_iteration: int = 5
    max_total_actions: int = 20
    allowed_regions: list[str] = field(default_factory=list)     # empty = every region
    protected_regions: list[str] = field(default_factory=list)
    protected_zones: list[ProtectedZone] = field(default_factory=list)
    protect_spawn_radius: float = 250.0      # keep the agents' spawn settlement untouched
    max_height_delta_m: float = 80.0         # largest change of any cell in one edit
    max_total_height_delta_m: float = 160.0  # largest cumulative change of any cell
    max_modified_area_fraction: float = 0.2  # share of land cells the AI may change (> 0.5 m)
    param_ranges: list[ParamRange] = field(default_factory=_default_param_ranges)
    max_llm_calls: int = 16
    max_llm_seconds: float = 900.0
    max_prompt_chars: int = 24000
    num_ctx: int = 8192
    max_output_tokens: int = 1500
    retries: int = 2                # repair attempts when the reply is not valid JSON / schema
    samples: int = 1                # self-consistency: independent reviews merged per call
    json_schema: bool = True        # send a JSON schema (structured outputs) when supported
    dry_run: bool = False           # log what the model would do without applying it
    min_objective_gain: float = 0.0 # accept an iteration only if the objective rises by more than this


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
    scope: AIScopeConfig = field(default_factory=AIScopeConfig)


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
    fluvial: FluvialConfig = field(default_factory=FluvialConfig)
    erosion: ErosionConfig = field(default_factory=ErosionConfig)
    climate: ClimateConfig = field(default_factory=ClimateConfig)
    hydrology: HydrologyConfig = field(default_factory=HydrologyConfig)
    caves: CaveConfig = field(default_factory=CaveConfig)
    cliffs: CliffConfig = field(default_factory=CliffConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    detail: DetailConfig = field(default_factory=DetailConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    psych: PsychConfig = field(default_factory=PsychConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
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


def _list_item_type(obj: Any, key: str) -> type | None:
    """Dataclass element type of a ``list[SomeDataclass]`` field, else None."""
    hint = typing.get_type_hints(type(obj)).get(key)
    if typing.get_origin(hint) is list:
        args = typing.get_args(hint)
        if args and is_dataclass(args[0]):
            return args[0]
    return None


def _merge(obj: Any, data: dict[str, Any], path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        full = f"{path}.{key}" if path else key
        if key not in known:
            raise ConfigError(f"unknown config key '{full}'")
        current = getattr(obj, key)
        item_type = _list_item_type(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, full)
        elif item_type is not None and isinstance(value, list):
            items = []
            for i, entry in enumerate(value):
                if not isinstance(entry, dict):
                    raise ConfigError(f"{full}[{i}]: expected a table")
                item = item_type()
                _merge(item, entry, f"{full}[{i}]")
                items.append(item)
            setattr(obj, key, items)
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
    ev = cfg.evaluation
    if not 2 <= ev.region_divs <= 8:
        raise ConfigError("evaluation.region_divs must be between 2 and 8")
    if ev.runs < 1:
        raise ConfigError("evaluation.runs must be >= 1")
    if not ev.profiles or sum(max(0, p.count) for p in ev.profiles) == 0:
        raise ConfigError("evaluation.profiles must define at least one agent")
    if any(v < 0 for v in asdict(ev.weights).values()) or sum(asdict(ev.weights).values()) <= 0:
        raise ConfigError("evaluation.weights must be non-negative and not all zero")
    for name in ("event_gap_s", "bored_fraction", "discoveries_per_hour", "coverage_seen",
                 "landmark_coverage", "lost_fraction", "backtrack_ratio"):
        band = getattr(ev.targets, name)
        if len(band) != 2 or band[0] > band[1]:
            raise ConfigError(f"evaluation.targets.{name} must be [low, high]")
    sc = cfg.ai.scope
    bad = [a for a in sc.allowed_actions if a not in AI_ACTIONS]
    if bad:
        raise ConfigError(f"ai.scope.allowed_actions has unknown actions: {', '.join(bad)}")
    for pr in sc.param_ranges:
        if pr.min > pr.max:
            raise ConfigError(f"ai.scope.param_ranges: {pr.path} has min > max")
    if cfg.fluvial.resolution_divisor not in (1, 2, 4):
        raise ConfigError("fluvial.resolution_divisor must be 1, 2 or 4")
