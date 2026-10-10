"""Writes a complete, engine-agnostic world package plus a manifest.

    <out>/world.json                 manifest (Unity JsonUtility friendly)
    <out>/terrain/heightmap.r16      Unity RAW, 16-bit LE, rows south->north
    <out>/terrain/heightmap_ue.png   Unreal Landscape 16-bit PNG, rows north->south
    <out>/terrain/holes.r8           terrain holes, 1 byte per cell (1 = hole)
    <out>/terrain/splat.raw          uint8 weights [y][x][layer]
    <out>/terrain/ue_layer_*.png     Unreal weight-map per material layer
    <out>/meshes/*.glb|*.ofmmesh     volumetric chunks (caves, overhangs, arches...)
    <out>/water/*.glb|*.ofmmesh      lakes and river ribbons
    <out>/scatter/*.bin              float32 [x, y, z, yaw_deg, scale] per instance
    <out>/previews/*.png             maps, cave profiles, 3D thumbnails
    <out>/report.md, report.json     evaluation, AI critique, director history
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .. import __version__
from ..grid import resample
from ..physics.biome import BIOME_NAMES, MATERIALS
from ..volume.builder import VolumeResult
from ..world import World
from .coords import to_unity, to_unreal
from .meshio import write_glb, write_ofmmesh
from .water import lake_meshes, river_meshes

UE_RESOLUTIONS = [127, 253, 505, 1009, 2017, 4033, 8129]


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_png16(arr01: np.ndarray, path: Path) -> None:
    v = np.clip(arr01, 0, 1) * 65535.0
    img = Image.fromarray(np.round(v).astype(np.uint16))
    img.save(path)


def export_terrain(world: World, vol: VolumeResult, out: Path) -> dict:
    tdir = out / "terrain"
    tdir.mkdir(parents=True, exist_ok=True)
    h = world.hires
    res = h.shape[0]
    hmin = float(min(h.min(), world.cfg.world.sea_level - 5.0))
    hmax = float(h.max()) + 1.0
    norm = (h - hmin) / (hmax - hmin)
    # Unity RAW (rows south -> north, matching TerrainData.SetHeights[z, x])
    (np.round(np.clip(norm, 0, 1) * 65535).astype("<u2")).tofile(tdir / "heightmap.r16")
    _write_png16(norm[::-1], tdir / "heightmap.png")

    holes = vol.holes.astype(np.uint8)
    holes.tofile(tdir / "holes.r8")
    Image.fromarray((holes[::-1] * 255).astype(np.uint8)).save(tdir / "holes.png")

    splat_res = res - 1
    layers = []
    weights = []
    for k, (name, color) in enumerate(MATERIALS):
        w = resample(world.splat[..., k], (splat_res, splat_res))
        weights.append(np.clip(w, 0, 1))
        layers.append({"name": name, "color": list(color)})
    wst = np.stack(weights, axis=-1)
    wst /= np.maximum(wst.sum(axis=-1, keepdims=True), 1e-6)
    (np.round(wst * 255).astype(np.uint8)).tofile(tdir / "splat.raw")
    for i in range(0, len(MATERIALS), 4):
        rgba = np.zeros((splat_res, splat_res, 4), dtype=np.uint8)
        part = (np.round(wst[..., i:i + 4] * 255)).astype(np.uint8)
        rgba[..., :part.shape[-1]] = part
        Image.fromarray(rgba[::-1], "RGBA").save(tdir / f"splat_{i // 4}.png")

    # Unreal Landscape: recommended resolution, 16-bit centred at 32768, rows north -> south
    ue_res = min(UE_RESOLUTIONS, key=lambda r: abs(r - res))
    h_ue = resample(h, (ue_res, ue_res), cubic=True)
    mid = 0.5 * (hmin + hmax)
    z_scale = (hmax - hmin) * 100.0 / 512.0 * 1.002    # Unreal "Z scale" in percent-of-metre units
    vals = 32768.0 + (h_ue - mid) * 100.0 * 128.0 / z_scale
    Image.fromarray(np.clip(np.round(vals), 0, 65535).astype(np.uint16)[::-1]).save(tdir / "heightmap_ue.png")
    ue_layers = []
    for k, (name, _) in enumerate(MATERIALS):
        lw = resample(world.splat[..., k], (ue_res, ue_res))
        p = tdir / f"ue_layer_{name}.png"
        Image.fromarray(np.round(np.clip(lw, 0, 1) * 255).astype(np.uint8)[::-1]).save(p)
        ue_layers.append({"name": name, "file": _rel(p, out)})
    xy_scale = world.extent * 100.0 / (ue_res - 1)
    return {
        "unity": {
            "heightmap_resolution": res,
            "heightmap_raw16": "terrain/heightmap.r16",
            "heightmap_png16": "terrain/heightmap.png",
            "height_min": hmin,
            "height_max": hmax,
            "size_m": world.extent,
            "holes_resolution": res - 1,
            "holes_raw8": "terrain/holes.r8",
            "splat_resolution": splat_res,
            "splat_raw": "terrain/splat.raw",
            "splat_png": [f"terrain/splat_{i}.png" for i in range((len(MATERIALS) + 3) // 4)],
            "layers": layers,
        },
        "unreal": {
            "heightmap_png16": "terrain/heightmap_ue.png",
            "resolution": ue_res,
            "location_cm": [0.0, -world.extent * 100.0, mid * 100.0],
            "scale": [xy_scale, xy_scale, z_scale],
            "weightmaps": ue_layers,
            "note": "Landscape: Import from File with this heightmap, location and scale. "
                    "Terrain holes must be painted with a visibility layer (see terrain/holes.png).",
        },
    }


def export_meshes(world: World, vol: VolumeResult, out: Path, glb: bool = True) -> list[dict]:
    mdir = out / "meshes"
    mdir.mkdir(parents=True, exist_ok=True)
    entries = []
    for ch in vol.chunks:
        origin = ch.origin.copy()
        name = ch.name
        e = {"name": name, "kinds": ch.kinds, "origin": [float(v) for v in origin],
             "origin_unity": [float(v) for v in to_unity(origin)],
             "origin_unreal": [float(v) for v in to_unreal(origin)],
             "triangles": ch.mesh.triangle_count, "unity": f"meshes/{name}.ofmmesh"}
        write_ofmmesh(ch.mesh, mdir / f"{name}.ofmmesh", origin)
        if glb:
            write_glb(ch.mesh, mdir / f"{name}.glb", origin, name)
            e["glb"] = f"meshes/{name}.glb"
        entries.append(e)
    return entries


def export_water(world: World, out: Path, glb: bool = True) -> dict:
    wdir = out / "water"
    wdir.mkdir(parents=True, exist_ok=True)
    lakes, rivers = [], []
    for lake, mesh, uvs in lake_meshes(world):
        origin = np.array([mesh.positions[:, 0].min(), mesh.positions[:, 1].min(), lake["level"]])
        name = lake["id"]
        write_ofmmesh(mesh, wdir / f"{name}.ofmmesh", origin, uvs)
        e = {"id": name, "name": lake.get("name", ""), "description": lake.get("description", ""),
             "level": lake["level"], "closed_basin": lake["closed_basin"], "saline": lake["saline"],
             "area_m2": lake["area_m2"], "max_depth": lake["max_depth"],
             "origin": [float(v) for v in origin], "origin_unity": [float(v) for v in to_unity(origin)],
             "origin_unreal": [float(v) for v in to_unreal(origin)], "unity": f"water/{name}.ofmmesh"}
        if glb:
            write_glb(mesh, wdir / f"{name}.glb", origin, name, uvs, (0.25, 0.45, 0.65, 0.8), alpha_blend=True)
            e["glb"] = f"water/{name}.glb"
        lakes.append(e)
    for river, mesh, uvs in river_meshes(world):
        origin = mesh.positions.min(axis=0)
        name = river["id"]
        write_ofmmesh(mesh, wdir / f"{name}.ofmmesh", origin, uvs)
        e = {"id": name, "name": river.get("name", ""), "length_m": river["length_m"],
             "points": [round(float(v), 3) for p in river["points"] for v in p],
             "widths": [round(float(v), 3) for v in river["width"]],
             "origin": [float(v) for v in origin], "origin_unity": [float(v) for v in to_unity(origin)],
             "origin_unreal": [float(v) for v in to_unreal(origin)], "unity": f"water/{name}.ofmmesh"}
        if glb:
            write_glb(mesh, wdir / f"{name}.glb", origin, name, uvs, (0.3, 0.5, 0.7, 0.8), alpha_blend=True)
            e["glb"] = f"water/{name}.glb"
        rivers.append(e)
    falls = [{"id": f["id"], "x": f["x"], "y": f["y"], "top": f["top"], "bottom": f["bottom"],
              "drop": f["drop"], "discharge": f["discharge"]} for f in world.waterfalls]
    return {"ocean": {"enabled": bool(world.ocean.any()), "level": world.cfg.world.sea_level},
            "lakes": lakes, "rivers": rivers, "waterfalls": falls}


def export_scatter(world: World, out: Path) -> list[dict]:
    sdir = out / "scatter"
    sdir.mkdir(parents=True, exist_ok=True)
    entries = []
    for kind, arr in world.scatter.items():
        if len(arr) == 0:
            continue
        p = sdir / f"{kind}.bin"
        arr.astype("<f4").tofile(p)
        entries.append({"type": kind, "count": int(len(arr)), "file": _rel(p, out),
                        "layout": "float32 x,y,z,yaw_deg,scale (OFM coordinates)"})
    return entries


def _point(o) -> dict:
    p = np.array([o.x, o.y, o.z])
    return {"position_unity": [float(v) for v in to_unity(p)],
            "position_unreal": [float(v) for v in to_unreal(p)]}


def build_manifest(world: World, vol: VolumeResult, terrain: dict, meshes: list, water: dict,
                   scatter: list, previews: dict) -> dict:
    cfg = world.cfg
    pois = []
    for p in world.pois:
        d = {"id": p.id, "kind": p.kind, "name": p.name, "description": p.description, "x": p.x, "y": p.y,
             "z": p.z, "importance": round(p.importance, 3), "region": p.region, "source": p.source,
             "props": p.props}
        d.update(_point(p))
        pois.append(d)
    landmarks = []
    for lm in world.landmarks:
        d = {"id": lm.id, "kind": lm.kind, "name": lm.name, "description": lm.description, "x": lm.x,
             "y": lm.y, "z": lm.z, "height": lm.height, "salience": round(lm.salience, 3),
             "visibility": round(lm.visibility, 3)}
        d.update(_point(lm))
        landmarks.append(d)
    caves = []
    for c in world.caves:
        d = c.to_dict()
        for e in d["entrances"]:
            p = np.array([e["x"], e["y"], e["z"]])
            e["position_unity"] = [float(v) for v in to_unity(p)]
            e["position_unreal"] = [float(v) for v in to_unreal(p)]
        caves.append(d)
    return {
        "format": "ofm-world",
        "version": 1,
        "generator": f"OFM {__version__}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "name": cfg.world.name,
        "seed": cfg.world.seed,
        "extent_m": world.extent,
        "sea_level": cfg.world.sea_level,
        "coordinate_system": {
            "ofm": "metres, x east, y north, z up (right-handed)",
            "unity": "X=x, Y=z, Z=y (metres); .ofmmesh files are already in Unity space",
            "unreal": "X=100x, Y=-100y, Z=100z (centimetres); .glb follows the glTF convention",
        },
        "mesh_mode": vol.stats["mode"],
        "terrain": terrain["unity"],
        "unreal_landscape": terrain["unreal"],
        "meshes": meshes,
        "water": water,
        "pois": pois,
        "landmarks": landmarks,
        "caves": caves,
        "features": [f.to_dict() for f in world.features],
        "scatter": scatter,
        "biomes": BIOME_NAMES,
        "strata": world.strata.describe(),
        "lore": world.lore,
        "evaluation": {
            "simulation": {k: v for k, v in world.evaluation.get("simulation", {}).items()},
            "review": world.evaluation.get("review", {}),
            "pending_actions": world.evaluation.get("pending_actions", []),
            "terrain": world.evaluation.get("terrain", {}),
            "ai_audit": world.evaluation.get("ai_audit", []),
            "ai_scope": world.evaluation.get("ai_scope", {}),
        },
        "history": world.history,
        "edits": world.edits,
        "volume": vol.stats,
        "timings_s": world.timings,
        "previews": previews,
        "config": asdict(cfg),
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")


def export_world(world: World, vol: VolumeResult, out_dir: str | Path) -> Path:
    from ..report import write_report, write_previews
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = world.cfg
    terrain = export_terrain(world, vol, out)
    meshes = export_meshes(world, vol, out, cfg.export.glb)
    water = export_water(world, out, cfg.export.glb)
    scatter = export_scatter(world, out)
    previews = write_previews(world, vol, out) if cfg.export.previews else {}
    manifest = build_manifest(world, vol, terrain, meshes, water, scatter, previews)
    path = out / "world.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    write_report(world, vol, out, manifest)
    return path
