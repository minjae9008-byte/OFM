"""OFM world importer for the Unreal Editor (UE 5.x, "Python Editor Script Plugin" enabled).

Usage inside the editor (Output Log, Python):
    import ofm_unreal_import as ofm; ofm.import_world(r"C:/worlds/demo/world.json")

Headless:
    UnrealEditor-Cmd.exe MyGame.uproject -run=pythonscript
        -script="<path>/ofm_unreal_import.py --manifest <path>/world.json"

Remotely from OFM (Remote Control API plugin, port 30010):
    python -m ofm push unreal out/demo/world.json

What gets built:
  * volumetric chunks (caves, overhangs, arches, spires) as static meshes from .glb
    - in "full" mesh mode these chunks are the whole terrain (Nanite friendly);
    - in "hybrid" mode import the Landscape from terrain/heightmap_ue.png with the
      location/scale stored in the manifest ("unreal_landscape"); the script logs them;
  * lakes and river ribbons (.glb) and an ocean plane,
  * POI / landmark / cave-entrance TargetPoints tagged with their kind,
  * optional placeholder scatter (limited count) - use the scatter/*.bin data with PCG
    or foliage tools for production.

Coordinates: OFM (x east, y north, z up, metres) -> Unreal (100x, -100y, 100z) cm,
which matches how Unreal's glTF importer converts glTF Y-up assets.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys

try:
    import unreal
except ImportError:  # allows syntax checks and unit tests outside the editor
    unreal = None

ROOT_FOLDER = "OFM"
PLACEHOLDER_MESH = {
    "conifer": "/Engine/BasicShapes/Cone",
    "palm": "/Engine/BasicShapes/Cylinder",
    "dead_tree": "/Engine/BasicShapes/Cylinder",
    "cactus": "/Engine/BasicShapes/Cylinder",
    "reed": "/Engine/BasicShapes/Cylinder",
}
PLACEHOLDER_SCALE = {
    "conifer": (2.5, 2.5, 8.0), "broadleaf": (6.0, 6.0, 6.0), "palm": (0.6, 0.6, 7.0), "bush": (2.0, 2.0, 1.4),
    "boulder": (3.0, 2.6, 2.0), "cactus": (0.8, 0.8, 2.5), "reed": (0.5, 0.5, 1.0), "dead_tree": (0.6, 0.6, 4.0),
}


def to_unreal(x: float, y: float, z: float) -> tuple[float, float, float]:
    return x * 100.0, -y * 100.0, z * 100.0


def _log(msg: str) -> None:
    if unreal is not None:
        unreal.log(f"[OFM] {msg}")
    else:
        print(f"[OFM] {msg}")


def _actors():
    try:
        return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    except Exception:  # UE < 5.0
        return None


def _spawn_object(asset, location, rotation=None):
    rot = rotation or unreal.Rotator(0.0, 0.0, 0.0)
    sub = _actors()
    if sub is not None:
        return sub.spawn_actor_from_object(asset, location, rot)
    return unreal.EditorLevelLibrary.spawn_actor_from_object(asset, location, rot)


def _spawn_class(cls, location, rotation=None):
    rot = rotation or unreal.Rotator(0.0, 0.0, 0.0)
    sub = _actors()
    if sub is not None:
        return sub.spawn_actor_from_class(cls, location, rot)
    return unreal.EditorLevelLibrary.spawn_actor_from_class(cls, location, rot)


def _label(actor, label: str, folder: str, tags=()) -> None:
    actor.set_actor_label(label)
    actor.set_folder_path(f"{ROOT_FOLDER}/{folder}")
    if tags:
        actor.tags = [unreal.Name(t) for t in tags]


def import_meshes(files: list[str], destination: str) -> dict:
    """Import .glb files; returns {file: StaticMesh or None}."""
    tasks = []
    for f in files:
        task = unreal.AssetImportTask()
        task.filename = f
        task.destination_path = destination
        task.automated = True
        task.replace_existing = True
        task.save = True
        tasks.append(task)
    if tasks:
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)
    out = {}
    for f, task in zip(files, tasks):
        mesh = None
        for path in list(task.imported_object_paths or []):
            asset = unreal.load_asset(path)
            if isinstance(asset, unreal.StaticMesh):
                mesh = asset
                break
        if mesh is None:  # some importers report only the package; search the destination
            stem = os.path.splitext(os.path.basename(f))[0]
            for path in unreal.EditorAssetLibrary.list_assets(destination, recursive=True):
                if stem in path:
                    asset = unreal.load_asset(path)
                    if isinstance(asset, unreal.StaticMesh):
                        mesh = asset
                        break
        out[f] = mesh
    return out


def _read_scatter(path: str) -> list[tuple[float, float, float, float, float]]:
    data = open(path, "rb").read()
    n = len(data) // 20
    return [struct.unpack_from("<5f", data, i * 20) for i in range(n)]


def landscape_instructions(manifest: dict, base: str) -> str:
    ue = manifest.get("unreal_landscape", {})
    if not ue:
        return ""
    loc = ue["location_cm"]
    sc = ue["scale"]
    return (f"Landscape > Import from File: {os.path.join(base, ue['heightmap_png16'])}\n"
            f"  resolution {ue['resolution']}x{ue['resolution']}, location ({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f}), "
            f"scale ({sc[0]:.3f}, {sc[1]:.3f}, {sc[2]:.3f}); weight maps: terrain/ue_layer_*.png")


def import_world(manifest_path: str, meshes: bool = True, water: bool = True, markers: bool = True,
                 scatter: bool = False, scatter_limit: int = 1500, content_root: str = "/Game/OFM") -> dict:
    if unreal is None:
        raise RuntimeError("ofm_unreal_import must run inside the Unreal Editor")
    manifest_path = os.path.abspath(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    if m.get("format") != "ofm-world":
        raise ValueError(f"{manifest_path} is not an OFM world manifest")
    base = os.path.dirname(manifest_path)
    name = f"{m['name']}_{m['seed']}".replace(" ", "_")
    dest = f"{content_root}/{name}"
    summary = {"meshes": 0, "water": 0, "markers": 0, "scatter": 0}

    with unreal.ScopedSlowTask(5, "Importing OFM world") as task:
        task.make_dialog(True)

        task.enter_progress_frame(1, "Volume meshes")
        if meshes and m.get("meshes"):
            files = [os.path.join(base, e["glb"]) for e in m["meshes"] if e.get("glb")]
            imported = import_meshes(files, f"{dest}/Volume")
            for e in m["meshes"]:
                asset = imported.get(os.path.join(base, e.get("glb", "")))
                if asset is None:
                    continue
                o = e["origin"]
                actor = _spawn_object(asset, unreal.Vector(*to_unreal(*o)))
                _label(actor, e["name"], "Volume", ["OFM"] + e.get("kinds", []))
                summary["meshes"] += 1

        task.enter_progress_frame(1, "Water")
        if water:
            w = m.get("water", {})
            bodies = list(w.get("lakes", [])) + list(w.get("rivers", []))
            files = [os.path.join(base, b["glb"]) for b in bodies if b.get("glb")]
            imported = import_meshes(files, f"{dest}/Water")
            for b in bodies:
                asset = imported.get(os.path.join(base, b.get("glb", "")))
                if asset is None:
                    continue
                actor = _spawn_object(asset, unreal.Vector(*to_unreal(*b["origin"])))
                _label(actor, b.get("name") or b["id"], "Water", ["OFM", "water"])
                summary["water"] += 1
            ocean = w.get("ocean", {})
            if ocean.get("enabled"):
                plane = unreal.load_asset("/Engine/BasicShapes/Plane")
                ext = m["extent_m"]
                actor = _spawn_object(plane, unreal.Vector(*to_unreal(ext / 2, ext / 2, ocean.get("level", 0.0))))
                actor.set_actor_scale3d(unreal.Vector(ext * 3.0, ext * 3.0, 1.0))   # plane is 1 m x 1 m
                _label(actor, "Ocean", "Water", ["OFM", "ocean"])

        task.enter_progress_frame(1, "Markers")
        if markers:
            for p in m.get("pois", []):
                a = _spawn_class(unreal.TargetPoint, unreal.Vector(*to_unreal(p["x"], p["y"], p["z"])))
                _label(a, p.get("name") or p["id"], "POIs", ["OFM", "poi", p["kind"], p.get("source", "")])
                summary["markers"] += 1
            for lm in m.get("landmarks", []):
                a = _spawn_class(unreal.TargetPoint, unreal.Vector(*to_unreal(lm["x"], lm["y"], lm["z"] + lm.get("height", 0))))
                _label(a, lm.get("name") or lm["id"], "Landmarks", ["OFM", "landmark", lm["kind"]])
                summary["markers"] += 1
            for c in m.get("caves", []):
                for i, e in enumerate(c.get("entrances", [])):
                    a = _spawn_class(unreal.TargetPoint, unreal.Vector(*to_unreal(e["x"], e["y"], e["z"])))
                    _label(a, f"{c.get('name') or c['id']} #{i + 1}", "Caves", ["OFM", "cave", e.get("kind", "")])
                    summary["markers"] += 1

        task.enter_progress_frame(1, "Scatter")
        if scatter:
            for s in m.get("scatter", []):
                mesh = unreal.load_asset(PLACEHOLDER_MESH.get(s["type"], "/Engine/BasicShapes/Sphere"))
                rows = _read_scatter(os.path.join(base, s["file"]))
                stride = max(1, math.ceil(len(rows) / max(1, scatter_limit)))
                sx, sy, sz = PLACEHOLDER_SCALE.get(s["type"], (2.0, 2.0, 2.0))
                for x, y, z, yaw, scale in rows[::stride]:
                    a = _spawn_object(mesh, unreal.Vector(*to_unreal(x, y, z)), unreal.Rotator(0.0, 0.0, -yaw))
                    a.set_actor_scale3d(unreal.Vector(sx * scale, sy * scale, sz * scale))
                    _label(a, s["type"], f"Scatter/{s['type']}")
                    summary["scatter"] += 1

        task.enter_progress_frame(1, "Done")
    if m.get("mesh_mode") == "hybrid":
        _log("hybrid world: import the Landscape for ordinary ground:\n" + landscape_instructions(m, base))
    _log(f"imported '{m['name']}': {summary}")
    return summary


def _main(argv: list[str]) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Import an OFM world into the open Unreal level")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--no-meshes", action="store_true")
    ap.add_argument("--no-water", action="store_true")
    ap.add_argument("--no-markers", action="store_true")
    ap.add_argument("--scatter", action="store_true")
    ap.add_argument("--scatter-limit", type=int, default=1500)
    ap.add_argument("--save", action="store_true", help="save the current level afterwards")
    args = ap.parse_args(argv)
    import_world(args.manifest, meshes=not args.no_meshes, water=not args.no_water,
                 markers=not args.no_markers, scatter=args.scatter, scatter_limit=args.scatter_limit)
    if args.save:
        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)


if __name__ == "__main__":
    _main(sys.argv[1:])
