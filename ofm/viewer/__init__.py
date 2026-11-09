"""Browser viewer for an exported world: ``viewer.html`` next to ``world.json``.

The page needs neither a server nor the internet. Its data is a script,
``viewer_data.js``, which browsers load from local files (they refuse ``fetch`` on
``file://``); binary arrays travel inside it as base64. It shows what the engines
receive: the detail heightfield with its holes, the volume meshes (caves, arches,
overhangs) that fill them, the water sheets and ribbons, every marker, the play-test
agents' routes and the evaluation, plus a form for a person to rate the world.
"""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ..grid import bilinear, normalize01
from ..physics.biome import BIOME_COLORS, BIOME_NAMES, MATERIALS

TEMPLATE_DIR = Path(__file__).resolve().parent
VIEWER_MAX_VERTS = 1025          # terrain vertices per side shown (finer grids are strided)
VIEWER_MAX_TRIANGLES = 2_500_000  # volume triangles kept (largest chunks first beyond this)

SCORE_KEYS = ["scenic", "variety", "mystery", "prospect_refuge", "terrain_quality", "pacing",
              "exploration", "legibility", "orientation", "overall"]


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _terrain(world, vol) -> dict:
    from ..export.exporter import _cells_at_vertices
    h0 = world.hires
    n0 = h0.shape[0]
    stride = max(1, int(np.ceil((n0 - 1) / (VIEWER_MAX_VERTS - 1))))
    h = h0[::stride, ::stride]
    n = h.shape[0]
    hmin, hmax = float(h.min()), float(h.max())
    q = np.round((h - hmin) / max(hmax - hmin, 1e-6) * 65535.0).astype("<u2")
    w = np.stack([np.clip(_cells_at_vertices(world.splat[..., k], n, world), 0.0, 1.0)
                  for k in range(len(MATERIALS))], axis=-1)
    w /= np.maximum(w.sum(axis=-1, keepdims=True), 1e-6)
    pal = np.array([c for _, c in MATERIALS], dtype=np.float64)
    rgb = np.clip(w @ pal, 0, 255).astype(np.uint8)
    holes = np.zeros((n - 1, n - 1), dtype=np.uint8)
    if vol is not None and vol.holes is not None and vol.holes.any():
        hv = vol.holes[: (n - 1) * stride, : (n - 1) * stride]
        holes = hv.reshape(n - 1, stride, n - 1, stride).any(axis=(1, 3)).astype(np.uint8)
    return {"n": n, "spacing": world.hires_spacing * stride, "hmin": hmin, "hmax": hmax,
            "h": _b64(q), "rgb": _b64(rgb), "holes": _b64(holes)}


def _volume(vol) -> list[dict]:
    if vol is None or not vol.chunks:
        return []
    chunks = sorted(vol.chunks, key=lambda c: -len(c.mesh.indices))
    total = sum(len(c.mesh.indices) for c in chunks)
    out = []
    kept = 0
    for ch in chunks:
        m = ch.mesh
        if not len(m.indices):
            continue
        if total > VIEWER_MAX_TRIANGLES and kept + len(m.indices) > VIEWER_MAX_TRIANGLES:
            continue
        kept += len(m.indices)
        p = np.asarray(m.positions, dtype=np.float64)
        lo, hi = p.min(axis=0), p.max(axis=0)
        span = np.maximum(hi - lo, 1e-3)
        qp = np.round((p - lo) / span * 65535.0).astype("<u2")
        idx = np.asarray(m.indices).reshape(-1)
        wide = len(p) > 65535
        col = (np.asarray(m.colors)[:, :3] if m.colors is not None
               else np.full((len(p), 3), 128)).astype(np.uint8)
        out.append({"lo": [float(v) for v in lo], "span": [float(v) for v in span],
                    "p": _b64(qp), "i": _b64(idx.astype("<u4" if wide else "<u2")), "wide": wide,
                    "c": _b64(col), "kinds": list(ch.kinds)})
    return out


def _water(world) -> dict:
    from ..export.water import creek_meshes, lake_meshes, river_meshes
    lakes = []
    for lake, mesh, _ in lake_meshes(world):
        lakes.append({"id": lake["id"], "name": lake.get("name", ""), "level": float(lake["level"]),
                      "xy": _b64(mesh.positions[:, :2].astype("<f4")),
                      "i": _b64(mesh.indices.reshape(-1).astype("<u4"))})
    rivers = []
    for river, mesh, _ in list(river_meshes(world)) + list(creek_meshes(world)):
        rivers.append({"id": river["id"], "name": river.get("name", ""), "kind": river.get("kind", "river"),
                       "xyz": _b64(mesh.positions.astype("<f4")),
                       "i": _b64(mesh.indices.reshape(-1).astype("<u4"))})
    return {"sea": float(world.cfg.world.sea_level), "ocean": bool(world.ocean is not None and world.ocean.any()),
            "lakes": lakes, "rivers": rivers}


def _agents(world) -> dict | None:
    psych = world.psych or {}
    per = psych.get("perception")
    logs = psych.get("agents") or []
    if per is None or not logs:
        return None
    paths = []
    for log in logs[:64]:
        if len(log.path) < 2:
            continue
        paths.append({"profile": log.profile or log.personality, "run": log.run,
                      "bored": round(float(log.bored_fraction), 3),
                      "cells": _b64(np.asarray(log.path, dtype="<u4"))})
    spawn = None
    if world.spawn is not None:
        spawn = [float(world.spawn[0]), float(world.spawn[1])]
    return {"grid": int(per.grid), "cell": float(per.cell), "paths": paths, "spawn": spawn}


def _ramp(v, stops) -> np.ndarray:
    t = np.clip(normalize01(v), 0.0, 1.0)
    xs = np.linspace(0.0, 1.0, len(stops))
    stops = np.asarray(stops, dtype=np.float64)
    return np.stack([np.interp(t, xs, stops[:, c]) for c in range(3)], axis=-1)


def _layer_images(world, out: Path) -> list[dict]:
    """2D map layers at the detail resolution, north up, without baked-in markers."""
    from ..detail.lakes import lake_surface
    from ..preview import hillshade
    h = world.hires
    n = h.shape[0]
    sp = world.hires_spacing
    shade = hillshade(h, sp, exaggeration=1.2)[..., None]
    c = np.arange(n) * sp / world.cell - 0.5

    def cells(field):
        return bilinear(np.asarray(field, dtype=np.float64), c[None, :], c[:, None])

    sea = world.cfg.world.sea_level
    ocean = (h < sea) & (cells(world.ocean) > 0.3) if world.ocean is not None else np.zeros_like(h, bool)
    lakes = np.zeros_like(h, dtype=bool)
    surf = lake_surface(world)
    if surf is not None and np.isfinite(surf).any():
        lv = np.full((n, n), -np.inf)
        s = np.where(np.isfinite(surf), surf, -np.inf)
        for dj in (0, 1):
            for di in (0, 1):
                lv[dj:dj + n - 1, di:di + n - 1] = np.maximum(lv[dj:dj + n - 1, di:di + n - 1], s)
        lakes = h < lv
    river = cells(world.river_mask) > 0.45 if world.river_mask is not None else np.zeros_like(h, bool)

    def water(rgb):
        rgb = rgb * (0.42 + 0.78 * shade)
        rgb[ocean] = np.array([34, 70, 118]) * (0.85 + 0.25 * shade[ocean])
        rgb[lakes] = (58, 112, 166)
        rgb[river & ~ocean & ~lakes] = (72, 136, 196)
        return rgb

    layers = []
    vdir = out / "viewer"
    vdir.mkdir(parents=True, exist_ok=True)

    def save(key, label, rgb, legend):
        img = Image.fromarray(np.clip(rgb[::-1], 0, 255).astype(np.uint8))
        p = vdir / f"map_{key}.png"
        img.save(p, optimize=True)
        layers.append({"key": key, "label": label, "file": f"viewer/{p.name}", "legend": legend})

    land = h >= sea
    t = normalize01(np.where(land, h, sea))
    relief = _ramp(t, [(70, 110, 60), (140, 150, 90), (175, 150, 110), (150, 140, 130), (245, 245, 245)])
    save("relief", "지형(고도)", water(relief),
         [f"{sea:.0f} m", f"{float(h.max()):.0f} m"])
    if world.biome is not None:
        save("biome", "바이옴", water(cells_nearest(world, BIOME_COLORS[world.biome].astype(np.float64), n)),
             sorted({BIOME_NAMES[b] for b in np.unique(world.biome)}))
    if world.precipitation is not None:
        save("precip", "강수량", water(_ramp(cells(world.precipitation), [(150, 110, 60), (200, 190, 120),
                                                                        (90, 170, 120), (40, 90, 170)])),
             [f"{float(world.precipitation.min()):.0f} mm", f"{float(world.precipitation.max()):.0f} mm"])
    if world.temperature is not None:
        save("temp", "기온", water(_ramp(cells(world.temperature), [(60, 80, 170), (120, 190, 200),
                                                                   (240, 220, 120), (210, 70, 50)])),
             [f"{float(world.temperature.min()):.1f} °C", f"{float(world.temperature.max()):.1f} °C"])
    psych = world.psych or {}
    if "interest" in psych:
        save("interest", "흥미도(시뮬레이션)", water(_ramp(cells(psych["interest"]), [(30, 40, 90), (120, 80, 160),
                                                                               (255, 210, 60)])),
             ["낮음", "높음"])
    if "boredom" in psych:
        save("boredom", "지루함(에이전트)", water(_ramp(cells(psych["boredom"]), [(40, 60, 70), (200, 120, 60),
                                                                           (250, 60, 40)])),
             ["없음", "많음"])
    return layers


def cells_nearest(world, field_rgb: np.ndarray, n: int) -> np.ndarray:
    """A per-cell RGB field on the detail vertices (nearest cell: classes do not blend)."""
    idx = np.clip((np.arange(n) * world.hires_spacing / world.cell).astype(int), 0, world.size - 1)
    return field_rgb[np.ix_(idx, idx)]


def viewer_payload(world, vol, manifest: dict, report: dict | None = None, layers=None) -> dict:
    from ..export.exporter import strict_json
    sim = (report or {}).get("simulation") or world.evaluation.get("simulation", {})
    keep = {k: manifest.get(k) for k in ("name", "seed", "extent_m", "sea_level", "mesh_mode", "pois", "landmarks",
                                       "features", "spawn", "lore", "history", "volume", "timings_s", "created")}
    keep["region_divs"] = int(getattr(world, "region_divs", 4))
    keep["caves"] = [{k: c[k] for k in ("id", "kind", "name", "description", "length_m", "entrances", "segments",
                                        "chambers")} for c in manifest.get("caves", [])]
    keep["waterfalls"] = manifest.get("water", {}).get("waterfalls", [])
    keep["lakes"] = [{k: lk.get(k) for k in ("id", "name", "level", "area_m2", "max_depth", "closed_basin")}
                     for lk in manifest.get("water", {}).get("lakes", [])]
    keep["rivers"] = [{"id": r["id"], "name": r.get("name", ""), "length_m": r.get("length_m"),
                       "kind": r.get("kind", "river")} for r in manifest.get("water", {}).get("rivers", [])]
    keep["scatter"] = [{"type": s["type"], "count": s["count"]} for s in manifest.get("scatter", [])]
    payload = {
        "format": "ofm-viewer", "version": 1,
        "world": keep,
        "evaluation": {
            "scores": sim.get("scores", {}), "metrics": sim.get("metrics", {}),
            "profiles": sim.get("profiles", {}), "regions": sim.get("regions", {}),
            "terrain": (report or {}).get("terrain") or world.evaluation.get("terrain", {}),
            "review": (report or {}).get("review") or world.evaluation.get("review", {}),
            "score_keys": SCORE_KEYS,
        },
        "terrain": _terrain(world, vol),
        "volume": _volume(vol),
        "water": _water(world),
        "agents": _agents(world),
        "layers": layers or [],
    }
    return strict_json(payload)


def write_viewer(world, vol, out: str | Path, manifest: dict, report: dict | None = None) -> Path:
    """Write ``viewer.html``, ``viewer_data.js`` and the 2D map layers into ``out``."""
    out = Path(out)
    layers = _layer_images(world, out)
    payload = viewer_payload(world, vol, manifest, report, layers)
    js = "window.OFM_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                                          allow_nan=False) + ";\n"
    (out / "viewer_data.js").write_text(js, encoding="utf-8")
    page = out / "viewer.html"
    shutil.copyfile(TEMPLATE_DIR / "viewer.html", page)
    return page
