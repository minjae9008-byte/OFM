"""Preview renderers (PNG) used for reports and for multimodal LLM evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .grid import gradient, normalize01
from .physics.biome import BIOME_COLORS


def hillshade(h: np.ndarray, cell: float, azimuth: float = 315.0, altitude: float = 45.0,
              exaggeration: float = 1.5) -> np.ndarray:
    gx, gy = gradient(h * exaggeration, cell)
    az = np.radians(azimuth)
    alt = np.radians(altitude)
    # Light direction in (east, north, up)
    lx, ly, lz = np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)
    nz = 1.0 / np.sqrt(1.0 + gx * gx + gy * gy)
    shade = (-gx * lx - gy * ly + lz) * nz
    return np.clip(shade, 0.0, 1.0)


def _to_image(rgb: np.ndarray) -> Image.Image:
    # rows grow north; flip so north is up in the image
    return Image.fromarray(np.clip(rgb[::-1], 0, 255).astype(np.uint8))


def render_map(world, overlay: str = "biome", markers: bool = True, scale: int = 2) -> Image.Image:
    h = world.height
    shade = hillshade(h, world.cell)
    if overlay == "biome" and world.biome is not None:
        base = BIOME_COLORS[world.biome].astype(np.float64)
    elif overlay == "interest" and "interest" in world.psych:
        v = world.psych["interest"]
        base = _ramp(v)
    elif overlay == "boredom" and "boredom" in world.psych:
        base = _ramp(world.psych["boredom"], cold=(40, 60, 70), hot=(250, 60, 40))
    else:
        t = normalize01(h)
        base = np.stack([t * 200 + 40, t * 180 + 60, t * 120 + 60], axis=-1)
    rgb = base * (0.45 + 0.75 * shade[..., None])
    if world.ocean is not None:
        rgb[world.ocean] = np.array([38, 74, 122]) * (0.9 + 0.2 * shade[world.ocean, None])
    if world.lake_mask is not None:
        rgb[world.lake_mask] = (60, 116, 168)
    if world.river_mask is not None:
        rgb[world.river_mask & ~world.water] = (70, 130, 190)
    img = _to_image(rgb)
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    if markers:
        draw = ImageDraw.Draw(img)
        px = scale / world.cell

        def pt(x, y):
            return x * px, img.height - y * px

        for cave in world.caves:
            for e in cave.entrances:
                cx, cy = pt(e["x"], e["y"])
                draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(20, 20, 20), outline=(255, 255, 255))
        for f in world.waterfalls:
            cx, cy = pt(f["x"], f["y"])
            draw.polygon([(cx, cy - 4), (cx - 3, cy + 3), (cx + 3, cy + 3)], fill=(150, 220, 255))
        for lm in world.landmarks:
            cx, cy = pt(lm.x, lm.y)
            draw.polygon([(cx, cy - 6), (cx - 5, cy + 4), (cx + 5, cy + 4)], outline=(255, 230, 80), fill=(120, 80, 20))
        for poi in world.pois:
            cx, cy = pt(poi.x, poi.y)
            draw.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], fill=(230, 60, 60), outline=(255, 255, 255))
        for feat in world.features:
            cx, cy = pt(feat.x, feat.y)
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=(255, 120, 255), width=2)
    return img


def _ramp(v: np.ndarray, cold=(30, 40, 90), hot=(255, 210, 60)) -> np.ndarray:
    t = normalize01(v)[..., None]
    return np.array(cold) * (1 - t) + np.array(hot) * t


def save_previews(world, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for overlay in ("biome", "interest", "boredom", "height"):
        if overlay in ("interest", "boredom") and overlay not in world.psych:
            continue
        img = render_map(world, overlay=overlay, markers=overlay != "height")
        path = out_dir / f"preview_{overlay}.png"
        img.save(path)
        files[overlay] = path.name
    return files


def png_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
