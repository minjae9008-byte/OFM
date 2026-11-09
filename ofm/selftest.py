"""``python -m ofm selftest``: does this installation work? (no pytest needed)

Generates a small world end to end into a temporary folder, exports it with the viewer,
and checks the invariants a tester would otherwise have to eyeball: the files the engines
and the viewer read, physically consistent water, nothing placed inside rock or under
water, scores in range. Prints one line per check and a verdict.
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _checks(world, out: Path) -> list[tuple[str, bool, str]]:
    from .detail.lakes import lake_surface, under_lake
    from .detail.pois import river_surface
    from .grid import NEIGHBORS8, shifted
    from .volume.features import rock_at
    res = []

    def check(name, ok, detail=""):
        res.append((name, bool(ok), detail))

    for f in ("world.json", "report.json", "report.md", "viewer.html", "viewer_data.js",
              "terrain/heightmap.r16", "terrain/heightmap.png"):
        check(f"파일 {f}", (out / f).exists())
    try:
        js = (out / "viewer_data.js").read_text(encoding="utf-8")
        data = json.loads(js[js.index("=") + 1:].rstrip().rstrip(";"))
        t = data["terrain"]
        n = len(base64.b64decode(t["h"])) // 2
        check("뷰어 데이터 해석", n == t["n"] * t["n"], f"지형 {t['n']}×{t['n']}, 볼륨 청크 {len(data['volume'])}")
    except (OSError, ValueError, KeyError) as exc:
        check("뷰어 데이터 해석", False, str(exc))
    m = json.loads((out / "world.json").read_text(encoding="utf-8"))
    check("world.json 형식", m.get("format") == "ofm-world" and m["terrain"]["heightmap_resolution"] > 1)

    # water: lakes cover their basins and keep their shores; rivers never flow uphill
    if world.lake_mask is not None and world.lake_mask.any():
        lv = np.where(world.lake_mask, world.lake_level, -np.inf)
        rim = np.full(lv.shape, -np.inf)
        for dy, dx in NEIGHBORS8:
            rim = np.maximum(rim, shifted(lv, dy, dx, -np.inf))
        rec = world.receiver.reshape(-1)
        into = np.zeros(rec.size, dtype=bool)
        into[rec >= 0] = world.lake_mask.reshape(-1)[rec[rec >= 0]]
        shore = np.isfinite(rim) & ~world.lake_mask & ~world.ocean & ~(world.river_mask & into.reshape(lv.shape))
        check("호숫가가 호수 수위보다 낮지 않음", (world.height[shore] >= rim[shore] - 1e-6).all())
    up = [r["id"] for r in world.rivers if (np.diff(np.array(r["points"])[:, 2]) > 1e-6).any()]
    check("하천이 오르막으로 흐르지 않음", not up, ", ".join(up))
    surf = lake_surface(world)
    bad = [p.kind for p in world.pois if p.kind != "cave_entrance" and (
        rock_at(world, p.x, p.y) or under_lake(world, p.x, p.y, p.z, surf)
        or p.z < (river_surface(world, p.x, p.y) or -np.inf) - 1e-6)]
    check("장소가 바위 속·물 아래에 없음", not bad, ", ".join(bad))
    sc = sum(int(under_lake(world, a[:, 0], a[:, 1], a[:, 2], surf).sum()) for a in world.scatter.values() if len(a))
    check("식생이 호수 아래에 없음", sc == 0, f"{sc}개")
    for f in world.features:
        x, y = f.footprint()
        if f.z - f.foot > float(world.ground_at(x, y).min()) - 1.99:
            check(f"지형 요소 {f.id} 접지", False)
            break
    else:
        check("지형 요소가 지면에 닿음", True, f"{len(world.features)}개")
    scores = (world.evaluation.get("simulation") or {}).get("scores") or {}
    check("평가 점수 0–10", scores and all(v is None or 0.0 <= v <= 10.0 for v in scores.values()),
          f"종합 {scores.get('overall')}")
    return res


def run(keep: str | None = None) -> int:
    from .config import load_config
    from .export.exporter import export_world
    from .pipeline import generate
    from .volume.builder import build_volume
    t0 = time.perf_counter()
    print("OFM 자가진단: 1 km 월드를 만들어 내보내고 결과를 검사합니다…", flush=True)
    cfg = load_config(Path(__file__).resolve().parent / "presets" / "quick.toml",
                      {"world.size": 64, "world.seed": 11, "psych.grid": 24, "psych.agents": 3,
                       "psych.agent_minutes": 6.0, "volume.voxel_size": 8.0, "volume.chunk_voxels": 16,
                       "ai.provider": "none", "ai.iterations": 0, "hydrology.river_min_discharge": 0.1})
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(keep) if keep else Path(tmp) / "selftest"
        world = generate(cfg)
        vol = build_volume(world)
        export_world(world, vol, out)
        results = _checks(world, out)
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"  {'통과' if ok else '실패'}  {name.ljust(width)}  {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"{'모두 통과' if not failed else f'{len(failed)}개 실패'} ({time.perf_counter() - t0:.1f}초)"
          + (f" · 결과: {out}" if keep else ""), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(run())
