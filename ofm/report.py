"""Human-readable outputs: preview images and a Markdown report."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .preview import save_previews
from .render3d import cave_profile, render_mesh
from .volume.density import DensityField
from .volume.mesher import surface_nets

SCORE_LABELS_KO = {
    "scenic": "경관 선호(Kaplan)", "variety": "다양성", "legibility": "랜드마크 가독성", "mystery": "미스터리",
    "pacing": "페이싱", "exploration": "탐험", "prospect_refuge": "조망-피신", "overall": "종합",
    "orientation": "방향 감각(길 잃음·되돌아감)", "terrain_quality": "지형 품질",
}


def local_mesh(world, field: DensityField, x: float, y: float, radius: float, voxel: float | None = None):
    """Mesh everything (terrain + volume) in a box around (x, y) for thumbnails."""
    from .volume.builder import _vertex_colors
    voxel = voxel or world.cfg.volume.voxel_size
    xs = np.arange(max(0.0, x - radius), min(world.extent, x + radius), voxel)
    ys = np.arange(max(0.0, y - radius), min(world.extent, y + radius), voxel)
    if len(xs) < 4 or len(ys) < 4:
        return None
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    surf = field.surface(gx, gy)
    z0, z1 = float(surf.min()) - 40.0, float(surf.max()) + 10.0
    for f in world.features:
        if abs(f.x - x) < radius + f.radius and abs(f.y - y) < radius + f.radius:
            z1 = max(z1, f.top + 8.0)
    for c in world.caves:
        lo, hi = c.bbox()
        if hi[0] > xs[0] and lo[0] < xs[-1] and hi[1] > ys[0] and lo[1] < ys[-1]:
            z0 = min(z0, float(lo[2]) - 8.0)
    zs = np.arange(z0, z1, voxel)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    D, _ = field.sample(X, Y, Z)
    D[:, :, 0] = np.maximum(D[:, :, 0], 1.0)
    mesh = surface_nets(D, np.array([xs[0], ys[0], zs[0]]), voxel)
    if mesh.is_empty():
        return None
    mesh.colors = _vertex_colors(world, field, mesh)
    return mesh


def write_previews(world, vol, out: Path) -> dict:
    pdir = out / "previews"
    files = {k: f"previews/{v}" for k, v in save_previews(world, pdir).items()}
    field = DensityField(world)
    for cave in world.caves[:4]:
        if cave.main_count < 3:
            continue
        img = cave_profile(world, field, cave)
        if img.width > 1600:
            img = img.resize((1600, max(32, int(img.height * 1600 / img.width))))
        p = pdir / f"cave_profile_{cave.id}.png"
        img.save(p)
        files[f"cave_profile_{cave.id}"] = f"previews/{p.name}"
    # 3D thumbnails of volumetric features and cave mouths
    shots = []
    for f in world.features[:4]:
        shots.append((f"feature_{f.kind}_{f.id}", f.x, f.y, f.z + f.height * 0.4, max(f.height, 30.0) * 2.4))
    for cave in world.caves[:2]:
        e = cave.entrances[0]
        shots.append((f"cave_mouth_{cave.id}", e["x"], e["y"], e["z"], 70.0))
    for name, x, y, z, dist in shots:
        mesh = local_mesh(world, field, x, y, dist * 1.6, voxel=max(world.cfg.volume.voxel_size, dist * 1.6 / 55.0))
        if mesh is None:
            continue
        eye = (x + dist * 0.7, y + dist * 0.55, z + dist * 0.45)   # light comes from +x +y
        img = render_mesh(mesh, eye, (x, y, z), width=560, height=360)
        p = pdir / f"view_{name}.png"
        img.save(p)
        files[f"view_{name}"] = f"previews/{p.name}"
    return files


def write_report(world, vol, out: Path, manifest: dict) -> None:
    import json
    sim = world.evaluation.get("simulation", {})
    review = world.evaluation.get("review", {})
    lines = [f"# OFM 월드 리포트: {world.cfg.world.name}", ""]
    lines += [f"- 시드: `{world.cfg.world.seed}`, 크기: {world.extent / 1000:.1f} km × {world.extent / 1000:.1f} km, "
              f"메시 모드: `{vol.stats['mode']}`",
              f"- 동굴 {len(world.caves)}개, 호수 {len(world.lakes)}개, 하천 {len(world.rivers)}개, "
              f"폭포 {len(world.waterfalls)}개, 지형 랜드마크 {len(world.features)}개, POI {len(world.pois)}개",
              f"- 볼륨 메시: 청크 {vol.stats['mesh_chunks']}개, 삼각형 {vol.stats['triangles']:,}개, "
              f"터레인 구멍 셀 {vol.stats['hole_cells']:,}개", ""]
    if sim:
        lines += ["## 흥미도 평가 (시뮬레이션, 0-10)", "", "| 항목 | 점수 |", "|---|---|"]
        for k, v in sim["scores"].items():
            lines.append(f"| {SCORE_LABELS_KO.get(k, k)} | {v:.2f} |")
        m = sim["metrics"]
        if m.get("bored_fraction") is None:
            lines += ["", "- 플레이테스트: 측정 불가 (출발점에서 걸을 수 있는 칸이 없거나 시뮬레이션 시간이 0)"]
        else:
            lines += ["", f"- 플레이테스트 에이전트 지루함 비율: {m['bored_fraction'] * 100:.1f}%",
                      f"- 평균/90% 이벤트 간격: {m['mean_event_gap_s']:.0f}s / {m['p90_event_gap_s']:.0f}s",
                      f"- 분당 경험 이벤트: {m['events_per_minute']:.2f}, "
                      f"시야 커버리지: {m['coverage_seen'] * 100:.0f}%"]
        lines += [f"- 랜드마크 가시 커버리지: {m['landmark_coverage'] * 100:.0f}%", ""]
        if sim.get("overall_basis") == "unweighted mean":
            lines += ["- 종합 점수: 가중치가 있는 항목이 모두 없어 가용 항목의 단순 평균을 사용했습니다.", ""]
        lines += _profile_section(sim) + _statistics_section(sim)
    terrain = world.evaluation.get("terrain") or {}
    if terrain.get("score") is not None:
        lines += _terrain_section(terrain)
    if review:
        src = review.get("source", "heuristic")
        lines += ["## AI 리뷰", "", f"- 평가 출처: `{src}`", f"- 목적함수: {review.get('objective', 0):.3f}", ""]
        if review.get("critique"):
            lines += ["> " + review["critique"], ""]
        llm = review.get("llm") or {}
        if llm.get("scores"):
            lines += ["LLM 점수: " + ", ".join(f"{k} {v:.1f}" for k, v in llm["scores"].items()), ""]
    if world.history:
        lines += ["## 디렉터 개선 이력", "", "| 반복 | 채택 | 목적함수 | 액션 |", "|---|---|---|---|"]
        for h in world.history:
            acts = "; ".join(f"{a['type']}({a.get('region', a.get('path', ''))}"
                             f"{', ' + a['kind'] if a.get('kind') else ''})" for a in h.get("actions", []))
            lines.append(f"| {h['iteration']} | {'✔' if h['accepted'] else '✘'} | {h['objective']:.3f} | {acts or '-'} |")
        lines.append("")
    audit = world.evaluation.get("ai_audit") or []
    if audit:
        lines += _audit_section(audit, world.evaluation.get("ai_scope"))
    if world.pois:
        lines += ["## 주요 장소", "", "| 이름 | 종류 | 지역 | 출처 |", "|---|---|---|---|"]
        for p in sorted(world.pois, key=lambda p: -p.importance)[:30]:
            lines.append(f"| {p.name or p.id} | {p.kind} | {p.region} | {p.source} |")
        lines.append("")
    if manifest.get("previews"):
        lines += ["## 미리보기", ""]
        for k, v in manifest["previews"].items():
            lines.append(f"![{k}]({v})")
        lines.append("")
    lines += ["## 소요 시간 (초)", "", "```", json.dumps(world.timings, indent=1), "```", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    report = {"simulation": sim, "review": review, "history": world.history, "volume": vol.stats,
              "terrain": terrain, "ai_audit": audit, "timings": world.timings}
    from .export.exporter import strict_json
    (out / "report.json").write_text(json.dumps(strict_json(report), ensure_ascii=False, indent=1, allow_nan=False,
                                                default=str), encoding="utf-8")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return "-" if v is None else str(v)


def _profile_section(sim: dict) -> list[str]:
    """Per-profile score breakdown, if the evaluation produced one."""
    profiles = sim.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        return []
    keys: list[str] = []
    for row in profiles.values():
        for k in (row.get("scores") or row):
            if isinstance((row.get("scores") or row).get(k), (int, float)) and k not in keys:
                keys.append(k)
    keys = keys[:8]
    lines = ["### 에이전트 프로필별", "", "| 프로필 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for name, row in profiles.items():
        vals = row.get("scores") or row
        lines.append(f"| {name} | " + " | ".join(_fmt(vals.get(k, "")) for k in keys) + " |")
    return lines + [""]


def _statistics_section(sim: dict) -> list[str]:
    """Monte Carlo statistics (mean, CI) for key metrics, if present."""
    stats = sim.get("statistics")
    if not isinstance(stats, dict) or not stats:
        return []
    lines = ["### 몬테카를로 통계", "", "| 지표 | 평균 | 표준편차 | 신뢰구간 |", "|---|---|---|---|"]
    for name, st in stats.items():
        if not isinstance(st, dict):
            continue
        ci = st.get("ci") or [st.get("ci_low"), st.get("ci_high")]
        ci_txt = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci and ci[0] is not None else "-"
        lines.append(f"| {name} | {_fmt(st.get('mean', '-'))} | {_fmt(st.get('std', '-'))} | {ci_txt} |")
    return lines + [""]


def _terrain_section(terrain: dict) -> list[str]:
    try:
        from .quality.metrics import format_quality_markdown
        return [format_quality_markdown(terrain, "ko"), ""]
    except (ImportError, AttributeError, KeyError, TypeError):
        lines = ["## 지형 품질", "", f"- 종합: {terrain['score']:.2f}/10", ""]
        for k, v in (terrain.get("sub_scores") or {}).items():
            lines.append(f"- {k}: {_fmt(v)}")
        for f in terrain.get("flags") or []:
            lines.append(f"- ⚠ {f}")
        return lines + [""]


STATUS_KO = {"applied": "적용", "rejected": "거절", "reverted": "되돌림", "dry_run": "모의 실행"}
STAGE_KO = {"apply": "적용 단계", "review": "제안 검토"}


def _audit_section(audit: list, scope: dict | None = None) -> list[str]:
    lines = ["## AI 범위 감사 로그", ""]
    if scope:
        allowed = ", ".join(scope.get("allowed_actions", []) or []) or "-"
        lines += [f"- 허용 액션: {allowed}"]
        if scope.get("allowed_regions"):
            lines.append(f"- 허용 지역: {', '.join(scope['allowed_regions'])}")
        if scope.get("protected_zones"):
            lines.append(f"- 보호 구역: {len(scope['protected_zones'])}곳")
        budget = scope.get("llm_budget") or {}
        if budget:
            lines.append(f"- LLM 호출: {budget.get('calls', 0)}/{budget.get('max_calls', '-')}, "
                         f"{float(budget.get('seconds', 0.0)):.0f}s/{budget.get('max_seconds', '-')}s")
        lines.append("")
    lines += ["| 반복 | 단계 | 액션 | 상태 | 사유 |", "|---|---|---|---|---|"]
    for e in audit[:60]:
        a = e.get("action") or {}
        label = a.get("type", "") if isinstance(a, dict) else str(a)
        if not label:
            label = "(형식 오류 제안)"
        if isinstance(a, dict) and (a.get("region") or a.get("path")):
            label += f"({a.get('region') or a.get('path')}{', ' + a['kind'] if a.get('kind') else ''})"
        stage = STAGE_KO.get(e.get("stage", "apply"), e.get("stage", ""))
        status = STATUS_KO.get(e.get("status", ""), e.get("status", ""))
        reason = str(e.get("reason", "")).replace("|", "/")[:90]
        lines.append(f"| {e.get('iteration', '')} | {stage} | {label} | {status} | {reason} |")
    if len(audit) > 60:
        lines.append(f"| … | {len(audit) - 60}개 더 (report.json 참고) | | |")
    return lines + [""]
