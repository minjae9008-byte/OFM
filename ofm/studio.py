"""OFM Studio: make, look at and rate worlds from a browser.

    python -m ofm studio                 # http://127.0.0.1:8770, worlds in out/studio
    python -m ofm view out/demo          # open one exported world's viewer

Standard library only. The studio runs each generation as its own process (the same
``python -m ofm generate`` a person would type), streams its progress to the page, lists
every world in its folder with the simulation's score next to the scores people gave it
(``feedback.json`` per world), and serves each world's ``viewer.html``.
It listens on 127.0.0.1 unless told otherwise: it runs generations on request.
"""
from __future__ import annotations

import json
import mimetypes
import re
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

PRESET_DIR = Path(__file__).resolve().parent / "presets"
PAGE = Path(__file__).resolve().parent / "viewer" / "studio.html"
SIZES = {64: "1 km · 몇 초", 128: "2 km · 약 10초", 256: "4 km · 약 1분", 512: "8 km · 수 분"}
AI_PROVIDERS = ("auto", "none", "ollama", "openai")
STAGES = ["tectonics", "strata", "fluvial", "erosion", "climate", "hydrology", "biomes", "caves_cliffs",
          "detail", "quality", "psychology", "local AI", "director", "lore", "volumetric meshing", "volume:",
          "exported"]
SCORE_KEYS = ["scenic", "variety", "mystery", "prospect_refuge", "terrain_quality", "pacing", "exploration",
              "legibility", "orientation", "overall"]


PRESET_KO = {
    "quick": "작고 빠른 섬(약 2 km). 처음 써 보거나 설정을 바꿔 볼 때.",
    "default": "온대 섬 대륙(약 4 km). 호수, 카르스트 동굴, 해식주.",
    "alpine": "높고 추운 산지. 정상의 빙하, 석회암 능선 아래 깊은 카르스트 동굴.",
    "archipelago": "따뜻한 군도. 해식절벽·해식동굴·해식주, 습한 바람받이와 건조한 비그늘.",
    "canyon": "건조한 고원. 단단한 사암 캡록과 무른 셰일이 만드는 메사, 뷰트, 후두, 계단식 협곡.",
    "designer_eval": "평가 튜닝 예시: 엄격한 페이싱 목표, 몬테카를로 3회, 6×6 지역, 사용자 정의 에이전트.",
    "scoped_ai": "AI 범위 제한 예시: 동쪽 절반에서 장소·랜드마크·동굴 추가만 허용, 시작 마을 보호.",
}


def presets() -> list[dict]:
    out = []
    for p in sorted(PRESET_DIR.glob("*.toml")):
        lines = p.read_text(encoding="utf-8").splitlines()
        desc = " ".join(ln.lstrip("# ").strip() for ln in lines[:3] if ln.startswith("#"))
        out.append({"name": p.stem, "description": PRESET_KO.get(p.stem, desc)})
    return out


@dataclass
class Job:
    id: str
    params: dict
    out: Path
    status: str = "queued"            # queued | running | done | failed | cancelled
    log: list[str] = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0
    proc: subprocess.Popen | None = None

    def progress(self) -> float:
        text = "\n".join(self.log[-40:])
        best = 0
        for k, s in enumerate(STAGES):
            if s in text:
                best = k + 1
        return 1.0 if self.status == "done" else min(0.98, best / len(STAGES))

    def to_dict(self, tail: int = 200) -> dict:
        return {"id": self.id, "params": self.params, "world": self.out.name, "status": self.status,
                "progress": round(self.progress(), 3), "log": self.log[-tail:],
                "elapsed": round((self.finished or time.time()) - self.started, 1) if self.started else 0.0}


class Studio:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        threading.Thread(target=self._worker, daemon=True).start()

    # -- generation ------------------------------------------------------------------
    def submit(self, params: dict) -> Job:
        names = {p["name"] for p in presets()}
        preset = str(params.get("preset", "quick"))
        if preset not in names:
            raise ValueError(f"알 수 없는 프리셋: {preset}")
        seed = int(params.get("seed", 1))
        if not 0 <= seed < 2 ** 31:
            raise ValueError("시드는 0 이상 2147483647 이하의 정수여야 합니다")
        size = int(params.get("size", 128))
        if size not in SIZES:
            raise ValueError(f"크기는 {sorted(SIZES)} 중 하나여야 합니다")
        iterations = int(params.get("iterations", 1))
        if not 0 <= iterations <= 8:
            raise ValueError("AI 개선 반복은 0–8회입니다")
        ai = str(params.get("ai", "auto"))
        if ai not in AI_PROVIDERS:
            raise ValueError(f"AI 공급자는 {', '.join(AI_PROVIDERS)} 중 하나입니다")
        mesh = str(params.get("mesh_mode", "hybrid"))
        if mesh not in ("hybrid", "full"):
            raise ValueError("메시 모드는 hybrid 또는 full입니다")
        clean = {"preset": preset, "seed": seed, "size": size, "iterations": iterations, "ai": ai, "mesh_mode": mesh}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        with self.lock:
            jid = f"job{len(self.jobs) + 1}"
            out = self.root / f"{preset}_s{seed}_{size}_{stamp}"
            job = Job(id=jid, params=clean, out=out)
            self.jobs[jid] = job
            self.wake.notify()
        return job

    def cancel(self, jid: str) -> bool:
        with self.lock:
            job = self.jobs.get(jid)
            if job is None or job.status not in ("queued", "running"):
                return False
            if job.proc is not None:
                job.proc.terminate()
            job.status = "cancelled"
            job.finished = time.time()
        return True

    def _worker(self) -> None:
        while True:
            with self.lock:
                job = next((j for j in self.jobs.values() if j.status == "queued"), None)
                if job is None:
                    self.wake.wait()
                    continue
                job.status = "running"
                job.started = time.time()
            p = job.params
            cmd = [sys.executable, "-u", "-m", "ofm", "generate", "--preset", p["preset"], "--seed", str(p["seed"]),
                   "--size", str(p["size"]), "--iterations", str(p["iterations"]), "--ai", p["ai"],
                   "--mesh-mode", p["mesh_mode"], "--out", str(job.out)]
            job.log.append("$ " + " ".join(cmd[1:]))
            try:
                job.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                            encoding="utf-8", errors="replace", bufsize=1)
                for line in job.proc.stdout:
                    job.log.append(line.rstrip())
                code = job.proc.wait()
            except OSError as exc:
                job.log.append(f"실행 실패: {exc}")
                code = -1
            with self.lock:
                if job.status != "cancelled":
                    job.status = "done" if code == 0 and (job.out / "viewer.html").exists() else "failed"
                job.finished = time.time()
                job.proc = None

    # -- worlds ----------------------------------------------------------------------
    def worlds(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            manifest = d / "world.json"
            if not d.is_dir() or not manifest.exists():
                continue
            try:
                m = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            sim = (m.get("evaluation") or {}).get("simulation") or {}
            fb = self.feedback(d)
            human = {}
            for k in SCORE_KEYS:
                vals = [f["ratings"][k] for f in fb if isinstance(f.get("ratings"), dict) and k in f["ratings"]]
                if vals:
                    human[k] = round(sum(vals) / len(vals), 2)
            terrain = (m.get("evaluation") or {}).get("terrain") or {}
            out.append({
                "dir": d.name, "name": m.get("name"), "seed": m.get("seed"), "extent_m": m.get("extent_m"),
                "created": m.get("created"), "viewer": (d / "viewer.html").exists(),
                "scores": sim.get("scores") or {}, "terrain_flags": terrain.get("flags") or [],
                "human": human, "ratings": len(fb),
                "counts": {"caves": len(m.get("caves", [])), "lakes": len((m.get("water") or {}).get("lakes", [])),
                           "pois": len(m.get("pois", []))},
            })
        return out

    def feedback(self, d: Path) -> list[dict]:
        p = d / "feedback.json"
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return data if isinstance(data, list) else [data]

    def add_feedback(self, world: str, fb: dict) -> Path:
        d = self.world_dir(world)
        if d is None:
            raise ValueError("그런 월드가 없습니다")
        ratings = fb.get("ratings")
        if not isinstance(ratings, dict) or not ratings:
            raise ValueError("평가 점수가 없습니다")
        clean = {k: float(v) for k, v in ratings.items() if k in SCORE_KEYS and 0.0 <= float(v) <= 10.0}
        entry = {"rated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tester": str(fb.get("tester", ""))[:80],
                 "ratings": clean, "notes": str(fb.get("notes", ""))[:8000], "simulation": fb.get("simulation") or {}}
        with self.lock:
            entries = self.feedback(d) + [entry]
            (d / "feedback.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
        return d / "feedback.json"

    def world_dir(self, name: str) -> Path | None:
        if not name or name in (".", "..") or re.search(r"[/\\\x00]", name):
            return None
        d = (self.root / name).resolve()
        if d.parent != self.root.resolve() or not (d / "world.json").exists():
            return None
        return d


def _handler(studio: Studio | None, view_root: Path | None = None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OFMStudio/1"

        def log_message(self, fmt, *args):      # quiet: the page shows what matters
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _file(self, path: Path) -> None:
            if not path.is_file():
                self._send(404, b"not found", "text/plain")
                return
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
                ctype += "; charset=utf-8"
            self._send(200, path.read_bytes(), ctype)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 2_000_000:
                raise ValueError("요청이 너무 큽니다")
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                if studio is None and view_root is not None:
                    self.send_response(HTTPStatus.FOUND)
                    self.send_header("Location", f"/worlds/{view_root.name}/viewer.html")
                    self.end_headers()
                    return
                return self._file(PAGE)
            if path == "/api/ping":
                return self._json({"ok": True, "studio": studio is not None})
            if path.startswith("/worlds/"):
                rest = path[len("/worlds/"):]
                name, _, sub = rest.partition("/")
                base = view_root if (studio is None and view_root is not None and name == view_root.name) else \
                    (studio.world_dir(name) if studio is not None else None)
                if base is None:
                    return self._send(404, "그런 월드가 없습니다".encode("utf-8"), "text/plain; charset=utf-8")
                target = (base / (sub or "viewer.html")).resolve()
                if base.resolve() not in target.parents and target != base.resolve():
                    return self._send(403, b"forbidden", "text/plain")
                return self._file(target)
            if studio is None:
                return self._send(404, b"not found", "text/plain")
            if path == "/api/presets":
                return self._json({"presets": presets(), "sizes": [{"size": k, "label": v} for k, v in SIZES.items()],
                                   "ai": list(AI_PROVIDERS)})
            if path == "/api/worlds":
                return self._json({"worlds": studio.worlds(), "root": str(studio.root.resolve())})
            if path == "/api/jobs":
                return self._json({"jobs": [j.to_dict(tail=3) for j in studio.jobs.values()]})
            if path.startswith("/api/jobs/"):
                job = studio.jobs.get(path.rsplit("/", 1)[-1])
                return self._json(job.to_dict()) if job else self._json({"error": "그런 작업이 없습니다"}, 404)
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            if studio is None and path != "/api/feedback":
                return self._send(404, b"not found", "text/plain")
            try:
                body = self._body()
                if path == "/api/generate":
                    return self._json(studio.submit(body).to_dict())
                if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    return self._json({"cancelled": studio.cancel(path.split("/")[3])})
                if path == "/api/feedback":
                    if studio is None:
                        # view-only server: save next to the world it serves
                        target = Studio.__new__(Studio)
                        target.root, target.lock = view_root.parent, threading.Lock()
                        return self._json({"saved": str(target.add_feedback(view_root.name, body.get("feedback") or {}))})
                    return self._json({"saved": str(studio.add_feedback(str(body.get("world", "")), body.get("feedback") or {}))})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                return self._json({"error": str(exc)}, 400)
            self._send(404, b"not found", "text/plain")

    return Handler


def serve(root: str | Path = "out/studio", host: str = "127.0.0.1", port: int = 8770, open_browser: bool = True) -> None:
    studio = Studio(Path(root))
    httpd = _bind(host, port, _handler(studio))
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{httpd.server_address[1]}/"
    print(f"OFM 스튜디오: {url}  (월드 폴더: {studio.root.resolve()})", flush=True)
    print("브라우저에서 위 주소를 여세요. 끝내려면 Ctrl+C.", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def view(target: str | Path, serve_http: bool = False, host: str = "127.0.0.1", port: int = 8771,
         open_browser: bool = True) -> int:
    """Open an exported world's viewer: as a local file, or through a small server (for a
    machine without a desktop browser, or to save ratings straight into the world folder)."""
    p = Path(target)
    d = p.parent if p.is_file() else p
    page = d / "viewer.html"
    if not page.exists():
        print(f"{page}가 없습니다. 먼저 `python -m ofm generate --out {d}`로 월드를 만드세요.", file=sys.stderr)
        return 2
    if not serve_http:
        url = page.resolve().as_uri()
        print(f"뷰어: {url}", flush=True)
        if open_browser:
            webbrowser.open(url)
        return 0
    httpd = _bind(host, port, _handler(None, d.resolve()))
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{httpd.server_address[1]}/"
    print(f"뷰어: {url}  (Ctrl+C로 끝내기)", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def _bind(host: str, port: int, handler) -> ThreadingHTTPServer:
    last = None
    for p in range(port, port + 20):            # the next free port if this one is taken
        try:
            httpd = ThreadingHTTPServer((host, p), handler)
            httpd.daemon_threads = True
            return httpd
        except OSError as exc:
            last = exc
    raise OSError(f"포트 {port}–{port + 19}를 열 수 없습니다: {last}")
