"""Human testing: the browser viewer in every export, the studio web app, the self-test,
and creeks (streams too small to carve, draped as thin water)."""
import base64
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from ofm.grid import bilinear

from .conftest import small_config


def _payload(out: Path) -> dict:
    js = (out / "viewer_data.js").read_text(encoding="utf-8")
    assert js.startswith("window.OFM_DATA = ")
    return json.loads(js[len("window.OFM_DATA = "):].rstrip().rstrip(";"))


@pytest.fixture(scope="module")
def exported(small_world, tmp_path_factory):
    from ofm.export.exporter import export_world
    from ofm.volume.builder import build_volume
    out = tmp_path_factory.mktemp("viewer")
    export_world(small_world, build_volume(small_world), out)
    return out


def test_export_writes_a_self_contained_viewer(small_world, exported):
    out = exported
    page = (out / "viewer.html").read_text(encoding="utf-8")
    assert '<script src="viewer_data.js"></script>' in page
    # offline: nothing is fetched from the network
    assert "http://" not in page.replace("http://' + location", "") and "https://" not in page
    d = _payload(out)
    t = d["terrain"]
    n = t["n"]
    assert n == small_world.hires.shape[0]
    h = np.frombuffer(base64.b64decode(t["h"]), dtype="<u2")
    assert h.size == n * n
    heights = t["hmin"] + h.astype(float) / 65535 * (t["hmax"] - t["hmin"])
    assert np.allclose(heights.reshape(n, n), small_world.hires, atol=(t["hmax"] - t["hmin"]) / 65535 + 1e-6)
    assert len(base64.b64decode(t["rgb"])) == n * n * 3
    assert len(base64.b64decode(t["holes"])) == (n - 1) ** 2
    for ch in d["volume"]:
        p = np.frombuffer(base64.b64decode(ch["p"]), dtype="<u2")
        idx = np.frombuffer(base64.b64decode(ch["i"]), dtype="<u4" if ch["wide"] else "<u2")
        assert p.size % 3 == 0 and idx.size % 3 == 0 and idx.max() < p.size // 3
    w = d["world"]
    assert len(w["pois"]) == len(small_world.pois) and len(w["features"]) == len(small_world.features)
    for layer in d["layers"]:
        assert (out / layer["file"]).exists(), layer
    assert {"relief", "biome"} <= {layer["key"] for layer in d["layers"]}
    assert set(d["evaluation"]["scores"]) >= {"overall"}


def _serve(studio_root: Path):
    from ofm.studio import Studio, _bind, _handler
    studio = Studio(studio_root)
    httpd = _bind("127.0.0.1", 0, _handler(studio))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return studio, httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read()


def _post(url: str, body: dict):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_studio_generates_serves_and_records_ratings(tmp_path):
    studio, httpd, base = _serve(tmp_path / "worlds")
    try:
        assert json.loads(_get(base + "/api/ping")[1])["studio"] is True
        assert _post(base + "/api/generate", {"preset": "nope", "size": 64})[0] == 400
        assert _post(base + "/api/generate", {"preset": "quick", "size": 100})[0] == 400
        code, job = _post(base + "/api/generate", {"preset": "quick", "seed": 3, "size": 64, "iterations": 0, "ai": "none"})
        assert code == 200
        deadline = time.time() + 240
        while time.time() < deadline:
            j = json.loads(_get(f"{base}/api/jobs/{job['id']}")[1])
            if j["status"] not in ("queued", "running"):
                break
            time.sleep(0.5)
        assert j["status"] == "done", "\n".join(j["log"][-20:])
        worlds = json.loads(_get(base + "/api/worlds")[1])["worlds"]
        assert len(worlds) == 1 and worlds[0]["viewer"]
        name = worlds[0]["dir"]
        code, page = _get(f"{base}/worlds/{name}/viewer.html")
        assert code == 200 and b"viewer_data.js" in page
        for bad in (f"/worlds/{name}/../../etc/passwd", "/worlds/../secret", f"/worlds/{name}/%2e%2e/%2e%2e/x"):
            with pytest.raises(urllib.error.HTTPError) as e:
                _get(base + bad)
            assert e.value.code in (403, 404)
        ratings = {"scenic": 6.5, "overall": 7.0, "bogus": 3, "mystery": 11}
        code, res = _post(base + "/api/feedback", {"world": name, "feedback": {"ratings": ratings, "notes": "좋음", "tester": "t"}})
        assert code == 200
        fb = json.loads((tmp_path / "worlds" / name / "feedback.json").read_text(encoding="utf-8"))
        assert fb[0]["ratings"] == {"scenic": 6.5, "overall": 7.0}          # unknown keys and out-of-range values dropped
        w = json.loads(_get(base + "/api/worlds")[1])["worlds"][0]
        assert w["ratings"] == 1 and w["human"]["overall"] == 7.0
        assert _post(base + "/api/feedback", {"world": "../x", "feedback": {"ratings": {"overall": 5}}})[0] == 400
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_selftest_passes():
    from ofm.selftest import run
    assert run() == 0


@pytest.fixture(scope="module")
def creek_world():
    from ofm.pipeline import generate
    w = generate(small_config(**{"hydrology.creek_min_discharge": 0.03, "world.seed": 5}), psych=False)
    assert w.creeks
    return w


def test_creeks_extend_the_drainage_network(creek_world):
    w = creek_world
    stop = w.river_mask | w.water
    for c in w.creeks:
        p = np.asarray(c["points"])
        i = np.clip((p[:, 0] / w.cell).astype(int), 0, w.size - 1)
        j = np.clip((p[:, 1] / w.cell).astype(int), 0, w.size - 1)
        # the trace runs on dry cells and ends in the water it drains into (the last
        # smoothed points lie on that final step; corner cutting may graze a river cell)
        assert stop[j[:-4], i[:-4]].mean() <= 0.1, c["id"]
        assert c["mouth"] in ("water", "join", "edge")
    m = w.evaluation["terrain"]["metrics"]
    total = (sum(r["length_m"] for r in w.rivers) + sum(c["length_m"] for c in w.creeks)) / 1000.0
    assert m["river_km"] == pytest.approx(total, rel=1e-6)


def test_creek_water_follows_the_ground(creek_world):
    from ofm.export.water import creek_meshes
    w = creek_world
    sp = w.hires_spacing
    meshes = creek_meshes(w)
    assert meshes
    buried = total = 0
    for creek, mesh, _ in meshes:
        P = mesh.positions
        lift = P[:, 2] - bilinear(w.hires, P[:, 0] / sp, P[:, 1] / sp)
        assert lift.min() >= 0.1 - 1e-6 and lift.max() <= 0.55, creek["id"]
        T = mesh.indices.astype(np.int64)
        q = (P[T[:, 0]] + P[T[:, 1]] + P[T[:, 2]]) / 3.0
        g = bilinear(w.hires, q[:, 0] / sp, q[:, 1] / sp)
        buried += int((g > q[:, 2]).sum())
        total += len(q)
    assert buried <= 0.005 * total


def test_no_trees_on_creeks(creek_world):
    from ofm.detail.scatter import creek_mask, on_creek
    w = creek_world
    m = creek_mask(w)
    assert m is not None and m.any()
    for kind, a in w.scatter.items():
        if kind != "reed" and len(a):
            assert not on_creek(m, a[:, 0], a[:, 1]).any(), kind


def test_creeks_follow_the_volume_where_it_replaces_the_terrain(creek_world):
    from ofm.export.water import creek_meshes
    from ofm.volume.builder import build_volume, volume_ground
    w = creek_world
    vol = build_volume(w)
    sp = w.hires_spacing
    n = vol.deviation.shape[0]
    for creek, mesh, _ in creek_meshes(w, vol):
        P = mesh.positions
        d = vol.deviation[np.clip((P[:, 1] / sp).astype(int), 0, n - 1), np.clip((P[:, 0] / sp).astype(int), 0, n - 1)]
        if d.any():
            z, ok = volume_ground(vol.field, P[d, 0], P[d, 1], P[d, 2])
            gap = P[d, 2] - z
            assert (gap[ok] <= 0.8).all() and (gap[ok] >= -0.3).all(), creek["id"]
