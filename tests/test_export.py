import json

import numpy as np
import pytest

from ofm.export.exporter import export_world
from ofm.export.meshio import read_glb, read_ofmmesh
from ofm.volume.builder import build_volume


@pytest.fixture(scope="module")
def exported(small_world, tmp_path_factory):
    out = tmp_path_factory.mktemp("world")
    vol = build_volume(small_world)
    path = export_world(small_world, vol, out)
    return small_world, vol, out, json.loads(path.read_text(encoding="utf-8"))


def test_manifest_has_engine_fields(exported):
    w, vol, out, m = exported
    assert m["format"] == "ofm-world" and m["version"] == 1
    t = m["terrain"]
    res = t["heightmap_resolution"]
    assert res == w.hires.shape[0] == w.size * w.cfg.detail.upscale + 1
    assert (out / t["heightmap_raw16"]).stat().st_size == res * res * 2
    assert (out / t["holes_raw8"]).stat().st_size == (res - 1) ** 2
    assert (out / t["splat_raw"]).stat().st_size == t["splat_resolution"] ** 2 * len(t["layers"])
    for e in m["meshes"]:
        assert (out / e["unity"]).exists() and (out / e["glb"]).exists()
        assert len(e["origin_unity"]) == 3
    for p in m["pois"]:
        x, y, z = p["x"], p["y"], p["z"]
        assert p["position_unity"] == pytest.approx([x, z, y])
        assert p["position_unreal"] == pytest.approx([x * 100, -y * 100, z * 100])
    assert m["unreal_landscape"]["resolution"] in (127, 253, 505, 1009, 2017, 4033, 8129)


def test_heightmap_roundtrip(exported):
    w, vol, out, m = exported
    t = m["terrain"]
    res = t["heightmap_resolution"]
    raw = np.fromfile(out / t["heightmap_raw16"], dtype="<u2").reshape(res, res)
    h = t["height_min"] + raw / 65535.0 * (t["height_max"] - t["height_min"])
    assert np.abs(h - w.hires).max() < (t["height_max"] - t["height_min"]) / 65535.0 * 2


def test_mesh_files_roundtrip(exported):
    w, vol, out, m = exported
    by_name = {c.name: c for c in vol.chunks}
    e = max(m["meshes"], key=lambda e: e["triangles"])
    chunk = by_name[e["name"]]
    um = read_ofmmesh(out / e["unity"])
    origin = chunk.origin
    # Unity: (x, z, y) relative to origin, winding reversed
    expect = chunk.mesh.positions - origin
    assert np.allclose(um["positions"], expect[:, [0, 2, 1]], atol=1e-3)
    assert np.array_equal(um["indices"].reshape(-1, 3), chunk.mesh.indices[:, ::-1])
    js, binary = read_glb(out / e["glb"])
    acc = js["accessors"][js["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    view = js["bufferViews"][acc["bufferView"]]
    pos = np.frombuffer(binary, "<f4", acc["count"] * 3, view["byteOffset"]).reshape(-1, 3)
    assert np.allclose(pos, np.stack([expect[:, 0], expect[:, 2], -expect[:, 1]], 1), atol=1e-3)
    assert acc["min"] == pytest.approx(pos.min(axis=0).tolist(), abs=1e-4)


def test_report_and_scatter(exported):
    w, vol, out, m = exported
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# OFM")
    for s in m["scatter"]:
        arr = np.fromfile(out / s["file"], dtype="<f4").reshape(-1, 5)
        assert len(arr) == s["count"]
        assert (arr[:, 0] >= 0).all() and (arr[:, 0] <= w.extent).all()
