import importlib
import json
import socket
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from ofm.engines import bridge
from ofm.export.exporter import export_world
from ofm.volume.builder import build_volume


@pytest.fixture(scope="module")
def manifest(small_world, tmp_path_factory):
    out = tmp_path_factory.mktemp("engines")
    return export_world(small_world, build_volume(small_world), out)


def test_unity_bridge_protocol(manifest):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    seen = {}

    def serve():
        conn, _ = srv.accept()
        data = b""
        while not data.endswith(b"\n"):
            data += conn.recv(4096)
        seen["req"] = json.loads(data)
        conn.sendall(b'{"ok":true,"message":"OFM quick (11)","unity":"2022.3.10f1"}\n')
        conn.close()

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    resp = bridge.UnityBridge(port=port, timeout=5).import_world(manifest, scatter=False)
    th.join(2)
    srv.close()
    assert resp["ok"] and seen["req"]["cmd"] == "import"
    assert seen["req"]["manifest"] == str(Path(manifest).resolve()) and seen["req"]["scatter"] is False


def test_unity_bridge_reports_unreachable():
    with pytest.raises(bridge.BridgeError):
        bridge.UnityBridge(port=9, timeout=1).ping()


def test_unreal_bridge_calls_python_library(manifest):
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_PUT(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

    srv = HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    bridge.UnrealBridge(port=srv.server_address[1], timeout=5).import_world(manifest)
    srv.shutdown()
    path, body = calls[0]
    assert path == "/remote/object/call"
    assert body["objectPath"].endswith("PythonScriptLibrary")
    assert body["functionName"] == "ExecutePythonCommand"
    code = body["parameters"]["PythonCommand"]
    compile(code, "<remote>", "exec")
    assert "ofm_unreal_import" in code and Path(manifest).resolve().as_posix() in code


def _fake_unreal():
    """Just enough of the `unreal` module to run the importer outside the editor."""
    u = types.ModuleType("unreal")
    log = {"spawned": [], "imports": [], "messages": []}

    class Vector:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z

    class Rotator(Vector):
        pass

    class Name(str):
        pass

    class StaticMesh:
        def __init__(self, path):
            self.path = path

    class Actor:
        def __init__(self, what, loc):
            self.what, self.loc, self.tags = what, loc, []

        def set_actor_label(self, label):
            self.label = label

        def set_folder_path(self, p):
            self.folder = p

        def set_actor_scale3d(self, s):
            self.scale = s

    class ActorSub:
        def spawn_actor_from_object(self, obj, loc, rot):
            a = Actor(obj, loc)
            log["spawned"].append(a)
            return a

        def spawn_actor_from_class(self, cls, loc, rot):
            a = Actor(cls, loc)
            log["spawned"].append(a)
            return a

    class AssetImportTask:
        def __init__(self):
            self.imported_object_paths = []

    class Tools:
        def import_asset_tasks(self, tasks):
            for t in tasks:
                log["imports"].append(t.filename)
                t.imported_object_paths = [t.destination_path + "/" + Path(t.filename).stem]

    class Slow:
        def __init__(self, n, msg):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def make_dialog(self, b):
            pass

        def enter_progress_frame(self, n, msg):
            pass

    u.Vector, u.Rotator, u.Name, u.StaticMesh = Vector, Rotator, Name, StaticMesh
    u.AssetImportTask = AssetImportTask
    u.AssetToolsHelpers = types.SimpleNamespace(get_asset_tools=lambda: Tools())
    u.load_asset = lambda p: StaticMesh(p)
    u.EditorActorSubsystem = object
    u.get_editor_subsystem = lambda cls: ActorSub()
    u.TargetPoint = "TargetPoint"
    u.ScopedSlowTask = Slow
    u.EditorAssetLibrary = types.SimpleNamespace(list_assets=lambda *a, **k: [])
    u.log = lambda m: log["messages"].append(m)
    return u, log


def test_unreal_importer_with_fake_editor(manifest, monkeypatch):
    fake, log = _fake_unreal()
    monkeypatch.setitem(sys.modules, "unreal", fake)
    sys.path.insert(0, str(bridge.UNREAL_SCRIPT.parent))
    try:
        mod = importlib.import_module("ofm_unreal_import")
        mod = importlib.reload(mod)
        summary = mod.import_world(str(manifest), scatter=True, scatter_limit=5)
    finally:
        sys.path.remove(str(bridge.UNREAL_SCRIPT.parent))
    m = json.loads(Path(manifest).read_text(encoding="utf-8"))
    assert summary["meshes"] == len(m["meshes"])
    assert summary["markers"] >= len(m["pois"])
    assert summary["scatter"] > 0
    poi = m["pois"][0]
    tp = next(a for a in log["spawned"] if getattr(a, "label", "") == (poi["name"] or poi["id"]))
    assert (tp.loc.x, tp.loc.y, tp.loc.z) == pytest.approx(tuple(poi["position_unreal"]))
    assert any("Landscape" in msg for msg in log["messages"])


def test_install_helpers(tmp_path):
    unity = tmp_path / "UnityProj"
    (unity / "Assets").mkdir(parents=True)
    dest = bridge.install_unity(unity)
    assert (dest / "Editor" / "OFMWorldImporter.cs").exists()
    assert (dest / "Runtime" / "OFMMarker.cs").exists()
    ue = tmp_path / "UEProj"
    ue.mkdir()
    (ue / "Game.uproject").write_text("{}")
    assert bridge.install_unreal(ue).exists()
    with pytest.raises(bridge.BridgeError):
        bridge.install_unity(tmp_path / "missing")


def test_batch_commands(manifest):
    cmd = bridge.unity_batch_command("Unity", "/p", manifest)
    assert "-executeMethod" in cmd and "OFM.EditorTools.OFMWorldImporter.ImportFromCommandLine" in cmd
    cmd = bridge.unreal_batch_command("UnrealEditor-Cmd", "/p/G.uproject", manifest)
    assert "-run=pythonscript" in cmd
