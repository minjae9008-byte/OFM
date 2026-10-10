"""External access to running Unity / Unreal editors.

Unity:  talks to OFMBridgeServer.cs (TCP, loopback, newline-delimited JSON).
Unreal: uses the Remote Control API plugin (HTTP, default port 30010) to call
        PythonScriptLibrary.ExecutePythonCommand, which runs the bundled
        ofm_unreal_import.py inside the editor.
Both can also be run headless through the engines' command lines.
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
UNREAL_SCRIPT = ENGINE_DIR / "unreal" / "ofm_unreal_import.py"
UNITY_ASSETS = ENGINE_DIR / "unity"


class BridgeError(RuntimeError):
    pass


class UnityBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 900.0):
        self.host, self.port, self.timeout = host, port, timeout

    def send(self, payload: dict) -> dict:
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
        except OSError as exc:
            raise BridgeError(f"Unity bridge not reachable on {self.host}:{self.port} ({exc}). "
                              "Open the project with the OFM editor scripts installed "
                              "(Tools > OFM > Bridge > Start).") from exc
        if not buf:
            raise BridgeError("Unity bridge closed the connection without a reply")
        resp = json.loads(buf.decode("utf-8"))
        if not resp.get("ok"):
            raise BridgeError(resp.get("message", "Unity reported an error"))
        return resp

    def ping(self) -> dict:
        return self.send({"cmd": "ping"})

    def import_world(self, manifest: str | Path, **options) -> dict:
        payload = {"cmd": "import", "manifest": str(Path(manifest).resolve())}
        payload.update(options)
        return self.send(payload)

    def clear(self) -> dict:
        return self.send({"cmd": "clear"})

    def save(self) -> dict:
        return self.send({"cmd": "save"})


class UnrealBridge:
    LIBRARY = "/Script/PythonScriptPlugin.Default__PythonScriptLibrary"

    def __init__(self, host: str = "127.0.0.1", port: int = 30010, timeout: float = 900.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _http(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                text = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise BridgeError(f"Unreal Remote Control error {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise BridgeError(f"Unreal Remote Control not reachable at {self.base} ({exc}). Enable the "
                              "'Remote Control API' and 'Python Editor Script Plugin' plugins.") from exc
        return json.loads(text) if text.strip() else {}

    def ping(self) -> dict:
        return self._http("GET", "/remote/info")

    def exec_python(self, code: str) -> dict:
        return self._http("PUT", "/remote/object/call", {
            "objectPath": self.LIBRARY,
            "functionName": "ExecutePythonCommand",
            "parameters": {"PythonCommand": code},
            "generateTransaction": True,
        })

    def import_world(self, manifest: str | Path, scatter: bool = False, scatter_limit: int = 1500) -> dict:
        script_dir = UNREAL_SCRIPT.parent.as_posix()
        code = (
            "import sys, importlib\n"
            f"sys.path.insert(0, r'{script_dir}') if r'{script_dir}' not in sys.path else None\n"
            "import ofm_unreal_import as _ofm\n"
            "importlib.reload(_ofm)\n"
            f"_ofm.import_world(r'{Path(manifest).resolve().as_posix()}', scatter={bool(scatter)}, "
            f"scatter_limit={int(scatter_limit)})\n"
        )
        return self.exec_python(code)


def unity_batch_command(unity_exe: str, project: str, manifest: str | Path) -> list[str]:
    return [unity_exe, "-batchmode", "-nographics", "-projectPath", str(project),
            "-executeMethod", "OFM.EditorTools.OFMWorldImporter.ImportFromCommandLine",
            "-ofmManifest", str(Path(manifest).resolve()), "-quit", "-logFile", "-"]


def unreal_batch_command(editor_cmd: str, uproject: str, manifest: str | Path, scatter: bool = False) -> list[str]:
    script = f"{UNREAL_SCRIPT.as_posix()} --manifest {Path(manifest).resolve().as_posix()} --save"
    if scatter:
        script += " --scatter"
    return [editor_cmd, str(uproject), "-run=pythonscript", f"-script={script}", "-unattended", "-nosplash"]


def run_batch(cmd: list[str]) -> int:
    return subprocess.call(cmd)


def install_unity(project: str | Path) -> Path:
    project = Path(project)
    if not (project / "Assets").is_dir():
        raise BridgeError(f"{project} does not look like a Unity project (no Assets folder)")
    dest = project / "Assets" / "OFM"
    for sub in ("Editor", "Runtime", "Shaders"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
        for f in (UNITY_ASSETS / sub).iterdir():
            if f.is_file():
                shutil.copy2(f, dest / sub / f.name)
    return dest


def install_unreal(project: str | Path) -> Path:
    project = Path(project)
    if not list(project.glob("*.uproject")):
        raise BridgeError(f"{project} does not contain a .uproject file")
    dest = project / "Content" / "Python"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(UNREAL_SCRIPT, dest / UNREAL_SCRIPT.name)
    return dest / UNREAL_SCRIPT.name
