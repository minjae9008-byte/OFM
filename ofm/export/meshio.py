"""Mesh writers: binary glTF (.glb) and the raw Unity format (.ofmmesh).

Both write vertex positions relative to a local origin so meshes keep float
precision and get sensible pivots; engines place the object at that origin.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from ..volume.mesher import Mesh
from .coords import to_gltf, to_unity

OFMM_MAGIC = b"OFMM"
OFMM_VERSION = 1


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(mesh: Mesh, path: Path, origin: np.ndarray, name: str = "mesh",
              uvs: np.ndarray | None = None, base_color=(1.0, 1.0, 1.0, 1.0),
              alpha_blend: bool = False) -> None:
    pos = to_gltf(mesh.positions - origin).astype(np.float32)
    nrm = to_gltf(mesh.normals.astype(np.float64)).astype(np.float32)
    idx = mesh.indices.astype(np.uint32).reshape(-1)
    views, accessors, blobs = [], [], []
    offset = 0

    def add(data: np.ndarray, comp: int, typ: str, count: int, target: int | None,
            minmax: bool = False, normalized: bool = False):
        nonlocal offset
        raw = _pad4(data.tobytes())
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data.tobytes())}
        if target:
            view["target"] = target
        views.append(view)
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": count, "type": typ}
        if normalized:
            acc["normalized"] = True
        if minmax:
            acc["min"] = [float(v) for v in data.reshape(count, -1).min(axis=0)]
            acc["max"] = [float(v) for v in data.reshape(count, -1).max(axis=0)]
        accessors.append(acc)
        blobs.append(raw)
        offset += len(raw)
        return len(accessors) - 1

    attrs = {"POSITION": add(pos, 5126, "VEC3", len(pos), 34962, minmax=True),
             "NORMAL": add(nrm, 5126, "VEC3", len(nrm), 34962)}
    if mesh.colors is not None:
        attrs["COLOR_0"] = add(np.ascontiguousarray(mesh.colors.astype(np.uint8)), 5121, "VEC4",
                               len(mesh.colors), 34962, normalized=True)
    if uvs is not None:
        attrs["TEXCOORD_0"] = add(uvs.astype(np.float32), 5126, "VEC2", len(uvs), 34962)
    ind = add(idx, 5125, "SCALAR", len(idx), 34963)
    material = {"name": f"{name}_mat", "pbrMetallicRoughness": {
        "baseColorFactor": list(base_color), "metallicFactor": 0.0, "roughnessFactor": 0.92}}
    if alpha_blend:
        material["alphaMode"] = "BLEND"
    gltf = {
        "asset": {"version": "2.0", "generator": "OFM"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": attrs, "indices": ind, "material": 0}]}],
        "materials": [material],
        "buffers": [{"byteLength": offset}],
        "bufferViews": views,
        "accessors": accessors,
    }
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    binary = b"".join(blobs)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<I4s", len(js), b"JSON"))
        fh.write(js)
        fh.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        fh.write(binary)


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF" and version == 2 and total == len(data)
    jlen, jtype = struct.unpack_from("<I4s", data, 12)
    js = json.loads(data[20:20 + jlen])
    blen, btype = struct.unpack_from("<I4s", data, 20 + jlen)
    return js, data[28 + jlen:28 + jlen + blen]


def write_ofmmesh(mesh: Mesh, path: Path, origin: np.ndarray, uvs: np.ndarray | None = None) -> None:
    """Unity-ready raw mesh: Unity axes, Unity winding, local to ``origin``.

    Layout (little endian):
      char[4] 'OFMM', uint32 version, uint32 vertexCount, uint32 indexCount,
      uint32 flags (1 = has colours, 2 = has uv), float32[3] origin (Unity space),
      float32[3*n] positions, float32[3*n] normals, [uint8[4*n] colours], [float32[2*n] uv],
      uint32[m] indices
    """
    pos = to_unity(mesh.positions - origin).astype("<f4")
    nrm = to_unity(mesh.normals.astype(np.float64)).astype("<f4")
    idx = mesh.indices.astype(np.uint32)[:, ::-1].reshape(-1).astype("<u4")
    flags = (1 if mesh.colors is not None else 0) | (2 if uvs is not None else 0)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sIIII", OFMM_MAGIC, OFMM_VERSION, len(pos), len(idx), flags))
        fh.write(to_unity(np.asarray(origin, dtype=np.float64)).astype("<f4").tobytes())
        fh.write(pos.tobytes())
        fh.write(nrm.tobytes())
        if mesh.colors is not None:
            fh.write(mesh.colors.astype(np.uint8).tobytes())
        if uvs is not None:
            fh.write(uvs.astype("<f4").tobytes())
        fh.write(idx.tobytes())


def read_ofmmesh(path: Path) -> dict:
    data = Path(path).read_bytes()
    magic, version, nv, ni, flags = struct.unpack_from("<4sIIII", data, 0)
    assert magic == OFMM_MAGIC and version == OFMM_VERSION
    off = 20
    origin = np.frombuffer(data, "<f4", 3, off)
    off += 12
    pos = np.frombuffer(data, "<f4", nv * 3, off).reshape(nv, 3)
    off += nv * 12
    nrm = np.frombuffer(data, "<f4", nv * 3, off).reshape(nv, 3)
    off += nv * 12
    col = None
    if flags & 1:
        col = np.frombuffer(data, np.uint8, nv * 4, off).reshape(nv, 4)
        off += nv * 4
    uv = None
    if flags & 2:
        uv = np.frombuffer(data, "<f4", nv * 2, off).reshape(nv, 2)
        off += nv * 8
    idx = np.frombuffer(data, "<u4", ni, off)
    return {"origin": origin, "positions": pos, "normals": nrm, "colors": col, "uv": uv, "indices": idx}
