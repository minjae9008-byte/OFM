"""Coordinate conventions, centralised.

OFM world: metres, right-handed, x = east, y = north, z = up.

Unity:  metres, left-handed, Y up.   (X, Y, Z) = (x, z, y)
        The axis swap mirrors handedness, so triangle winding is reversed.
glTF:   metres, right-handed, Y up, +Z forward.   (X, Y, Z) = (x, z, -y)
        A proper rotation, winding unchanged.
Unreal: centimetres, left-handed, Z up.   (X, Y, Z) = (100x, -100y, 100z)
        This matches Unreal's glTF importer, which maps glTF (X, Y, Z) to
        (X, Z, Y) and scales metres to centimetres.
"""
from __future__ import annotations

import numpy as np


def to_unity(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], p[..., 1]], axis=-1)


def to_gltf(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def to_unreal(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0] * 100.0, -p[..., 1] * 100.0, p[..., 2] * 100.0], axis=-1)


def unreal_yaw_deg(yaw_ofm_deg: float) -> float:
    """OFM yaw is counter-clockwise from +x seen from above; Unreal yaw is clockwise."""
    return -float(yaw_ofm_deg)


def unity_yaw_deg(yaw_ofm_deg: float) -> float:
    """Unity yaw rotates clockwise around +Y seen from above, starting at +Z (north)."""
    return 90.0 - float(yaw_ofm_deg)
