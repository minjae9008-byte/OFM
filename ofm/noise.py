"""Vectorised gradient noise (Perlin style) in 2D and 3D.

All functions accept numpy arrays of any shape and return float64 arrays of
the same shape, so whole grids or voxel blocks are evaluated in one call.
"""
from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(seed: int, *tags: object) -> int:
    """Stable sub-seed for a named purpose, so stages never share RNG streams."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(seed)).encode())
    for tag in tags:
        h.update(b"/")
        h.update(str(tag).encode())
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class Noise:
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        p = rng.permutation(256).astype(np.int64)
        self._perm = np.concatenate([p, p])
        ang = rng.uniform(0.0, 2.0 * np.pi, 256)
        self._g2 = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
        g3 = rng.normal(size=(256, 3))
        self._g3 = g3 / np.linalg.norm(g3, axis=1, keepdims=True)
        self._offsets = rng.uniform(-4096.0, 4096.0, size=(32, 3))

    # -- base lattice noise -------------------------------------------------
    def perlin2(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx = x - x0
        fy = y - y0
        xi = x0.astype(np.int64) & 255
        yi = y0.astype(np.int64) & 255
        perm = self._perm
        px0 = perm[xi]
        px1 = perm[xi + 1]
        g = self._g2

        def corner(h, dx, dy):
            return g[h, 0] * dx + g[h, 1] * dy

        n00 = corner(perm[px0 + yi], fx, fy)
        n10 = corner(perm[px1 + yi], fx - 1.0, fy)
        n01 = corner(perm[px0 + yi + 1], fx, fy - 1.0)
        n11 = corner(perm[px1 + yi + 1], fx - 1.0, fy - 1.0)
        u = _fade(fx)
        v = _fade(fy)
        a = n00 + u * (n10 - n00)
        b = n01 + u * (n11 - n01)
        return (a + v * (b - a)) * 1.414

    def perlin3(self, x, y, z) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
        fx, fy, fz = x - x0, y - y0, z - z0
        xi = x0.astype(np.int64) & 255
        yi = y0.astype(np.int64) & 255
        zi = z0.astype(np.int64) & 255
        perm = self._perm
        g = self._g3
        u, v, w = _fade(fx), _fade(fy), _fade(fz)
        result = np.zeros(np.broadcast(x, y, z).shape, dtype=np.float64)
        for cx in (0, 1):
            wx = u if cx else 1.0 - u
            px = perm[xi + cx]
            for cy in (0, 1):
                wy = v if cy else 1.0 - v
                pxy = perm[px + yi + cy]
                for cz in (0, 1):
                    wz = w if cz else 1.0 - w
                    h = perm[pxy + zi + cz]
                    dot = g[h, 0] * (fx - cx) + g[h, 1] * (fy - cy) + g[h, 2] * (fz - cz)
                    result += wx * wy * wz * dot
        return result * 1.15

    # -- fractal combinations ----------------------------------------------
    def fbm2(self, x, y, octaves: int = 5, lacunarity: float = 2.0, gain: float = 0.5) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        total = np.zeros(np.broadcast(x, y).shape)
        amp, freq, norm = 1.0, 1.0, 0.0
        for o in range(octaves):
            ox, oy, _ = self._offsets[o % 32]
            total += amp * self.perlin2(x * freq + ox, y * freq + oy)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm

    def fbm3(self, x, y, z, octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        total = np.zeros(np.broadcast(x, y, z).shape)
        amp, freq, norm = 1.0, 1.0, 0.0
        for o in range(octaves):
            ox, oy, oz = self._offsets[o % 32]
            total += amp * self.perlin3(x * freq + ox, np.asarray(y) * freq + oy, np.asarray(z) * freq + oz)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm

    def ridged2(self, x, y, octaves: int = 6, lacunarity: float = 2.0, gain: float = 2.0,
                offset: float = 1.0) -> np.ndarray:
        """Musgrave ridged multifractal, returns values roughly in [0, 1]."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        total = np.zeros(np.broadcast(x, y).shape)
        weight = np.ones_like(total)
        freq, norm = 1.0, 0.0
        for o in range(octaves):
            ox, oy, _ = self._offsets[(o + 7) % 32]
            signal = offset - np.abs(self.perlin2(x * freq + ox, y * freq + oy))
            signal = signal * signal * weight
            weight = np.clip(signal * gain, 0.0, 1.0)
            amp = lacunarity ** (-1.3 * o)
            total += signal * amp
            norm += amp
            freq *= lacunarity
        return np.clip(total / norm, 0.0, 1.0)

    def warp2(self, x, y, strength: float, scale: float = 1.0, octaves: int = 4):
        """Domain warp: returns displaced coordinates."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        wx = self.fbm2(x * scale + 31.7, y * scale + 12.9, octaves)
        wy = self.fbm2(x * scale - 18.3, y * scale + 47.1, octaves)
        return x + strength * wx, y + strength * wy
