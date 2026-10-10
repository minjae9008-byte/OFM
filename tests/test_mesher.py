import numpy as np

from ofm.volume.mesher import merge_meshes, surface_nets


def _directed_edges(indices):
    t = indices.astype(np.int64)
    e = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    return e[:, 0] * 10**7 + e[:, 1], e[:, 1] * 10**7 + e[:, 0]


def _sphere(n=36, r=11.0):
    g = np.arange(n) - n / 2 + 0.25
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return r - np.sqrt(X ** 2 + Y ** 2 + Z ** 2)


def test_sphere_is_closed_and_outward():
    D = _sphere()
    m = surface_nets(D, np.zeros(3), 1.0)
    key, rkey = _directed_edges(m.indices)
    assert len(np.unique(key)) == len(key), "each directed edge appears once"
    assert np.isin(rkey, key).all(), "every edge has its opposite (closed surface)"
    centre = np.full(3, 18.0 - 0.25)
    out = np.sum(m.normals * (m.positions - centre), axis=1)
    assert (out > 0).mean() > 0.99


def test_chunks_join_without_gaps_or_duplicates():
    D = _sphere(40, 13.0)
    full = surface_nets(D, np.zeros(3), 1.0)
    # split along x into two chunks that sample one apron layer around their owned range
    parts = []
    for x0, x1 in ((0, 20), (20, 39)):
        lo = max(0, x0 - 1)
        hi = min(40, x1 + 2)
        sub = D[lo:hi]
        owned = ((x0 - lo, x1 - lo), (0, 40), (0, 40))
        parts.append(surface_nets(sub, np.array([lo, 0.0, 0.0]), 1.0, owned))
    merged = merge_meshes(parts)
    key = np.round(merged.positions * 1e4).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    welded = inv.ravel()[merged.indices.astype(np.int64)]
    k, rk = _directed_edges(welded)
    assert merged.triangle_count == full.triangle_count
    assert len(np.unique(k)) == len(k)
    assert np.isin(rk, k).all()


def test_render_mesh_handles_offscreen_geometry():
    from ofm.render3d import render_mesh
    m = surface_nets(_sphere(20, 6.0), np.zeros(3), 1.0)
    img = render_mesh(m, eye=(10, -40, 10), target=(10, 10, 10), width=64, height=48)
    assert img.size == (64, 48)
    # camera looking away from everything: nothing visible, background only
    img = render_mesh(m, eye=(10, -40, 10), target=(10, -80, 10), width=64, height=48)
    assert len(np.unique(np.asarray(img).reshape(-1, 3), axis=0)) == 1
