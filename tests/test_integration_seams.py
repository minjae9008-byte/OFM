"""Seams between stages: everything placed in the world or exported must agree with the
exported ground (the detail heightfield) and with the volume where it replaces the terrain."""
import numpy as np
import pytest

from ofm.grid import bilinear, cells_to_cells, gradient


def test_objects_sit_on_the_exported_ground(small_world):
    w = small_world
    for c in w.caves:
        for e in c.entrances:
            if e["kind"] in ("sinkhole", "spring", "grotto"):
                assert abs(e["z"] - float(w.ground_at(e["x"], e["y"]))) < 1.0, (c.id, e["kind"])
    for f in w.features:
        if f.kind != "sea_stack":
            assert abs(f.z - float(w.ground_at(f.x, f.y))) < 1e-6, f.id
    for p in w.pois:
        if p.kind != "cave_entrance":
            assert abs(p.z - float(w.ground_at(p.x, p.y))) < 1e-6, p.kind
    sp = w.hires_spacing
    gx, gy = gradient(w.hires, sp)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    for kind, a in w.scatter.items():
        if not len(a):
            continue
        assert np.allclose(a[:, 2], w.ground_at(a[:, 0], a[:, 1]))
        assert (a[:, 2] >= w.cfg.world.sea_level).all(), kind
        limit = 52.0 if kind == "boulder" else 44.0
        assert (bilinear(slope, a[:, 0] / sp, a[:, 1] / sp) <= limit + 1e-6).all(), kind


def test_caves_keep_their_cover_under_the_exported_ground(small_world):
    w = small_world
    for c in w.caves:
        seg = np.asarray(c.segments)
        mid = 0.5 * (seg[:, 0:3] + seg[:, 3:6])
        roof = mid[:, 2] + np.maximum(seg[:, 6], seg[:, 7])
        ent = np.array([[e["x"], e["y"]] for e in c.entrances])
        far = np.min(np.hypot(mid[:, None, 0] - ent[None, :, 0], mid[:, None, 1] - ent[None, :, 1]), axis=1) > 60.0
        if far.any():
            assert (w.ground_at(mid[far, 0], mid[far, 1]) - roof[far]).min() > 0.0, c.id


def test_water_meshes_fit_the_terrain(small_world):
    from ofm.export.water import lake_meshes, river_meshes
    w = small_world
    sp = w.hires_spacing
    for lake, mesh, _ in lake_meshes(w):
        v = mesh.positions
        g = bilinear(w.hires, v[:, 0] / sp, v[:, 1] / sp)
        bed = bilinear(w.lake_mask.astype(float), v[:, 0] / w.cell - 0.5, v[:, 1] / w.cell - 0.5) >= 0.5
        assert (g[~bed] >= lake["level"] - 2.0).all(), "lake sheet hanging over lower ground"
    for river, mesh, _ in river_meshes(w):
        v = mesh.positions
        g = bilinear(w.hires, v[:, 0] / sp, v[:, 1] / sp)
        assert (v[:, 2] - g).max() <= 0.1 + 1e-6, f"{river['id']} ribbon edge floats"


def test_objects_in_terrain_holes_stand_on_the_volume(small_world):
    from ofm.volume.builder import build_volume, seat_objects
    w = small_world
    vol = build_volume(w)
    before = {k: v.copy() for k, v in w.scatter.items()}
    seated, poi_z = seat_objects(w, vol)
    for k in before:                                   # the world itself is left unchanged
        assert np.array_equal(w.scatter[k], before[k])
    sp = w.hires_spacing
    n = vol.deviation.shape[0]
    checked = 0
    for kind, a in seated.items():
        if not len(a):
            continue
        m = vol.deviation[np.clip((a[:, 1] / sp).astype(int), 0, n - 1), np.clip((a[:, 0] / sp).astype(int), 0, n - 1)]
        if not m.any():
            continue
        X, Y = a[m, 0].reshape(-1, 1, 1), a[m, 1].reshape(-1, 1, 1)
        below, _ = vol.field.sample(X, Y, a[m, 2].reshape(-1, 1, 1) - 0.3)
        above, _ = vol.field.sample(X, Y, a[m, 2].reshape(-1, 1, 1) + 0.3)
        good = (below.ravel() > 0) & (above.ravel() <= 0)
        assert good.mean() > 0.97, kind
        checked += int(m.sum())
    assert vol.deviation.any()


def test_engine_maps_are_vertex_aligned(small_world):
    from ofm.export.exporter import _cells_at_vertices
    w = small_world
    xs = (np.arange(w.size) + 0.5) * w.cell                  # a field equal to the cell-centre x
    field = np.tile(xs, (w.size, 1))
    res = 2 * w.size
    out = _cells_at_vertices(field, res, w)
    want = np.clip(np.arange(res) * w.extent / (res - 1), xs[0], xs[-1])
    assert np.allclose(out[0], want)


def test_cells_to_cells_is_centre_aligned():
    G, n, E = 8, 32, 512.0
    g = np.tile((np.arange(G) + 0.5) * E / G, (G, 1))          # x of each coarse cell centre
    fine = cells_to_cells(g, (n, n))
    want = np.clip((np.arange(n) + 0.5) * E / n, g[0, 0], g[0, -1])
    assert np.allclose(fine[0], want)


def test_no_cliff_weathering_under_water(small_world):
    from ofm.volume.density import DensityField
    w = small_world
    f = DensityField(w)
    c = np.arange(w.hires.shape[0]) * w.hires_spacing / w.cell - 0.5
    wet = bilinear(w.water.astype(float), c[None, :], c[:, None]) > 0.7
    assert f.cliff_w[wet].max(initial=0.0) == 0.0
