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


def _boundary_edges(tris):
    e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    return u[c == 1]


@pytest.fixture(scope="module")
def lake_world():
    """Small world with a lake and rivers (the default small world has neither)."""
    from ofm.pipeline import generate_base, rebuild
    from .conftest import small_config
    w = rebuild(generate_base(small_config(**{"hydrology.river_min_discharge": 0.1, "world.seed": 5})),
                psych=False)
    assert w.lakes and w.rivers
    return w


def test_water_meshes_fit_the_terrain(lake_world):
    from ofm.detail.lakes import SHORE_TOLERANCE
    from ofm.export.water import lake_meshes, river_meshes
    w = lake_world
    sp = w.hires_spacing
    heads = np.array([r["points"][0][:2] for r in w.rivers])
    for lake, mesh, _ in lake_meshes(w):
        v = mesh.positions
        a, b = (v[e] for e in _boundary_edges(mesh.indices.astype(np.int64)).T)
        t = np.linspace(0.0, 1.0, 5)[None, :, None]
        p = a[:, None, :2] + (b - a)[:, None, :2] * t
        gap = lake["level"] - bilinear(w.hires, p[..., 0] / sp, p[..., 1] / sp).min(axis=1)
        hang = gap > SHORE_TOLERANCE
        # the sheet edge tucks under the shore everywhere but at the outflow lip
        mid = 0.5 * (a[hang, :2] + b[hang, :2])
        lip = np.hypot(mid[:, None, 0] - heads[None, :, 0], mid[:, None, 1] - heads[None, :, 1]).min(axis=1)
        assert (lip < 3.0 * w.cell).all() and gap.max() < 2.0, lake["id"]
    for river, mesh, _ in river_meshes(w):
        v = mesh.positions
        g = bilinear(w.hires, v[:, 0] / sp, v[:, 1] / sp)
        assert (v[:, 2] - g).max() <= 0.1 + 1e-6, f"{river['id']} ribbon edge floats"


def test_nothing_is_placed_under_a_lake_sheet(lake_world):
    from ofm.detail.lakes import lake_surface, under_lake
    w = lake_world
    surface = lake_surface(w)
    assert np.isfinite(surface).any()
    for kind, a in w.scatter.items():
        if len(a):
            assert not under_lake(w, a[:, 0], a[:, 1], a[:, 2], surface).any(), kind
    for p in w.pois:
        assert not under_lake(w, p.x, p.y, p.z, surface), p.kind


def test_lake_sheet_grows_up_the_basin_wall_and_stops_at_the_outlet():
    from ofm.detail.lakes import SHORE_TOLERANCE, sheet_quads
    n, level = 24, 100.0
    x = np.arange(n + 1, dtype=float)
    # bed, a wall rising past the level between x = 8 and 12, and an outlet notch at
    # row 12 that keeps falling beyond the wall
    h = np.tile(np.interp(x, [0, 8, 12, 24], [90.0, 90.0, 104.0, 108.0]), (n + 1, 1))
    h[11:14, 8:] = np.interp(x[8:], [8, 11, 24], [96.0, 98.5, 90.0])
    wet = np.zeros((n, n), dtype=bool)
    wet[:, :8] = True
    quads = sheet_quads(h, wet, np.ones_like(wet), level)
    assert quads[:, :8].all()
    # grown over the wall up to the shore (where the ground rises past the level) ...
    assert quads[0, 8:10].all() and not quads[0, 12:].any()
    # ... but not down the outlet beyond its lip
    assert not quads[12, 11:].any()
    # every edge of the sheet ends on ground at the level, except at the lip of the outlet
    q = np.pad(quads, 1, constant_values=True)           # (the grid border cuts the lake)
    ends = []
    for j, i in zip(*np.nonzero(quads)):
        if not q[j, i + 1]:
            ends += [(j, i), (j, i + 1)]
        if not q[j + 2, i + 1]:
            ends += [(j + 1, i), (j + 1, i + 1)]
        if not q[j + 1, i]:
            ends += [(j, i), (j + 1, i)]
        if not q[j + 1, i + 2]:
            ends += [(j, i + 1), (j + 1, i + 1)]
    ends = np.array(ends)
    hang = h[ends[:, 0], ends[:, 1]] < level - SHORE_TOLERANCE
    assert hang.any() and np.isin(ends[hang, 0], [11, 12, 13]).all()


def test_lakes_cover_their_basin_and_keep_their_shores():
    """A bowl whose deep centre is ringed by a shelf shallower than ``min_lake_depth``,
    draining over a single sill: the shelf is lake (no river in it), and carving the
    outflow channel does not cut the shore below the lake."""
    from ofm.grid import NEIGHBORS8, shifted
    from ofm.physics.hydrology import simulate_hydrology
    from .conftest import small_config
    cfg = small_config(**{"world.size": 32, "hydrology.river_min_discharge": 0.1})
    n = 32
    jj, ii = np.mgrid[0:n, 0:n].astype(float)
    h = np.where(ii < 8, -5.0 + 8.0 * ii, 60.0)
    r = np.hypot(ii - 19, jj - 16)
    h = h - np.where(r < 4, 9.0 * (1 - r / 4) ** 2 + 0.3, np.where(r < 7, 0.3, 0.0))
    h[16, 12] = 59.9                                   # the sill
    h[16, 8:12] = np.linspace(55.0, 59.0, 4)           # gully down to the coast
    hy = simulate_hydrology(cfg, h, np.full((n, n), 5.0), np.full((n, n), 2500.0))
    lake = hy["lake_mask"]
    assert len(hy["lakes"]) == 1 and not hy["lakes"][0]["closed_basin"]
    assert np.array_equal(lake, r < 7)
    assert not (hy["river_mask"] & lake).any()
    level = np.where(lake, hy["lake_level"], -np.inf)
    rim = np.full(h.shape, -np.inf)
    for dy, dx in NEIGHBORS8:
        rim = np.maximum(rim, shifted(level, dy, dx, -np.inf))
    shore = np.isfinite(rim) & ~lake & ~hy["ocean"]
    assert (hy["height"][shore] >= rim[shore] - 1e-9).all()
    out = [rv for rv in hy["rivers"] if np.hypot(*(np.array(rv["points"][0][:2]) - [200.0, 264.0])) < 1.0]
    assert out, "the outflow starts at the sill"
    assert out[0]["points"][0][2] <= hy["lakes"][0]["level"] + 1e-9, "the outflow leaves at the lake surface"
    for rv in hy["rivers"]:
        assert (np.diff(np.array(rv["points"])[:, 2]) <= 1e-9).all(), rv["id"]


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


def test_faces_below_the_water_surface_are_not_weathered(lake_world):
    from ofm.detail.lakes import lake_surface
    from ofm.volume.density import DensityField
    w = lake_world
    f = DensityField(w)
    f.cliff_w = np.ones_like(f.cliff_w)               # weather every face, wet or dry
    f.caves, f.features, f._cave_boxes, f._feat_boxes = [], [], [], []
    sp = w.hires_spacing
    surface = lake_surface(w)
    qj, qi = np.nonzero(np.isfinite(surface))
    x, y = (qi + 0.5) * sp, (qj + 0.5) * sp
    z = np.minimum(surface[qj, qi], w.ground_at(x, y)) - 2.0
    D, D0 = f.sample(x.reshape(-1, 1, 1), y.reshape(-1, 1, 1), z.reshape(-1, 1, 1))
    assert np.allclose(D, D0)
    # while a dry face is weathered
    j, i = np.unravel_index(np.argmax(w.hires), w.hires.shape)
    X, Y = np.full((1, 1, 1), i * sp), np.full((1, 1, 1), j * sp)
    D, D0 = f.sample(X, Y, np.full((1, 1, 1), w.hires[j, i] - 2.0))
    assert not np.allclose(D, D0)


def test_no_cliff_weathering_under_water(lake_world):
    from ofm.volume.density import DensityField
    w = lake_world
    f = DensityField(w)
    c = np.arange(w.hires.shape[0]) * w.hires_spacing / w.cell - 0.5
    wet = bilinear((w.water | w.river_mask).astype(float), c[None, :], c[:, None]) > 0.7
    assert wet.any() and f.cliff_w[wet].max(initial=0.0) == 0.0


def test_landforms_reach_below_the_ground_under_their_footprint(small_world, lake_world):
    assert any(f.kind == "arch" for f in small_world.features)
    for w in (small_world, lake_world):
        for f in w.features:
            x, y = f.footprint()
            assert f.z - f.foot <= float(w.ground_at(x, y).min()) - 2.0 + 1e-6, f.id
            assert f.stands_dry(w), f"{f.id} walls off water"


def test_arch_legs_stand_on_a_slope():
    from types import SimpleNamespace
    from ofm.volume.features import Feature
    slope = 0.6                                        # ground falls 0.6 m per m along x
    world = SimpleNamespace(ground_at=lambda x, y: 100.0 - slope * (np.asarray(x) - 500.0))
    f = Feature(id="a", kind="arch", x=500.0, y=500.0, z=100.0, height=40.0, radius=8.0, angle=0.0)
    f.foot = f.footing(world)
    major = f.height * 0.62
    for side in (-1.0, 1.0):                           # rock all the way down to the ground
        lx = f.x + side * major
        g = float(world.ground_at(lx, f.y))
        zs = np.linspace(g - 1.0, f.z, 40)
        assert (f.sdf(np.full_like(zs, lx), np.full_like(zs, f.y), zs) < 0).all()
    # under the span the opening stays open down to the ground
    assert f.sdf(np.array([f.x]), np.array([f.y]), np.array([f.z + 2.0]))[0] > 0


def test_pois_stand_on_the_ground_or_on_the_water_they_mark(small_world, lake_world):
    from ofm.detail.lakes import under_lake
    from ofm.detail.pois import river_surface
    from ofm.volume.features import rock_at
    for w in (small_world, lake_world):
        kinds = set()
        for p in w.pois:
            kinds.add(p.kind)
            if p.kind == "cave_entrance":
                continue
            assert not rock_at(w, p.x, p.y), f"{p.kind} inside a rock landform"
            assert not under_lake(w, p.x, p.y, p.z), p.kind
            ws = river_surface(w, p.x, p.y)
            ground = float(w.ground_at(p.x, p.y))
            assert p.z >= (ground if ws is None else max(ground, ws)) - 1e-6, p.kind
            assert p.z >= w.cfg.world.sea_level - 0.05, p.kind
        assert "rock_formation" in kinds or not w.features


def test_agents_do_not_walk_through_rock_landforms(small_world):
    from ofm.psych.agents import build_nav
    from ofm.psych.perception import compute_perception
    from ofm.volume.features import rock_at
    w = small_world
    per = compute_perception(w)
    nav = build_nav(w, per)
    G = per.grid
    rock = rock_at(w, per.ox, per.oy).reshape(G, G)
    assert rock.any(), "the small world's butte covers some walking cells"
    assert nav.blocked[rock].all() and not per.land[rock].any()
    # observers stand on the walked ground, not on top of the rock beside them
    ground = bilinear(w.ground(), per.ox / w.cell - 0.5, per.oy / w.cell - 0.5)
    assert np.allclose(per.oz - ground, w.cfg.psych.eye_height)
