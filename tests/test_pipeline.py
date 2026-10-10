import numpy as np

from ofm.grid import bilinear
from ofm.pipeline import generate
from tests.conftest import small_config


def test_generation_is_deterministic():
    a = generate(small_config(), psych=False)
    b = generate(small_config(), psych=False)
    assert np.array_equal(a.height, b.height)
    assert len(a.caves) == len(b.caves)
    assert [p.kind for p in a.pois] == [p.kind for p in b.pois]


def test_hydrology_is_physically_consistent(small_world):
    w = small_world
    cfg = w.cfg
    assert w.ocean.any() and (~w.ocean).any()
    assert not (w.ocean & w.lake_mask).any()
    # lakes sit below their level and the lake surface is flat
    for lake in w.lakes:
        assert lake["max_depth"] > 0
    if w.lake_mask.any():
        assert (w.height[w.lake_mask] <= w.lake_level[w.lake_mask] + 1e-6).all()
    # rivers never flow uphill along their traced course
    for r in w.rivers:
        z = np.array(r["points"])[:, 2]
        assert (np.diff(z) <= 1e-6).all(), r["id"]
    # temperature falls with elevation (lapse rate)
    land = ~w.ocean
    hi = w.height > np.quantile(w.height[land], 0.9)
    lo = land & (w.height < np.quantile(w.height[land], 0.2))
    assert w.temperature[hi].mean() < w.temperature[lo].mean()
    assert cfg.world.size == w.height.shape[0]


def test_caves_start_at_the_surface_and_stay_inside(small_world):
    w = small_world
    assert w.caves, "the default strata contain karst, caves expected"
    for c in w.caves:
        assert len(c.entrances) >= 1
        e = c.entrances[0]
        ground = float(w.height_at(e["x"], e["y"]))
        assert abs(e["z"] - ground) < 15.0
        segs = c.segments
        assert (segs[:, [0, 1, 3, 4]] >= 0).all() and (segs[:, [0, 1, 3, 4]] <= w.extent).all()
        # the trunk passage is underground for most of its length
        mid = segs[len(segs) // 4: max(len(segs) // 4 + 1, 3 * len(segs) // 4)]
        surf = bilinear(w.height, mid[:, 3] / w.cell - 0.5, mid[:, 4] / w.cell - 0.5)
        assert np.mean(surf - mid[:, 5] > 2.0) > 0.8


def test_volume_density_carves_caves_and_adds_features(small_world):
    from ofm.volume.density import DensityField
    w = small_world
    f = DensityField(w)
    cave = next(c for c in w.caves if c.main_count > 4)
    seg = cave.segments[cave.main_count // 2]
    centre = (seg[0:3] + seg[3:6]) / 2
    pts = centre[None, None, None, :] + np.zeros((1, 1, 1, 3))
    D, D0 = f.sample(pts[..., 0], pts[..., 1], pts[..., 2] + 0.5)
    assert D.item() < 0, "cave passage centre must be air"
    assert D0.item() > 0, "...while the heightfield alone would be rock there"
    for feat in w.features:
        p = np.array([[[[feat.x, feat.y, feat.z + feat.height * 0.3]]]])
        if feat.kind == "arch":
            continue
        D, D0 = f.sample(p[..., 0], p[..., 1], p[..., 2])
        assert D.item() > 0, f"{feat.kind} should be solid above the ground"


def test_psychology_report(small_world):
    sim = small_world.evaluation["simulation"]
    for k, v in sim["scores"].items():
        assert 0.0 <= v <= 10.0, k
    assert len(sim["regions"]) == 16
    assert small_world.psych["agents"]
    assert small_world.landmarks
    per = small_world.psych["perception"]
    assert per.visible.shape == (per.grid ** 2, per.grid ** 2)
    assert per.visible[np.arange(per.grid ** 2), np.arange(per.grid ** 2)].all()


def test_hybrid_volume_cuts_holes_only_where_needed(small_world):
    from ofm.volume.builder import build_volume
    vol = build_volume(small_world)
    assert vol.chunks
    assert 0 < vol.holes.mean() < 0.5
    full = build_volume(small_world, mode="full")
    assert full.stats["triangles"] > vol.stats["triangles"]
    assert full.holes.sum() == 0
