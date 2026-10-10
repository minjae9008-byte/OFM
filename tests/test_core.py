import numpy as np
import pytest

from ofm.config import ConfigError, get_by_path, load_config, set_by_path
from ofm.grid import blur, chamfer_distance, label, resample
from ofm.noise import Noise, derive_seed


def test_noise_is_deterministic_and_bounded():
    a = Noise(5).fbm2(np.linspace(0, 10, 500), np.linspace(3, 7, 500), 6)
    b = Noise(5).fbm2(np.linspace(0, 10, 500), np.linspace(3, 7, 500), 6)
    assert np.array_equal(a, b)
    assert np.abs(a).max() < 1.0
    r = Noise(5).ridged2(np.linspace(0, 4, 200), np.zeros(200))
    assert r.min() >= 0.0 and r.max() <= 1.0
    p3 = Noise(1).perlin3(np.random.rand(1000) * 9, np.random.rand(1000) * 9, np.random.rand(1000) * 9)
    assert np.abs(p3).max() < 1.5
    assert derive_seed(1, "a") != derive_seed(1, "b")


def test_label_counts_components_2d_and_3d():
    m = np.zeros((20, 20), dtype=bool)
    m[2:5, 2:5] = True
    m[10:12, 10:18] = True
    m[15, 15] = True
    lab, n = label(m)
    assert n == 3
    assert len(np.unique(lab[m])) == 3
    v = np.zeros((10, 10, 10), dtype=bool)
    v[1:3, 1:3, 1:3] = True
    v[6:9, 6:9, 6:9] = True
    _, n3 = label(v, diagonal=False)
    assert n3 == 2


def test_chamfer_distance_approximates_euclidean():
    m = np.zeros((41, 41), dtype=bool)
    m[20, 20] = True
    d = chamfer_distance(m)
    assert d[20, 30] == pytest.approx(10.0, abs=0.01)
    assert d[30, 30] == pytest.approx(np.hypot(10, 10), rel=0.05)


def test_blur_and_resample_preserve_constants():
    c = np.full((32, 32), 3.5)
    assert np.allclose(blur(c, 4.0), 3.5)
    assert np.allclose(resample(c, (65, 65), cubic=True), 3.5)


def test_config_overrides_and_validation(tmp_path):
    cfg = load_config(None, {"world.seed": "42", "caves.density": 2, "export.engines": "unity"})
    assert cfg.world.seed == 42 and cfg.caves.density == 2.0 and cfg.export.engines == ["unity"]
    set_by_path(cfg, "cliffs.undercut_depth", 3)
    assert get_by_path(cfg, "cliffs.undercut_depth") == 3.0
    with pytest.raises(ConfigError):
        load_config(None, {"world.size": 100})
    with pytest.raises(ConfigError):
        load_config(None, {"nope.key": 1})
    p = tmp_path / "c.toml"
    p.write_text('[world]\nsize = 128\n[[strata.layers]]\nname = "chalk"\nthickness = 20.0\nhardness = 0.4\n'
                 'solubility = 0.9\ncolor = [240, 240, 230]\n')
    cfg2 = load_config(p)
    assert cfg2.world.size == 128 and cfg2.strata.layers[0].name == "chalk"


def test_presets_load():
    from ofm.cli import PRESET_DIR, list_presets
    names = list_presets()
    assert {"default", "quick", "alpine", "archipelago", "canyon"} <= set(names)
    for n in names:
        load_config(PRESET_DIR / f"{n}.toml")
