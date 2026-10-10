import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ofm.config import load_config  # noqa: E402
from ofm.pipeline import generate  # noqa: E402

SMALL = {
    "world.size": 64,
    "world.seed": 11,
    "erosion.droplets_per_cell": 0.5,
    "psych.grid": 24,
    "psych.agents": 3,
    "psych.agent_minutes": 6.0,
    "volume.voxel_size": 8.0,
    "volume.chunk_voxels": 16,
    "detail.scatter_max": 3000,
    "export.previews": False,
    "ai.provider": "none",
    "ai.iterations": 1,
}


def small_config(**extra):
    overrides = dict(SMALL)
    overrides.update(extra)
    return load_config(None, overrides)


@pytest.fixture(scope="session")
def small_world():
    return generate(small_config())
