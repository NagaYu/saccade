"""Shared, session-scoped fixtures so the suite stays fast and deterministic.

All tests run on the hermetic SyntheticBackbone (no model download, no webcam), so
the three claims (省エネ / 品質保持 / 予算保証) are verified reproducibly on CPU.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saccade import SyntheticBackbone, synthetic_walking_stream


@pytest.fixture(scope="session")
def backbone():
    return SyntheticBackbone(grid=(14, 14), img_size=224, embed_dim=384, seed=0)


@pytest.fixture(scope="session")
def stream():
    frames, labels = synthetic_walking_stream(img_size=224, seed=0)
    return frames, np.array(labels)


@pytest.fixture(scope="session")
def short_stream():
    # a compact stream for quick controller/engine invariant checks
    frames, labels = synthetic_walking_stream(
        img_size=224, seed=1,
        segments=[("static", 4), ("walk", 16), ("event", 8), ("turn", 10)])
    return frames, np.array(labels)


WARMUP = 6
