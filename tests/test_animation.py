import numpy as np

from core.animation.smoothing import OneEuroFilter, QuaternionSmoother, VectorSmoother
from core.animation.retarget import Retargeter
from core.animation.tracker import SyntheticTracker, create_tracker


def test_one_euro_converges():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    out = None
    for i in range(200):
        out = f(1.0, t=i * 0.033)
    assert abs(out - 1.0) < 0.05


def test_vector_smoother_dim():
    s = VectorSmoother(51)
    out = s([0.5] * 51, t=0.0)
    assert len(out) == 51


def test_quat_smoother_norm():
    q = QuaternionSmoother()
    out = None
    for _ in range(10):
        out = q([0.7071, 0.0, 0.7071, 0.0])
    norm = sum(c * c for c in out) ** 0.5
    assert abs(norm - 1.0) < 1e-5


def test_synthetic_tracker():
    t = create_tracker("synthetic")
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    r = t.process(img)
    assert r.present
    assert "jawOpen" in r.blendshapes
    assert len(r.landmarks) == 478


def test_retarget_output():
    tracker = SyntheticTracker()
    retarget = Retargeter(smooth=False)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = retarget.process(tracker.process(img))
    assert out["present"] is True
    assert len(out["weights"]) > 0
    assert 0.0 <= out["vrm"]["aa"] <= 1.0
    assert len(out["rotation"]) == 4
