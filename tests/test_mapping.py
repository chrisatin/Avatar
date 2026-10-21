import numpy as np

from core.animation.mapping import (
    ARKIT_51,
    ARKIT_52,
    JAW_INDEX_51,
    arkit_to_vrm,
    flame_expression_from_arkit,
    gaze_from_weights,
    validate_lut_coverage,
    vector_to_weights,
    weights_to_vector,
)


def test_arkit_lists():
    assert len(ARKIT_52) == 52
    assert len(ARKIT_51) == 51
    assert "_neutral" in ARKIT_52
    assert "_neutral" not in ARKIT_51
    assert ARKIT_51[JAW_INDEX_51] == "jawOpen"


def test_roundtrip_vector():
    w = {"jawOpen": 0.5, "mouthSmileLeft": 0.3, "eyeBlinkRight": 0.9}
    vec = weights_to_vector(w)
    assert len(vec) == 51
    back = vector_to_weights(vec)
    assert abs(back["jawOpen"] - 0.5) < 1e-6
    assert abs(back["eyeBlinkRight"] - 0.9) < 1e-6


def test_vrm_lut_valid():
    assert validate_lut_coverage() == []
    vrm = arkit_to_vrm({"jawOpen": 0.8, "mouthSmileLeft": 1.0, "mouthSmileRight": 1.0})
    assert vrm["aa"] > 0.5
    assert vrm["happy"] > 0.5
    assert all(0.0 <= v <= 1.0 for v in vrm.values())


def test_gaze():
    x, y = gaze_from_weights({"eyeLookOutLeft": 1.0, "eyeLookDownRight": 1.0})
    assert x > 0
    assert y < 0


def test_flame_w_mapping():
    m = np.zeros((51, 103), dtype=np.float32)
    m[:, :51] = np.eye(51, dtype=np.float32)
    m[JAW_INDEX_51, 100:103] = 1.0
    w = {"jawOpen": 1.0}
    y = flame_expression_from_arkit(w, m)
    assert y.shape == (103,)
    assert abs(float(y[24]) - 1.0) < 1e-6
    assert abs(float(y[100]) - 100.0) < 1e-3
