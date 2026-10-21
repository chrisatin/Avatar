import numpy as np

from core.creation.frames import ScoredFrame, brightness, crop_face, select_keyframes, sharpness


def _fake_frame(i: int, size: int = 120) -> np.ndarray:
    rng = np.random.default_rng(i)
    img = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    img[40:80, 40:80] = 200
    return img


def test_sharpness_and_brightness():
    img = _fake_frame(1)
    assert sharpness(img) > 0
    assert 0 < brightness(img) < 255


def test_select_keyframes():
    frames = [_fake_frame(i) for i in range(30)]
    picked = select_keyframes(frames, [None] * 30, target=8, min_frames=4)
    assert 4 <= len(picked) <= 8
    idx = [f.index for f in picked]
    assert idx == sorted(idx)


def test_crop_face_center():
    img = _fake_frame(2, 200)
    crop = crop_face(img, None, margin=0.2)
    assert crop.shape[0] == crop.shape[1]
    assert crop.shape[0] <= 200


def test_scored_frame_fields():
    img = _fake_frame(3)
    f = ScoredFrame(0, img, sharpness(img), brightness(img), 0.0, 1.0)
    assert f.index == 0
    assert f.image.shape == img.shape
