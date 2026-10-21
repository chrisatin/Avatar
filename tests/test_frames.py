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


def test_crop_face_with_offset():
    from core.creation.frames import crop_face_with_offset

    img = _fake_frame(2, 200)
    crop, ox, oy = crop_face_with_offset(img, None, margin=0.2)
    assert crop.shape[0] == crop.shape[1]
    assert ox >= 0 and oy >= 0
    center = np.array([[ox + crop.shape[1] / 2, oy + crop.shape[0] / 2]], dtype=np.float32)
    full_center = img[center[0, 1].astype(int), center[0, 0].astype(int)]
    assert np.array_equal(full_center, crop[crop.shape[0] // 2, crop.shape[1] // 2])


def test_splat_samples_texels():
    from core.creation.template_head import create_avatar_mesh
    from core.creation.texture import splat_albedo

    mesh = create_avatar_mesh(subdivisions=16)
    img = _fake_frame(4, 320)
    rng = np.random.default_rng(0)
    lms = rng.uniform(80, 240, size=(478, 3)).astype(np.float32)
    lms[1] = [160, 170, 0]
    lms[234] = [90, 160, 0]
    lms[454] = [230, 160, 0]
    lms[33] = [120, 130, 0]
    lms[263] = [200, 130, 0]
    f = ScoredFrame(0, img, 80.0, 128.0, 0.0, 1.0)
    tex = splat_albedo(mesh, [f], [lms], size=128)
    assert tex.shape == (128, 128, 3)
    colors = np.unique(tex.reshape(-1, 3), axis=0)
    assert len(colors) > 1


def test_scored_frame_fields():
    img = _fake_frame(3)
    f = ScoredFrame(0, img, sharpness(img), brightness(img), 0.0, 1.0)
    assert f.index == 0
    assert f.image.shape == img.shape
