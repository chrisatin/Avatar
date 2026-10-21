import numpy as np

from core.creation.template_head import (
    ARKIT_51,
    AvatarMesh,
    apply_identity,
    bake_blendshapes,
    build_template,
    create_avatar_mesh,
    default_measurements,
    fit_identity_params,
)


def test_template_topology():
    mesh = build_template(16)
    assert mesh.vertices.shape[1] == 3
    assert mesh.faces.shape[1] == 3
    assert mesh.uvs.shape == (mesh.vertices.shape[0], 2)
    assert mesh.faces.max() < mesh.vertices.shape[0]


def test_blendshape_bake():
    mesh = create_avatar_mesh(subdivisions=16)
    assert mesh.blendshape_names == ARKIT_51
    assert len(mesh.blendshape_deltas) == 51
    for d in mesh.blendshape_deltas:
        assert d.shape == mesh.vertices.shape
    jaw = mesh.blendshape_deltas[ARKIT_51.index("jawOpen")]
    lower = mesh.vertices[:, 1] < -0.2
    assert np.abs(jaw[lower]).max() > 0.01
    assert np.abs(jaw[mesh.vertices[:, 1] > 0.5]).max() < 0.01


def test_identity_fit_runs():
    rng = np.random.default_rng(0)
    lms = rng.normal(0, 0.02, size=(478, 3)).astype(np.float32)
    params = fit_identity_params(lms)
    assert set(params) == set(default_measurements())
    assert all(0.0 <= v <= 1.0 for v in params.values())


def test_apply_identity_changes_mesh():
    base = build_template(16)
    wide = {k: 0.9 for k in (
        "face_width","face_length","jaw_width","chin_length","nose_length",
        "nose_width","eye_size","eye_sep","mouth_width","brow_height",
    )}
    m = apply_identity(base, wide)
    assert m.vertices.shape == base.vertices.shape
    assert not np.allclose(m.vertices, base.vertices)


def test_create_avatar_mesh_no_landmarks():
    mesh = create_avatar_mesh(landmarks=None, subdivisions=12)
    assert isinstance(mesh, AvatarMesh)
    assert len(mesh.blendshape_names) == 51
