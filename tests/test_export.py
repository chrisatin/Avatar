import json
import struct

from core.creation.export import read_glb_header, write_glb
from core.creation.template_head import create_avatar_mesh


def test_glb_export(tmp_path):
    mesh = create_avatar_mesh(subdivisions=12)
    mesh.texture = __import__("numpy").zeros((8, 8, 3), dtype="uint8")
    path = write_glb(tmp_path / "a.glb", mesh, name="TestAvatar", texture=mesh.texture)
    assert path.exists()
    header = read_glb_header(path)
    assert header["magic"] == 0x46546C67
    assert header["version"] == 2
    assert header["morphs"] == 51
    assert header["has_texture"] is True
    with open(path, "rb") as fh:
        data = fh.read()
    total = struct.unpack("<I", data[8:12])[0]
    assert total == len(data)
    jlen = struct.unpack("<I", data[12:16])[0]
    meta = json.loads(data[20 : 20 + jlen])
    assert meta["meshes"][0]["primitives"][0]["extras"]["targetNames"][0] == "browDownLeft"
    assert "POSITION" in meta["meshes"][0]["primitives"][0]["targets"][0]
