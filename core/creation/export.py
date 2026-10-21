from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

from core.common.logging_utils import get_logger
from core.creation.template_head import AvatarMesh

logger = get_logger(__name__)


def encode_png_rgb(img: np.ndarray) -> bytes:
    try:
        from PIL import Image

        import io

        buf = io.BytesIO()
        Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        h, w = img.shape[:2]
        raw = b"".join(
            b"\x00" + img[y].astype(np.uint8).tobytes() for y in range(h)
        )

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b"")
        )


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    np.add.at(normals, faces[:, 0], fn)
    np.add.at(normals, faces[:, 1], fn)
    np.add.at(normals, faces[:, 2], fn)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens < 1e-8] = 1.0
    return (normals / lens).astype(np.float32)


class _BufferBuilder:
    def __init__(self) -> None:
        self.blob = bytearray()
        self.views: list[dict] = []

    def add(self, data: bytes, target: int | None = None) -> int:
        while len(self.blob) % 4 != 0:
            self.blob.append(0)
        view: dict = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.blob.extend(data)
        self.views.append(view)
        return len(self.views) - 1


def write_glb(
    path: str | Path,
    mesh: AvatarMesh,
    name: str = "Avatar",
    texture: np.ndarray | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    normals = compute_normals(vertices, mesh.faces)
    uvs = np.ascontiguousarray(mesh.uvs, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces)
    max_index = int(faces.max()) if faces.size else 0
    use_u16 = max_index < 65536
    idx_dtype = np.uint16 if use_u16 else np.uint32
    indices = np.ascontiguousarray(faces.astype(idx_dtype).reshape(-1))

    builder = _BufferBuilder()
    idx_view = builder.add(indices.tobytes(), 34963)
    pos_view = builder.add(vertices.tobytes(), 34962)
    nrm_view = builder.add(normals.tobytes(), 34962)
    uv_view = builder.add(uvs.tobytes(), 34962)
    target_views = []
    for delta in mesh.blendshape_deltas:
        d = np.ascontiguousarray(delta.astype(np.float32))
        target_views.append(builder.add(d.tobytes(), 34962))

    tex = texture if texture is not None else mesh.texture
    png_view = None
    if tex is not None:
        png = encode_png_rgb(np.asarray(tex))
        png_view = builder.add(png)

    pos_min = vertices.min(axis=0).tolist()
    pos_max = vertices.max(axis=0).tolist()

    accessors = [
        {
            "bufferView": idx_view,
            "componentType": 5123 if use_u16 else 5125,
            "count": int(indices.shape[0]),
            "type": "SCALAR",
        },
        {
            "bufferView": pos_view,
            "componentType": 5126,
            "count": int(vertices.shape[0]),
            "type": "VEC3",
            "min": pos_min,
            "max": pos_max,
        },
        {
            "bufferView": nrm_view,
            "componentType": 5126,
            "count": int(normals.shape[0]),
            "type": "VEC3",
        },
        {
            "bufferView": uv_view,
            "componentType": 5126,
            "count": int(uvs.shape[0]),
            "type": "VEC2",
        },
    ]
    for v in target_views:
        accessors.append(
            {
                "bufferView": v,
                "componentType": 5126,
                "count": int(vertices.shape[0]),
                "type": "VEC3",
            }
        )
    target_accessor_start = 4

    primitive: dict = {
        "attributes": {"POSITION": 1, "NORMAL": 2, "TEXCOORD_0": 3},
        "indices": 0,
        "material": 0,
    }
    if mesh.blendshape_names:
        primitive["targets"] = [
            {"POSITION": target_accessor_start + i}
            for i in range(len(mesh.blendshape_names))
        ]
        primitive["extras"] = {"targetNames": list(mesh.blendshape_names)}

    gltf: dict = {
        "asset": {"version": "2.0", "generator": "avatar-pipeline/0.1"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [primitive]}],
        "accessors": accessors,
        "bufferViews": builder.views,
        "buffers": [{"byteLength": len(builder.blob)}],
        "materials": [
            {
                "name": "AvatarSkin",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.85,
                },
            }
        ],
    }
    if png_view is not None:
        gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        gltf["samplers"] = [
            {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}
        ]
        gltf["textures"] = [{"sampler": 0, "source": 0}]
        gltf["images"] = [{"bufferView": png_view, "mimeType": "image/png"}]

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "
    bin_bytes = bytes(builder.blob)
    while len(bin_bytes) % 4 != 0:
        bin_bytes += b"\x00"
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        fh.write(json_bytes)
        fh.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
        fh.write(bin_bytes)
    logger.info(
        "wrote %s (%.2f MB, %d verts, %d morphs)",
        path,
        total / 1e6,
        vertices.shape[0],
        len(mesh.blendshape_names),
    )
    return path


def read_glb_header(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        magic, version, length = struct.unpack("<III", fh.read(12))
        jlen, jtype = struct.unpack("<II", fh.read(8))
        meta = json.loads(fh.read(jlen).decode("utf-8"))
    return {
        "magic": magic,
        "version": version,
        "length": length,
        "generator": meta.get("asset", {}).get("generator"),
        "meshes": len(meta.get("meshes", [])),
        "morphs": len(
            meta.get("meshes", [{}])[0]
            .get("primitives", [{}])[0]
            .get("targets", [])
        )
        if meta.get("meshes")
        else 0,
        "has_texture": "images" in meta,
    }
