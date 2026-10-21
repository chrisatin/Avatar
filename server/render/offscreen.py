from __future__ import annotations

import os
from typing import Any

from core.common.logging_utils import get_logger

logger = get_logger(__name__)


class OffscreenRenderer:
    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height
        self._scene = None
        self._renderer = None
        self._camera = None
        self.backend = "unavailable"
        self._init()

    def _init(self) -> None:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        try:
            import pyrender

            self._renderer = pyrender.OffscreenRenderer(self.width, self.height)
            self._scene = pyrender.Scene(
                bg_color=[0.08, 0.08, 0.1, 1.0], ambient_light=[0.5, 0.5, 0.5]
            )
            self._camera = pyrender.PerspectiveCamera(
                yfov=np_yfov(), znear=0.01, zfar=10.0
            )
            self._scene.add(
                self._camera,
                pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.8], [0, 0, 0, 1]],
            )
            self._scene.add(
                pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5),
                pose=[[1, 0, 0, 0.3], [0, 1, 0, 0.5], [0, 0, 1, 0.6], [0, 0, 0, 1]],
            )
            self.backend = "pyrender-egl"
        except Exception as exc:
            logger.warning(
                "EGL offscreen rendering unavailable (%s); "
                "set NVIDIA_DRIVER_CAPABILITIES=compute,graphics,video", exc
            )
            self.backend = "unavailable"

    @property
    def available(self) -> bool:
        return self.backend != "unavailable"

    def render_avatar_glb(self, glb_path: str) -> Any:
        if not self.available:
            raise RuntimeError("offscreen renderer unavailable")
        import numpy as np
        import pyrender
        import trimesh

        tm = trimesh.load(glb_path, force="mesh")
        if isinstance(tm, trimesh.Scene):
            tm = tm.dump(concatenate=True)
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=True)
        node = self._scene.add(mesh)
        try:
            color, _ = self._renderer.render(self._scene, flags=0)
            return color
        finally:
            self._scene.remove_node(node)

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.delete()
            except Exception:
                pass


def np_yfov() -> float:
    return 0.9
