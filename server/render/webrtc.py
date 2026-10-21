from __future__ import annotations

from fractions import Fraction

from core.common.logging_utils import get_logger

logger = get_logger(__name__)


def _av():
    try:
        import av

        return av
    except ImportError:
        return None


def nvenc_available() -> bool:
    av = _av()
    if av is None:
        return False
    try:
        av.codec.Codec("h264_nvenc", "w")
        return True
    except Exception:
        return False


def make_nvenc_encoder(width: int, height: int, fps: int = 60):
    av = _av()
    if av is None:
        raise RuntimeError("PyAV not installed; pip install av --no-binary av")
    codec_name = "h264_nvenc" if nvenc_available() else "libx264"
    if codec_name == "libx264":
        logger.warning("NVENC unavailable; falling back to libx264 (CPU)")
    ctx = av.codec.context.CodecContext.create(codec_name, "w")
    ctx.width = width
    ctx.height = height
    ctx.pix_fmt = "yuv420p"
    ctx.time_base = Fraction(1, fps)
    ctx.framerate = Fraction(fps, 1)
    if codec_name == "h264_nvenc":
        ctx.options = {"preset": "ll", "tune": "ll", "rc": "vbr", "b": "3M"}
    else:
        ctx.options = {"preset": "ultrafast", "tune": "zerolatency"}
    return ctx


class NvencH264Patcher:
    @staticmethod
    def apply() -> bool:
        try:
            from aiortc.codecs import h264 as aiortc_h264
        except Exception as exc:
            logger.warning("aiortc not installed: %s", exc)
            return False
        if not nvenc_available():
            logger.warning("h264_nvenc not available in this PyAV build")
            return False

        class NvencEncoder(aiortc_h264.H264Encoder):
            def __init__(self, bitrate: int = 1_000_000) -> None:
                super().__init__(bitrate)

        aiortc_h264.H264Encoder = NvencEncoder
        logger.info("aiortc H264Encoder patched to NVENC")
        return True
