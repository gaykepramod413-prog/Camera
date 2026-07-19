import cv2
import numpy as np

_ESRGAN_MODEL = None  # lazy-loaded singleton


def _load_realesrgan():
    """
    Loads Real-ESRGAN (x4). Weights are free & open source (Apache 2.0),
    downloaded once from the official repo - see README.md for setup.
    """
    global _ESRGAN_MODEL
    if _ESRGAN_MODEL is not None:
        return _ESRGAN_MODEL

    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                     num_block=23, num_grow_ch=32, scale=4)
    _ESRGAN_MODEL = RealESRGANer(
        scale=4,
        model_path="weights/RealESRGAN_x4plus.pth",
        model=model,
        tile=256,
        tile_pad=10,
        half=False,
    )
    return _ESRGAN_MODEL


def upscale_realesrgan(image_bgr: np.ndarray, target_long_edge: int) -> np.ndarray:
    upsampler = _load_realesrgan()
    output, _ = upsampler.enhance(image_bgr, outscale=4)

    h, w = output.shape[:2]
    scale = target_long_edge / max(h, w)
    if scale < 1.0:
        output = cv2.resize(output, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    return output


def upscale_lanczos(image_bgr: np.ndarray, target_long_edge: int) -> np.ndarray:
    """Free, dependency-free fallback. Decent for <=2x, soft beyond that."""
    h, w = image_bgr.shape[:2]
    scale = target_long_edge / max(h, w)
    if scale <= 1.0:
        return image_bgr
    new_size = (int(w * scale), int(h * scale))
    upscaled = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_LANCZOS4)

    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
    return sharpened


def upscale(image_bgr: np.ndarray, target_long_edge: int = 3840,
            method: str = "lanczos") -> np.ndarray:
    """
    target_long_edge=3840 -> long edge becomes 4K (3840px).
    """
    if method == "realesrgan":
        try:
            return upscale_realesrgan(image_bgr, target_long_edge)
        except Exception:
            return upscale_lanczos(image_bgr, target_long_edge)
    return upscale_lanczos(image_bgr, target_long_edge)