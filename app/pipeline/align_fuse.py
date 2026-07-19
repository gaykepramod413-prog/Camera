import cv2
import numpy as np


def _sharpness(gray: np.ndarray) -> float:
    """Laplacian variance - a simple, fast focus/sharpness score."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _align_to_reference(reference_gray: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Align `frame` to `reference_gray` using ECC (affine warp)."""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)

    try:
        _, warp_matrix = cv2.findTransformECC(
            reference_gray, frame_gray, warp_matrix, cv2.MOTION_AFFINE, criteria
        )
        aligned = cv2.warpAffine(
            frame,
            warp_matrix,
            (reference_gray.shape[1], reference_gray.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned
    except cv2.error:
        return frame


def align_burst(frames: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """
    Align all frames to the sharpest frame in the burst.
    Returns (aligned_frames, reference_index).
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    scores = [_sharpness(g) for g in grays]
    ref_idx = int(np.argmax(scores))
    ref_gray = grays[ref_idx]

    aligned = []
    for i, f in enumerate(frames):
        aligned.append(f if i == ref_idx else _align_to_reference(ref_gray, f))
    return aligned, ref_idx


def fuse_mertens(aligned_frames: list[np.ndarray]) -> np.ndarray:
    """Mertens exposure/detail fusion -> single 8-bit BGR image."""
    merge_mertens = cv2.createMergeMertens()
    fused = merge_mertens.process(aligned_frames)  # returns float32 in [0,1]
    fused_8u = np.clip(fused * 255, 0, 255).astype(np.uint8)
    return fused_8u


def fuse_weighted_denoise(aligned_frames: list[np.ndarray]) -> np.ndarray:
    """
    Fallback fusion: average the aligned frames, then apply a light
    edge-preserving denoise pass.
    """
    stacked = np.stack(aligned_frames, axis=0).astype(np.float32)
    averaged = np.mean(stacked, axis=0).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(averaged, None, 4, 4, 7, 21)
    return denoised


def align_and_fuse(frames: list[np.ndarray], method: str = "mertens") -> np.ndarray:
    """
    Main entry point: takes 2-4 BGR frames from a burst capture and returns
    one fused, sharper, lower-noise BGR image.
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames to fuse.")

    aligned, _ = align_burst(frames)

    if method == "mertens":
        return fuse_mertens(aligned)
    elif method == "average":
        return fuse_weighted_denoise(aligned)
    else:
        raise ValueError(f"Unknown fusion method: {method}")