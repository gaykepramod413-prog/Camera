import cv2
import numpy as np


def auto_grade(image_bgr: np.ndarray, saturation_boost: float = 1.12,
                contrast: float = 1.05) -> np.ndarray:
    """Gray-world white balance + mild contrast/saturation lift."""
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    b_avg, g_avg, r_avg = b.mean(), g.mean(), r.mean()
    gray_avg = (b_avg + g_avg + r_avg) / 3.0
    b *= gray_avg / (b_avg + 1e-6)
    g *= gray_avg / (g_avg + 1e-6)
    r *= gray_avg / (r_avg + 1e-6)
    balanced = cv2.merge([b, g, r])
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)

    contrasted = np.clip((balanced.astype(np.float32) - 128) * contrast + 128,
                          0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(contrasted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_boost, 0, 255)
    graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return graded


def _parse_cube_lut(path: str) -> np.ndarray:
    """Parses a standard 3D .cube LUT file into an (N,N,N,3) float32 array."""
    size = None
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
            elif line[0].isdigit() or line[0] == "-":
                r, g, bch = map(float, line.split())
                data.append((r, g, bch))
    if size is None:
        raise ValueError("Invalid .cube file: missing LUT_3D_SIZE")
    lut = np.array(data, dtype=np.float32).reshape((size, size, size, 3))
    return lut


def apply_cube_lut(image_bgr: np.ndarray, cube_path: str) -> np.ndarray:
    """Applies a 3D LUT via trilinear interpolation."""
    lut = _parse_cube_lut(cube_path)
    size = lut.shape[0]

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    scaled = img_rgb * (size - 1)
    idx = scaled.astype(np.int32)
    idx = np.clip(idx, 0, size - 2)

    frac = scaled - idx
    r0, g0, b0 = idx[..., 0], idx[..., 1], idx[..., 2]
    fr, fg, fb = frac[..., 0], frac[..., 1], frac[..., 2]

    def sample(dr, dg, db):
        return lut[r0 + dr, g0 + dg, b0 + db]

    c000, c100 = sample(0, 0, 0), sample(1, 0, 0)
    c010, c110 = sample(0, 1, 0), sample(1, 1, 0)
    c001, c101 = sample(0, 0, 1), sample(1, 0, 1)
    c011, c111 = sample(0, 1, 1), sample(1, 1, 1)

    fr = fr[..., None]
    fg = fg[..., None]
    fb = fb[..., None]

    c00 = c000 * (1 - fr) + c100 * fr
    c10 = c010 * (1 - fr) + c110 * fr
    c01 = c001 * (1 - fr) + c101 * fr
    c11 = c011 * (1 - fr) + c111 * fr

    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg

    result = c0 * (1 - fb) + c1 * fb
    result_bgr = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return result_bgr


def grade(image_bgr: np.ndarray, style: str = "auto", lut_path: str | None = None) -> np.ndarray:
    graded = auto_grade(image_bgr)
    if style == "lut" and lut_path:
        graded = apply_cube_lut(graded, lut_path)
    return graded