from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from app.pipeline.align_fuse import align_and_fuse
from app.pipeline.upscale import upscale
from app.pipeline.color_grade import grade

app = FastAPI(title="AI Camera Backend", version="0.1.0")


def _read_upload_as_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode one of the uploaded images.")
    return img


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
async def process_burst(
    frames: List[UploadFile] = File(..., description="2-4 burst photos"),
    fusion_method: str = Form("mertens"),          # "mertens" | "average"
    upscale_method: str = Form("lanczos"),          # "lanczos" | "realesrgan"
    target_long_edge: int = Form(3840),             # 3840 = 4K long edge
    grade_style: str = Form("auto"),                # "auto" | "lut"
    lut_path: Optional[str] = Form(None),            # path to a .cube file, required if grade_style="lut"
):
    if len(frames) < 2 or len(frames) > 4:
        raise HTTPException(status_code=400, detail="Upload 2-4 burst frames.")

    images = [_read_upload_as_bgr(await f.read()) for f in frames]

    # 1. Align + fuse the burst into one sharp, low-noise photo
    fused = align_and_fuse(images, method=fusion_method)

    # 2. Upscale toward the target resolution
    upscaled = upscale(fused, target_long_edge=target_long_edge, method=upscale_method)

    # 3. Color grade
    final = grade(upscaled, style=grade_style, lut_path=lut_path)

    ok, buf = cv2.imencode(".jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output image.")

    return Response(content=buf.tobytes(), media_type="image/jpeg")