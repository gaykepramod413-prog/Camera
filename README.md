# 📸 AI Camera Backend

<div align="center">

**A FastAPI image-processing service for burst-photo alignment, fusion, upscaling and colour grading.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

</div>

## Overview

AI Camera Backend accepts 2–4 burst photographs and processes them into one enhanced JPEG. It uses computer-vision techniques to align frames, combine image information, upscale the result and apply colour grading.

## Processing Pipeline

```text
2–4 Uploaded Images
        |
        v
Decode images (OpenCV)
        |
        v
Sharpness scoring
        |
        v
ECC affine alignment
        |
        v
Frame fusion
(Mertens or average + denoising)
        |
        v
Upscaling
(Lanczos or optional Real-ESRGAN)
        |
        v
Colour grading
(Auto or .cube LUT)
        |
        v
High-quality JPEG response
```

## Features

- Accepts 2–4 uploaded burst frames.
- Selects a reference frame using Laplacian-variance sharpness scoring.
- Aligns frames with Enhanced Correlation Coefficient (ECC) affine registration.
- Supports Mertens exposure fusion or average fusion with coloured-image denoising.
- Supports Lanczos resizing and optional Real-ESRGAN upscaling.
- Applies gray-world white balance, contrast and saturation adjustments.
- Supports optional 3D `.cube` LUT colour grading.
- Exposes interactive API documentation through FastAPI.

## Tech Stack

- **Language:** Python
- **API:** FastAPI, Uvicorn
- **Computer vision:** OpenCV
- **Numerical processing:** NumPy
- **Uploads:** python-multipart

## Project Structure

```text
Camera/
├── app/
│   ├── main.py
│   └── pipeline/
│       ├── align_fuse.py
│       ├── upscale.py
│       └── color_grade.py
├── requirements.txt
└── README.md
```

## Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/gaykepramod413-prog/Camera.git
   cd Camera
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the API:

   ```bash
   uvicorn app.main:app --reload
   ```

5. Open the API docs at http://127.0.0.1:8000/docs.

## API Reference

### Health check

`GET /health`

Returns:

```json
{
  "status": "ok"
}
```

### Process burst images

`POST /process` — multipart form-data.

| Field | Description | Default |
|---|---|---|
| `frames` | 2–4 image files | Required |
| `fusion_method` | `mertens` or `average` | `mertens` |
| `upscale_method` | `lanczos` or `realesrgan` | `lanczos` |
| `target_long_edge` | Requested long-edge size in pixels | `3840` |
| `grade_style` | `auto` or `lut` | `auto` |
| `lut_path` | Server-side path to a `.cube` LUT | None |

Example:

```bash
curl -X POST "http://127.0.0.1:8000/process" \
  -F "frames=@frame1.jpg" \
  -F "frames=@frame2.jpg" \
  -F "fusion_method=mertens" \
  -F "upscale_method=lanczos" \
  -F "target_long_edge=3840" \
  -F "grade_style=auto" \
  --output enhanced.jpg
```

## Technical Notes

- ECC alignment uses an affine warp and falls back to the original frame if OpenCV alignment raises an error.
- Mertens fusion combines exposure/detail information; it is not a general-purpose guarantee of perfect noise removal.
- Lanczos is an interpolation-based resizing method, not an AI super-resolution model.
- Real-ESRGAN requires its optional runtime and model weights. If that path fails, the current implementation falls back to Lanczos.
- LUT grading requires a valid server-accessible `.cube` file.

## Roadmap

- Add upload-size, image-dimension and decoded-pixel limits.
- Validate fusion/upscale/grade options and LUT paths.
- Add tests for alignment, fusion, upscaling and colour grading.
- Add processing-time and memory benchmarks.
- Add progress reporting for large image requests.
- Provide before/after sample images and reproducible test inputs.

## Interview Notes

**Why align frames before fusion?** Small camera movements cause scene details to appear at different coordinates. Alignment maps frames to a common reference so fusion combines corresponding regions.

**Why select the sharpest frame?** Laplacian variance provides a simple focus/sharpness estimate. The selected frame becomes the geometric reference for the other frames.

**What is the difference between Lanczos and Real-ESRGAN?** Lanczos resizes using interpolation, while Real-ESRGAN is a learned super-resolution model that can reconstruct plausible high-frequency details.

---

<div align="center">

**Built by [Pramod Gayke](https://github.com/gaykepramod413-prog)**

</div>
