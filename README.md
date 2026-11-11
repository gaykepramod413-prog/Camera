# 📸 AI Camera Backend

<div align="center">

**A FastAPI image-processing service that combines burst photos into a cleaner, sharper, upscaled output.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

</div>

## ✨ What It Does

Upload **2–4 burst frames** and the API processes them through three stages:

1. **Alignment and fusion** — aligns the input frames and combines them using the selected fusion method.
2. **Upscaling** — resizes the fused image toward the requested long-edge resolution.
3. **Color grading** — applies an automatic grade or a supplied `.cube` LUT.

The processed result is returned as a high-quality JPEG.

## 🧰 Tech Stack

- Python
- FastAPI and Uvicorn
- OpenCV
- NumPy
- python-multipart for file uploads

## 📁 Project Structure

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

## ⚙️ Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/gaykepramod413-prog/Camera.git
   cd Camera
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the API:

   ```bash
   uvicorn app.main:app --reload
   ```

5. Open the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## 🔌 API

### Health check

`GET /health`

Returns a simple service status:

```json
{
  "status": "ok"
}
```

### Process a burst

`POST /process` — multipart form-data.

| Field | Description | Default |
|---|---|---|
| frames | 2–4 uploaded image files | Required |
| fusion_method | `mertens` or `average` | `mertens` |
| upscale_method | `lanczos` or `realesrgan` | `lanczos` |
| target_long_edge | Desired long-edge size in pixels | `3840` |
| grade_style | `auto` or `lut` | `auto` |
| lut_path | Path to a `.cube` LUT when using LUT grading | None |

Example with cURL:

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

## 🔬 Pipeline Modules

- `align_fuse.py` — burst alignment and image fusion.
- `upscale.py` — image resizing/upscaling.
- `color_grade.py` — automatic or LUT-based color grading.

## ⚠️ Notes

- The API accepts 2–4 images per processing request.
- The default output is JPEG encoded at quality 95.
- Real-ESRGAN and LUT modes depend on the corresponding model/runtime or LUT file being available in your environment.

## 🗺️ Future Improvements

- Add upload-size and image-dimension limits.
- Add tests for each pipeline stage.
- Add progress reporting for long-running image processing.
- Document and package optional model dependencies separately.

---

<div align="center">

**Built by [Pramod Gayke](https://github.com/gaykepramod413-prog)**

</div>
