"""NeuroScan AI — ADHD EEG screening web app (FastAPI).

Run from the repository root:

    python -m uvicorn app.main:app --reload --port 8000

This is a research prototype. It is not a diagnostic device.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]

# Load local secrets (e.g. GOOGLE_MAPS_API_KEY) before importing modules that
# read them. Real environment variables take precedence over .env values.
load_dotenv(ROOT / ".env")

from app.doctors import get_doctors  # noqa: E402
from app.inference import MODEL_PATH, analyze_recording, get_metadata  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEMO_DIR = ROOT / "data" / "real" / "recordings"

DEMOS = {
    "adhd": {
        "label": "ADHD demo",
        "description": "Held-out recording of a child in the ADHD group (not used in training).",
        "path": DEMO_DIR / "v10p.npy",
    },
    "control": {
        "label": "Control demo",
        "description": "Held-out recording of a child in the control group (not used in training).",
        "path": DEMO_DIR / "v107.npy",
    },
}

ALLOWED_SUFFIXES = {".npy", ".csv", ".txt", ".mat"}
MAX_UPLOAD_BYTES = 80 * 1024 * 1024

app = FastAPI(
    title="NeuroScan AI",
    description="ADHD EEG screening research prototype with specialist referral.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API ────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health() -> dict:
    metadata = None
    try:
        metadata = get_metadata()
    except Exception:
        pass
    return {
        "status": "ok" if MODEL_PATH.exists() else "model_missing",
        "model_loaded": MODEL_PATH.exists(),
        "model": (metadata or {}).get("model"),
        "trained_at": (metadata or {}).get("trained_at"),
    }


@app.get("/api/meta")
def meta() -> dict:
    try:
        metadata = get_metadata()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Metadata unavailable: {exc}") from exc

    return {
        "model": metadata.get("model"),
        "threshold": metadata.get("threshold"),
        "n_train_subjects": metadata.get("n_train_subjects"),
        "n_train_epochs": metadata.get("n_train_epochs"),
        "n_features": metadata.get("n_features"),
        "k": metadata.get("k"),
        "excluded_subjects": metadata.get("excluded_subjects", []),
        "channels": metadata.get("channels", []),
        "bands": metadata.get("bands", {}),
        "metrics": metadata.get("metrics", {}),
        "notes": metadata.get("notes"),
        "demos": {
            key: {"label": item["label"], "description": item["description"],
                  "available": item["path"].exists()}
            for key, item in DEMOS.items()
        },
    }


async def _run_analysis(path: Path, source_fs: int, demo: Optional[str]) -> dict:
    try:
        return await run_in_threadpool(analyze_recording, path, source_fs, demo)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        detail = str(exc).replace(str(path), "the recording")
        raise HTTPException(status_code=422, detail=detail) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {type(exc).__name__}: {exc}"
        ) from exc


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    source_fs: int = Form(128),
) -> JSONResponse:
    filename = file.filename or "recording"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix or 'unknown'}'. "
                "Upload .npy, .csv, .txt, or .mat."
            ),
        )
    if not 64 <= source_fs <= 4096:
        raise HTTPException(status_code=422, detail="Sampling rate must be 64–4096 Hz.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 80 MB.")

    with tempfile.TemporaryDirectory(prefix="neuroscan_") as tmp:
        target = Path(tmp) / f"recording{suffix}"
        target.write_bytes(contents)
        result = await _run_analysis(target, source_fs, None)

    result["filename"] = filename
    result["file_size"] = len(contents)
    result["source"] = "upload"
    return JSONResponse(result)


@app.get("/api/demo/{name}")
async def demo(name: str) -> JSONResponse:
    key = name.lower().strip()
    if key not in DEMOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown demo '{name}'. Choose one of: {', '.join(DEMOS)}.",
        )
    item = DEMOS[key]
    if not item["path"].exists():
        raise HTTPException(
            status_code=404,
            detail=f"Demo recording not found: {item['path']}",
        )

    result = await _run_analysis(item["path"], 128, key)
    result["filename"] = item["path"].name
    result["file_size"] = item["path"].stat().st_size
    result["source"] = "demo"
    result["demo_label"] = item["label"]
    return JSONResponse(result)


@app.get("/api/doctors")
def doctors(
    city: str = Query("Pune", max_length=80),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lng: Optional[float] = Query(None, ge=-180, le=180),
) -> dict:
    if (lat is None) != (lng is None):
        raise HTTPException(
            status_code=422, detail="Provide both lat and lng, or neither."
        )
    try:
        return get_doctors(city=city, lat=lat, lng=lng)
    except Exception as exc:  # pragma: no cover - network dependent
        raise HTTPException(
            status_code=502, detail=f"Could not load specialist listings: {exc}"
        ) from exc


# ── FRONTEND ───────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    icon = STATIC_DIR / "favicon.svg"
    if icon.exists():
        return FileResponse(icon, media_type="image/svg+xml")
    raise HTTPException(status_code=404)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
