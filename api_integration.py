from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# пайплайн инит

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline import BeyondColorPipeline
        _pipeline = BeyondColorPipeline()
    return _pipeline


# роутер

router = APIRouter(tags=["BeyondColor"])


@router.post(
    "/process-image",
    summary="Full pipeline on a single image (SAM segmentation + patterns)",
)
async def process_image_endpoint(
    file:            UploadFile = File(...),
    pattern_opacity: int = Form(155, ge=50, le=255),
    return_report:   bool = Form(False),
):

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        raise HTTPException(400, "Unsupported image format")

    with tempfile.TemporaryDirectory() as tmp:
        in_path  = Path(tmp) / f"input{suffix}"
        out_path = Path(tmp) / "output.png"
        in_path.write_bytes(await file.read())

        pipeline = get_pipeline()
        pipeline.pattern_opacity = pattern_opacity
        report = pipeline.process_image(str(in_path), str(out_path))

        if return_report:
            return JSONResponse(report.to_dict())

        img_bytes = out_path.read_bytes()

    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type="image/png",
        headers={
            "Content-Disposition": "attachment; filename=processed.png",
            "X-Segments-Found":    str(report.segments_total),
            "X-Compliance-Score":  str(report.figures[0].contrast_after if report.figures else 0),
        },
    )


@router.post(
    "/process-pdf",
    summary="Full pipeline on a PDF (all pages, all figures)",
)
async def process_pdf_endpoint(
    file:            UploadFile = File(...),
    pattern_opacity: int = Form(155),
    return_report:   bool = Form(False),
):
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    with tempfile.TemporaryDirectory() as tmp:
        in_path  = Path(tmp) / "input.pdf"
        out_path = Path(tmp) / "processed.pdf"
        in_path.write_bytes(await file.read())

        pipeline = get_pipeline()
        pipeline.pattern_opacity = pattern_opacity
        report = pipeline.process_pdf(str(in_path), str(out_path))

        if return_report:
            return JSONResponse(report.to_dict())

        pdf_bytes = out_path.read_bytes()

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition":    "attachment; filename=processed.pdf",
            "X-Compliance-Score":     str(report.compliance_score),
            "X-Violations-Detected":  str(report.violations_detected),
            "X-Violations-Fixed":     str(report.violations_fixed),
            "X-Figures-Processed":    str(report.figures_found),
            "X-Segments-Total":       str(report.segments_total),
            "X-Processing-Time-Ms":   str(report.processing_time_ms),
        },
    )


@router.post("/process-pdf/report", summary="JSON compliance report only")
async def pdf_report_only(file: UploadFile = File(...)):
    """Same as /process-pdf but returns only JSON, no PDF download."""
    return await process_pdf_endpoint(file, return_report=True)


@router.get("/status", summary="Pipeline status and device info")
async def status():
    from model_loader import DEVICE
    return {
        "status":  "ready",
        "device":  DEVICE,
        "models":  {
            "sam":    "MobileSAM" if DEVICE == "cpu" else "SAM vit_b",
            "layout": "PubLayNet" if _pipeline and "PubLayNet" in _pipeline.layout_mode else "OpenCV",
        },
    }


# для теста

app = FastAPI(title="BeyondColor Pipeline API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_integration:app", host="0.0.0.0", port=8001, reload=False)
