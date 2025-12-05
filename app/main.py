from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

from .inference import (
    read_image_bytes,
    run_inference,
    deduplicate_detections,
    draw_annotations,
    encode_image_np_to_base64,
    CLASS_DESCRIPTIONS,
    CLASS_COLORS,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:8080",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Models
# ============================================================

class Point(BaseModel):
    x: int
    y: int


class Contour(BaseModel):
    points: List[Point]


class Detection(BaseModel):
    # NOTE: we keep "class_" to match your existing frontend
    class_: str
    display_name: str
    description: str
    color: str
    confidence: float
    bbox: List[int]
    mask_contours: Optional[List[Contour]] = None
    is_grossly_carious: bool = False
    is_internal_resorption: bool = False
    num_merged: int = 1
    source_models: List[str] = []


class AnalyzeResponse(BaseModel):
    width: int
    height: int
    detections: List[Detection]
    annotated_image_base64_png: str


# ============================================================
# Health
# ============================================================

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "message": "Backend is running"}


# ============================================================
# Helper: enrich detection for final response
# ============================================================

def format_detection_for_response(det: Dict[str, Any]) -> Detection:
    """
    Convert internal detection dict into API-facing Detection object.
    """
    unified = det["unified_class"]

    # Choose display_name:
    # - GROSSLY CARIOUS if flagged
    # - INTERNAL RESORPTION if flagged
    if det.get("is_grossly_carious"):
        display_name = "GROSSLY CARIOUS"
    elif det.get("is_internal_resorption"):
        display_name = "INTERNAL RESORPTION"
    else:
        display_name = unified

    description = CLASS_DESCRIPTIONS.get(display_name, CLASS_DESCRIPTIONS.get(unified, ""))
    color = CLASS_COLORS.get(display_name, CLASS_COLORS.get(unified, "#95A5A6"))

    return Detection(
        class_=det["raw_class"],
        display_name=display_name,
        description=description,
        color=color,
        confidence=float(det["confidence"]),
        bbox=[int(v) for v in det["bbox"]],
        mask_contours=det.get("mask_contours"),
        is_grossly_carious=bool(det.get("is_grossly_carious", False)),
        is_internal_resorption=bool(det.get("is_internal_resorption", False)),
        num_merged=int(det.get("num_merged", 1)),
        source_models=det.get("source_models", [det.get("source_model")])
        if det.get("source_model")
        else det.get("source_models", []),
    )


# ============================================================
# /analyze
# ============================================================

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type")

    try:
        # 1. Read image
        data = await file.read()
        img_np = read_image_bytes(data)
        h, w = img_np.shape[:2]

        # 2. Run inference across all 3 models
        raw_detections = run_inference(img_np)

        # 3. Deduplicate (IoU based, unified_class aware)
        merged_detections = deduplicate_detections(raw_detections, iou_threshold=0.5)

        # 4. Enrich for API response (description, colors, display names)
        response_detections: List[Detection] = [
            format_detection_for_response(d) for d in merged_detections
        ]

        # 5. Draw annotations using enriched detections (colors + labels)
        annotated_np = draw_annotations(
            img_np,
            [
                {
                    "bbox": d.bbox,
                    "display_name": d.display_name,
                    "confidence": d.confidence,
                    "color": d.color,
                    "mask_contours": [c.dict() for c in d.mask_contours]
                    if d.mask_contours
                    else None,
                    "unified_class": d.display_name,
                }
                for d in response_detections
            ],
        )
        annotated_b64 = encode_image_np_to_base64(annotated_np)

        return AnalyzeResponse(
            width=w,
            height=h,
            detections=response_detections,
            annotated_image_base64_png=annotated_b64,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")
