from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Union
from app.inference import (
    read_image_bytes, run_inference, draw_annotations, generate_report_html,
    encode_image_np_to_base64, np_to_pil
)

app = FastAPI()

# Add CORS middleware with explicit origins - this is the critical part
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Point(BaseModel):
    x: int
    y: int

class Contour(BaseModel):
    points: List[Point]

class Detection(BaseModel):
    class_: str
    display_name: str
    confidence: float
    bbox: List[int]
    is_grossly_carious: bool
    is_internal_resorption: bool
    has_mask: Optional[bool] = False
    mask_contours: Optional[List[Contour]] = None

class AnalyzeResponse(BaseModel):
    width: int
    height: int
    detections: List[Detection]
    annotated_image_base64_png: str

class ReportResponse(BaseModel):
    width: int
    height: int
    detections: List[Detection]
    annotated_image_base64_png: str
    report_html: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    data = await file.read()
    img_np = read_image_bytes(data)
    detections = run_inference(img_np)
    annotated_np, processed = draw_annotations(img_np, detections)
    annotated_b64 = encode_image_np_to_base64(annotated_np)
    payload = []
    for d in processed:
        detection = Detection(
            class_=d["class"],
            display_name=d["display_name"],
            confidence=float(d["confidence"]),
            bbox=[int(v) for v in d["bbox"]],
            is_grossly_carious=bool(d["is_grossly_carious"]),
            is_internal_resorption=bool(d["is_internal_resorption"]),
            has_mask=bool(d.get("has_mask", False)),
            mask_contours=d.get("mask_contours")
        )
        payload.append(detection)
    h, w = img_np.shape[:2]

@app.post("/report", response_model=ReportResponse)
async def report(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    data = await file.read()
    img_np = read_image_bytes(data)
    detections = run_inference(img_np)
    annotated_np, processed = draw_annotations(img_np, detections)
    annotated_pil = np_to_pil(annotated_np)
    annotated_b64 = encode_image_np_to_base64(annotated_np)
    html = generate_report_html(processed, annotated_pil)
    h, w = img_np.shape[:2]
    payload = []
    for d in processed:
        detection = Detection(
            class_=d["class"],
            display_name=d["display_name"],
            confidence=float(d["confidence"]),
            bbox=[int(v) for v in d["bbox"]],
            is_grossly_carious=bool(d["is_grossly_carious"]),
            is_internal_resorption=bool(d["is_internal_resorption"]),
            has_mask=bool(d.get("has_mask", False)),
            mask_contours=d.get("mask_contours")
        )
        payload.append(detection)
    return ReportResponse(width=w, height=h, detections=payload, annotated_image_base64_png=annotated_b64, report_html=html)
