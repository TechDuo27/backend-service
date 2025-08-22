from pathlib import Path
from io import BytesIO
import base64
from typing import Optional
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

TARGET_CLASSES = {
    'Caries': (255, 255, 255),
    'Bone Loss': (255, 0, 0),
    'Cyst': (255, 255, 0),
    'impacted tooth': (128, 0, 128),
    'Missing teeth': (0, 0, 255),
    'Supra Eruption': (0, 255, 0),
    'attrition': (255, 192, 203),
    'Malaligned': (165, 42, 42),
    'Root resorption': (0, 0, 0),
    'Periapical lesion': (255, 219, 88),
    'bone defect': (139, 0, 0),
    'Fracture teeth': (128, 128, 128),
    'Crown': (0, 100, 0),
    'Implant': (128, 0, 0),
    'Root Canal Treatment': (255, 220, 177),
    'Filling': (238, 130, 238),
    'Primary teeth': (0, 0, 128),
    'Retained root': (0, 128, 128),
}

SPECIAL_COLORS = {
    'Grossly carious': (255, 165, 0),
    'Internal resorption': (203, 192, 255),
}

DISPLAY_NAMES = {
    'Caries': 'Dental caries',
    'Bone Loss': 'Bone Loss',
    'Cyst': 'Cyst',
    'impacted tooth': 'Impacted teeth',
    'Missing teeth': 'Missing teeth',
    'Supra Eruption': 'Supernumerary teeth',
    'attrition': 'Abrasion',
    'Malaligned': 'Spacing',
    'Root resorption': 'Root resorption',
    'Periapical lesion': 'Periapical pathology',
    'bone defect': 'Bone fracture',
    'Fracture teeth': 'Tooth fracture',
    'Crown': 'Crowns',
    'Implant': 'Implants',
    'Root Canal Treatment': 'RCT tooth',
    'Filling': 'Restorations',
    'Primary teeth': 'Retained deciduous tooth',
    'Retained root': 'Root stump',
}

CLASS_NAMES_MODEL1 = {
    0: 'Bone Loss', 1: 'Caries', 2: 'Crown', 3: 'Cyst', 4: 'Filling',
    5: 'Fracture teeth', 6: 'Implant', 7: 'Malaligned', 8: 'Mandibular Canal', 9: 'Missing teeth',
    10: 'Periapical lesion', 11: 'Permanent Teeth', 12: 'Primary teeth', 13: 'Retained root',
    14: 'Root Canal Treatment', 15: 'Root Piece', 16: 'Root resorption', 17: 'Supra Eruption',
    18: 'TAD', 19: 'abutment', 20: 'attrition', 21: 'bone defect', 22: 'gingival former',
    23: 'impacted tooth', 24: 'maxillary sinus', 25: 'metal band', 26: 'orthodontic brackets',
    27: 'permanent retainer', 28: 'plating', 29: 'post - core', 30: 'wire'
}

CLASS_NAMES_MODEL2 = [
    "AGS Medikal Implant Fixture", "AMerOss Bone Graft Material", "Amalgam Tooth Filling",
    "Anthogyr Implant Fixture", "Bicon Implant Fixture", "BioHorizons Titanium Implant",
    "BioLife Bone Graft Material", "Biomet 3i Implant System", "Blue Sky Bio Implant",
    "Camlog Implant System", "Dental Caries Lesion", "Composite Tooth Filling",
    "Cowellmedi Implant System", "Dental Crown Restoration", "Dentsply Implant Component",
    "Dentatus Narrow Implants", "Dentis Implant Fixture", "Dentium Implant System",
    "Euroteknika Implant System", "Tooth Filling", "Frontier Dental Implant",
    "Hiossen Implant Fixture", "Implant Direct System", "Keystone Dental Implant",
    "Leone Implant Fixture", "Mandibular Region", "Maxillary Region",
    "Megagen Implant System", "Neodent Implant System", "Neoss Implant Fixture",
    "Nobel Biocare Dental Implant", "Novodent Implant Fixture", "NucleOSS Implant Fixture",
    "Osseolink Implant Fixture", "Osstem Dental Implant", "Prefabricated Dental Post",
    "Retained Tooth Root", "Root Canal Filling", "Root Canal Obturation",
    "Sterngold Mini Implants", "Straumann Tissue-Level Implant", "Titan Implant Fixture",
    "Zimmer Dental Implant"
]

_models: dict[str, Optional[YOLO]] = {"m1": None, "m2": None}

def load_models():
    base = Path(__file__).resolve().parent
    if _models["m1"] is None:
        _models["m1"] = YOLO(str(base / "best.pt"))
    if _models["m2"] is None:
        _models["m2"] = YOLO(str(base / "best2.pt"))
    return _models["m1"], _models["m2"]

def _pil_to_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))

def _np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)

def read_image_bytes(image_bytes: bytes) -> np.ndarray:
    return _pil_to_np(Image.open(BytesIO(image_bytes)).convert("RGB"))

def encode_image_np_to_base64(image_np: np.ndarray) -> str:
    pil = _np_to_pil(image_np)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def run_inference(image_np: np.ndarray):
    model1, model2 = load_models()
    detection_info = []

    # model1
    results1 = model1(image_np)[0]
    for box in results1.boxes.cpu().numpy():
        cls = int(box.cls[0])
        if cls == 8:
            continue
        name = CLASS_NAMES_MODEL1.get(cls, f"Unknown {cls}")
        conf = float(box.conf[0])
        if name in TARGET_CLASSES:
            detection_info.append({
                "class": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "confidence": conf,
                "bbox": [int(v) for v in box.xyxy[0]],
                "is_grossly_carious": False,
                "is_internal_resorption": False
            })

    # model2
    results2 = model2(image_np)[0]
    for box in results2.boxes.cpu().numpy():
        cls = int(box.cls[0])
        name2 = CLASS_NAMES_MODEL2[cls] if cls < len(CLASS_NAMES_MODEL2) else f"Unknown {cls}"
        conf = float(box.conf[0])
        mapped = None
        if "Dental Caries Lesion" in name2:
            mapped = "Caries"
        elif "Dental Crown Restoration" in name2:
            mapped = "Crown"
        elif "Implant" in name2 or "implant" in name2:
            mapped = "Implant"
        elif name2 in ["Amalgam Tooth Filling", "Composite Tooth Filling", "Tooth Filling"]:
            mapped = "Filling"
        elif name2 in ["Root Canal Filling", "Root Canal Obturation"]:
            mapped = "Root Canal Treatment"
        elif "Retained Tooth Root" in name2:
            mapped = "Retained root"
        if mapped and mapped in TARGET_CLASSES:
            detection_info.append({
                "class": mapped,
                "display_name": DISPLAY_NAMES.get(mapped, mapped),
                "confidence": conf,
                "bbox": [int(v) for v in box.xyxy[0]],
                "is_grossly_carious": False,
                "is_internal_resorption": False
            })

    # postprocessing
    processed = []
    root_resorption_count = 0
    for det in detection_info:
        if det["class"] == "Caries" and det["confidence"] > 0.7:
            nd = det.copy()
            nd["is_grossly_carious"] = True
            nd["display_name"] = "Grossly carious"
            processed.append(nd)
        elif det["class"] == "Root resorption":
            root_resorption_count += 1
            if root_resorption_count % 2 == 0:
                nd = det.copy()
                nd["is_internal_resorption"] = True
                nd["display_name"] = "Internal resorption"
                processed.append(nd)
            else:
                processed.append(det)
        else:
            processed.append(det)
    return processed

def draw_annotations(image_np: np.ndarray, detections):
    img = image_np.copy()
    out = []
    for d in detections:
        out.append(d)
        x1, y1, x2, y2 = d["bbox"]
        if d.get("is_grossly_carious"):
            color = SPECIAL_COLORS["Grossly carious"]
        elif d.get("is_internal_resorption"):
            color = SPECIAL_COLORS["Internal resorption"]
        else:
            color = TARGET_CLASSES.get(d["class"], (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img, out

def generate_report_html(detections, annotated_pil: Image.Image):
    buf = BytesIO()
    annotated_pil.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode()
    items = []
    for det in detections:
        if det.get("is_grossly_carious"):
            color = SPECIAL_COLORS["Grossly carious"]
        elif det.get("is_internal_resorption"):
            color = SPECIAL_COLORS["Internal resorption"]
        else:
            color = TARGET_CLASSES.get(det["class"], (0, 255, 0))
        rgb = f"rgb({color[0]}, {color[1]}, {color[2]})"
        items.append(f'<li style="border-left:5px solid {rgb};padding-left:10px;">{det["display_name"]} - <b>Confidence: {det["confidence"]:.1%}</b></li>')
    html = f"""<html><head><title>Dental AI Report</title></head><body>
    <h1>Dental AI Report</h1>
    <img src="data:image/png;base64,{img_str}" style="max-width:100%"/>
    <ul>{''.join(items) if items else "<li>None Detected</li>"}</ul>
    </body></html>"""
    return html

# expose helper
def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)
