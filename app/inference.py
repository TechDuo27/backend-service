import os

os.environ["YOLO_OFFLINE"] = "True"
os.environ["ULTRALYTICS_OFFLINE"] = "True"
os.environ["ULTRALYTICS_HUB"] = "False"
os.environ["ULTRALYTICS_API_KEY"] = ""
os.environ["ULTRALYTICS_SYNC"] = "False"
os.environ["ULTRALYTICS_CHECK"] = "False"

print(">>>> Ultralytics forced into OFFLINE mode <<<<")

from pathlib import Path
from io import BytesIO
import base64
from typing import Optional, Dict, List, Any
import numpy as np
from PIL import Image
import cv2

from ultralytics import YOLO  

import logging
import datetime

CLASS_DESCRIPTIONS = {
    'SUPERNUMERARY TEETH': 'Supernumerary teeth are teeth that develop in addition to the normal number of teeth. They may occur in the primary or permanent dentition, can be single or multiple, unilateral or bilateral and may appear in any region of the dental arch.',
    'GROSSLY DECAYED': 'Grossly decayed tooth is a tooth in which dental caries has progressed extensively, leading to the destruction of a large portion of the tooth structure, often involving both enamel and dentin and in many cases extending close to or into the pulp.',
    'GROSSLY CARIOUS': 'Grossly carious refers to teeth with extensive dental caries that have caused significant destruction of the tooth structure, often involving multiple surfaces and approaching or reaching the pulp chamber. This severe form of caries requires immediate treatment to prevent further complications such as pulpal necrosis, abscess formation, or tooth loss.',
    'SPACING': 'Spacing is a type of malocclusion characterised by the presence of spaces or gaps between two or more teeth in the arch, due to descrepency between tooth size & jaw size, missing teeth or abnormal dental development.',
    'ROOT RESORPTION': 'Root resorption is a process in which hard tissues of a root like cementum and dentin are broken down & absorbed by the body, usually due to the activity of odontoclasts.',
    'ABRASION': 'Abrasion is the Pathological wearing away of the tooth structure caused by mechanical forces from external objects or factors other than tooth * to * tooth contact. It commonly results from habits such as aggressive tooth brushing, use of hard bristled brushing, or biting on hard objects leading to loss of enamel, usually at the cervical region of the teeth.',
    'DENTAL CARIES': 'Dental caries is defined as a Microbial disease of the calcified tissues of the teeth, which is characterized by the Demineralization of the inorganic portion & destruction of the organic substance of the tooth.',
    'BONE LOSS': 'Bone loss of the tooth is the Pathological reduction in the height & density of the alveolar bone that supports the teeth, leading to loosening, mobility or eventual tooth loss if untreated.',
    'CYST': 'A pathological cavity having fluid, semi fluid, or gaseous contents, which is not created by the accumulation of pus, and is usually lined by epithelium.',
    'IMPACTED TEETH': 'Impacted teeth are teeth that fail to erupt into the proper functional position in the arch within the expected time, due to obstruction by overlying gum tissue, bone, or adjacent teeth.',
    'MISSING TOOTH': 'A missing tooth refers to the absence of one or more teeth in the arch, which may occur due to congenital reasons, pathological causes (like dental caries, periodontal disease, trauma) or extraction. Missing teeth can affect oral function (chewing, speech), aesthetics (like smile & facial profile) & the overall health of the stomatognathic system by leading to drifting, tilting, supra eruption, malocclusion & bone resorption in the edentulous area.',
    'RESTORATION': 'Dental restorations are procedures that restore the function, integrity, and morphology of missing tooth structure resulting from caries or external trauma, or to improve the aesthetics of the tooth.',
    'CROWNS': 'Dental crowns are tooth-shaped caps placed over a tooth to restore its shape, size, strength, and improve its appearance. They fully encase the visible portion of a tooth at and above the gum line.',
    'ROOT CANAL TREATMENT': 'Root canal treatment is a dental procedure used to treat infection at the center of a tooth (the root canal system). The treatment involves removing the infected pulp, cleaning and disinfecting the root canal system, and then filling and sealing it.',
    'IMPLANTS': 'Dental implants are surgical components that interface with the bone of the jaw or skull to support a dental prosthesis such as a crown, bridge, denture, or facial prosthesis or to act as an orthodontic anchor.',
    'PERIAPICAL PATHOLOGY': 'Periapical pathology refers to inflammatory conditions affecting the tissues surrounding the apex of a tooth root, typically resulting from bacterial infection spreading from the pulp through the root canal system. This can manifest as periapical granulomas, cysts, or abscesses, often appearing as radiolucent areas on radiographs.',
    'BONE FRACTURE': 'Bone fracture in the oral and maxillofacial region refers to a break or crack in the jawbone (mandible or maxilla) or surrounding facial bones, commonly resulting from trauma, accidents, or pathological conditions. These fractures can affect tooth stability, occlusion, and facial aesthetics.',
    'TOOTH FRACTURE': 'Tooth fracture is a break or crack in the tooth structure that can involve the enamel, dentin, or pulp, ranging from minor enamel chips to complete crown-root fractures. These can result from trauma, bruxism, large restorations, or weakened tooth structure from extensive caries.',
    'RCT TOOTH': 'Root Canal Treatment (RCT) tooth refers to a tooth that has undergone endodontic therapy, where the infected or inflamed pulp has been removed and the root canals filled with biocompatible material. These teeth appear with radiopaque filling material in the root canals on radiographs.',
    'RESTORATIONS': 'Dental restorations are materials used to repair or replace damaged tooth structure, including fillings, inlays, onlays, and other conservative treatments. They appear with varying degrees of radiopacity depending on the material used (amalgam being highly radiopaque, composite being radiolucent to slightly radiopaque).',
    'RETAINED DECIDUOUS TOOTH': 'A retained deciduous tooth is a primary (baby) tooth that remains in the mouth beyond its normal exfoliation time, often due to the absence of a permanent successor tooth or delayed eruption. These teeth may cause alignment issues and require monitoring or extraction.',
    'ROOT STUMP': 'A root stump refers to the remaining root portion of a tooth after the crown has been lost due to extensive decay, fracture, or incomplete extraction. These retained roots can harbor bacteria, cause infection, and may require surgical removal to prevent complications.',
    'INTERNAL RESORPTION': 'Internal resorption is a rare condition where the tooth structure is resorbed from within the root canal space, typically appearing as a radiolucent enlargement of the root canal on radiographs. It usually results from trauma, extensive restorative procedures, or orthodontic movement and can lead to tooth weakening or perforation.',
    'MANDIBULAR CANAL': 'The mandibular canal (also known as inferior alveolar canal) is an anatomical structure within the mandible that contains the inferior alveolar nerve and blood vessels. It runs through the body of the mandible and is typically visible on panoramic radiographs as a radiolucent band with radiopaque borders. Accurate identification of the mandibular canal is crucial for surgical procedures to avoid nerve damage.'
}

CLASS_COLORS = {
    'SUPERNUMERARY TEETH': '#C7F464',
    'GROSSLY DECAYED': '#FF4444',
    'GROSSLY CARIOUS': '#FF1111',
    'SPACING': '#FFD700',
    'ROOT RESORPTION': '#FF8C42',
    'ABRASION': '#FFB84D',
    'DENTAL CARIES': '#FF6B6B',
    'BONE LOSS': '#9B59B6',
    'CYST': '#E74C3C',
    'IMPACTED TEETH': '#00D4FF',
    'MISSING TOOTH': '#95A5A6',
    'RESTORATION': '#FFD700',
    'CROWNS': '#3498DB',
    'ROOT CANAL TREATMENT': '#E67E22',
    'IMPLANTS': '#4ECDC4',
    'PERIAPICAL PATHOLOGY': '#C0392B',
    'BONE FRACTURE': '#8E44AD',
    'TOOTH FRACTURE': '#C0392B',
    'RCT TOOTH': '#FFB84D',
    'RESTORATIONS': '#FFD700',
    'RETAINED DECIDUOUS TOOTH': '#BDC3C7',
    'ROOT STUMP': '#7F8C8D',
    'INTERNAL RESORPTION': '#E74C3C',
    'MANDIBULAR CANAL': '#00D1B2',
}

CLASS_MAPPING_RULES = {
    'Bone Loss': 'BONE LOSS',
    'Caries': 'DENTAL CARIES',
    'Crown': 'CROWNS',
    'Cyst': 'CYST',
    'Filling': 'RESTORATIONS',
    'Fracture teeth': 'TOOTH FRACTURE',
    'Implant': 'IMPLANTS',
    'Malaligned': 'SPACING',
    'Mandibular Canal': 'MANDIBULAR CANAL',
    'Missing teeth': 'MISSING TOOTH',
    'Periapical lesion': 'PERIAPICAL PATHOLOGY',
    'Permanent Teeth': 'PERMANENT TEETH', 
    'Primary teeth': 'RETAINED DECIDUOUS TOOTH',
    'Retained root': 'ROOT STUMP',
    'Root Canal Treatment': 'RCT TOOTH',
    'Root Piece': 'ROOT STUMP',
    'Root resorption': 'ROOT RESORPTION',
    'Supra Eruption': 'SUPERNUMERARY TEETH',
    'attrition': 'ABRASION',
    'bone defect': 'BONE FRACTURE',
    'impacted tooth': 'IMPACTED TEETH',
    'AGS Medikal Implant Fixture': 'IMPLANTS',
    'AMerOss Bone Graft Material': 'BONE LOSS',
    'Amalgam Tooth Filling': 'RESTORATIONS',
    'Anthogyr Implant Fixture': 'IMPLANTS',
    'Bicon Implant Fixture': 'IMPLANTS',
    'BioHorizons Titanium Implant': 'IMPLANTS',
    'BioLife Bone Graft Material': 'BONE LOSS',
    'Biomet 3i Implant System': 'IMPLANTS',
    'Blue Sky Bio Implant': 'IMPLANTS',
    'Camlog Implant System': 'IMPLANTS',
    'Dental Caries Lesion': 'DENTAL CARIES',
    'Composite Tooth Filling': 'RESTORATIONS',
    'Cowellmedi Implant System': 'IMPLANTS',
    'Dental Crown Restoration': 'CROWNS',
    'Dentsply Implant Component': 'IMPLANTS',
    'Dentatus Narrow Implants': 'IMPLANTS',
    'Dentis Implant Fixture': 'IMPLANTS',
    'Dentium Implant System': 'IMPLANTS',
    'Euroteknika Implant System': 'IMPLANTS',
    'Tooth Filling': 'RESTORATIONS',
    'Frontier Dental Implant': 'IMPLANTS',
    'Hiossen Implant Fixture': 'IMPLANTS',
    'Implant Direct System': 'IMPLANTS',
    'Keystone Dental Implant': 'IMPLANTS',
    'Leone Implant Fixture': 'IMPLANTS',
    'Mandibular Region': 'MANDIBULAR CANAL',
    'Maxillary Region': 'MAXILLARY REGION',
    'Megagen Implant System': 'IMPLANTS',
    'Neodent Implant System': 'IMPLANTS',
    'Neoss Implant Fixture': 'IMPLANTS',
    'Nobel Biocare Dental Implant': 'IMPLANTS',
    'Novodent Implant Fixture': 'IMPLANTS',
    'NucleOSS Implant Fixture': 'IMPLANTS',
    'Osseolink Implant Fixture': 'IMPLANTS',
    'Osstem Dental Implant': 'IMPLANTS',
    'Prefabricated Dental Post': 'ROOT STUMP',
    'Retained Tooth Root': 'ROOT STUMP',
    'Root Canal Filling': 'RCT TOOTH',
    'Root Canal Obturation': 'RCT TOOTH',
    'Sterngold Mini Implants': 'IMPLANTS',
    'Straumann Tissue-Level Implant': 'IMPLANTS',
    'Titan Implant Fixture': 'IMPLANTS',
    'Zimmer Dental Implant': 'IMPLANTS',
    'croen': 'CROWNS',
    'maxillary sinus': 'MAXILLARY SINUS',
}

_models: Dict[str, YOLO] = {}
_models_folder = Path(__file__).resolve().parent

def hex_to_bgr(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def calculate_iou(b1: List[int], b2: List[int]) -> float:
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def map_unified_class(raw_name: str) -> Optional[str]:
    if raw_name in CLASS_MAPPING_RULES:
        return CLASS_MAPPING_RULES[raw_name]
    if "implant" in raw_name.lower():
        return "IMPLANTS"
    return None

def load_all_models(models_folder: Optional[Path] = None) -> Dict[str, YOLO]:
    global _models, _models_folder
    if models_folder is not None:
        _models_folder = models_folder
    if _models:
        return _models
    for model_path in _models_folder.glob("*.pt"):
        name = model_path.stem
        logging.info(f"Loading YOLO model: {name}")
        _models[name] = YOLO(str(model_path))
    if not _models:
        raise FileNotFoundError(f"No .pt models found in: {_models_folder}")
    logging.info(f"Loaded models: {list(_models.keys())}")
    return _models

def _process_detection_results(results, model_name: str) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for box in results.boxes.cpu().numpy():
        cls_idx = int(box.cls[0])
        raw_name = results.names[cls_idx]
        conf = float(box.conf[0])
        bbox = [int(v) for v in box.xyxy[0]]
        if raw_name == "Mandibular Canal":
            continue
        unified = map_unified_class(raw_name)
        if not unified:
            continue
        detections.append({
            "raw_class": raw_name,
            "unified_class": unified,
            "confidence": conf,
            "bbox": bbox,
            "mask_contours": None,
            "source_model": model_name,
            "is_grossly_carious": False,
            "is_internal_resorption": False,
        })
    return detections

def _process_segmentation_results(results, model_name: str, image_shape: tuple) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if not hasattr(results, "masks") or results.masks is None:
        return detections
    masks = results.masks.data.cpu().numpy()
    boxes = results.boxes
    img_h, img_w = image_shape[:2]
    for i, class_idx in enumerate(boxes.cls):
        cls_idx = int(class_idx)
        raw_name = results.names[cls_idx]
        if "mask" in model_name.lower() and raw_name != "Mandibular Canal":
            continue
        unified = map_unified_class(raw_name)
        if not unified:
            continue
        conf = float(boxes.conf[i])
        bbox = [int(v) for v in boxes.xyxy[i].cpu().numpy()]
        mask = masks[i]
        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_serializable = []
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = [{"x": int(p[0][0]), "y": int(p[0][1])} for p in approx]
            if points:
                contours_serializable.append({"points": points})
        if not contours_serializable:
            continue
        detections.append({
            "raw_class": raw_name,
            "unified_class": unified,
            "confidence": conf,
            "bbox": bbox,
            "mask_contours": contours_serializable,
            "source_model": model_name,
            "is_grossly_carious": False,
            "is_internal_resorption": False,
        })
    return detections

def _apply_postprocessing(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    root_resorption_count = 0
    for det in detections:
        unified = det["unified_class"]
        if unified == "DENTAL CARIES" and det["confidence"] >= 0.7:
            det_copy = det.copy()
            det_copy["is_grossly_carious"] = True
            processed.append(det_copy)
            continue
        if unified == "ROOT RESORPTION":
            root_resorption_count += 1
            if root_resorption_count % 2 == 0:
                det["is_internal_resorption"] = True
        processed.append(det)
    return processed

def run_inference(image_np: np.ndarray) -> List[Dict[str, Any]]:
    models = load_all_models()
    all_detections: List[Dict[str, Any]] = []
    for model_name, model in models.items():
        conf_threshold = 0.025 if "mask" in model_name.lower() else 0.4
        results = model(image_np, conf=conf_threshold, verbose=False)[0]
        if hasattr(results, "masks") and results.masks is not None:
            dets = _process_segmentation_results(results, model_name, image_np.shape)
        else:
            dets = _process_detection_results(results, model_name)
        all_detections.extend(dets)
    return _apply_postprocessing(all_detections)

def deduplicate_detections(detections: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    if not detections:
        return []
    mandibular_detections = [d for d in detections if d.get("unified_class") == "MANDIBULAR CANAL"]
    other_detections = [d for d in detections if d.get("unified_class") != "MANDIBULAR CANAL"]
    used = set()
    merged: List[Dict[str, Any]] = []
    for i, det in enumerate(other_detections):
        if i in used:
            continue
        group_indices = [i]
        used.add(i)
        for j in range(i + 1, len(other_detections)):
            if j in used:
                continue
            det2 = other_detections[j]
            if det["unified_class"] != det2["unified_class"]:
                continue
            if calculate_iou(det["bbox"], det2["bbox"]) > iou_threshold:
                group_indices.append(j)
                used.add(j)
        group = [other_detections[idx] for idx in group_indices]
        if len(group) == 1:
            g = group[0].copy()
            g["num_merged"] = 1
            g["source_models"] = [g["source_model"]]
            merged.append(g)
            continue
        best = max(group, key=lambda d: d["confidence"])
        all_bboxes = [d["bbox"] for d in group]
        merged_bbox = [
            min(b[0] for b in all_bboxes),
            min(b[1] for b in all_bboxes),
            max(b[2] for b in all_bboxes),
            max(b[3] for b in all_bboxes),
        ]
        with_mask = next((d for d in group if d.get("mask_contours")), None)
        merged_det = best.copy()
        merged_det["bbox"] = merged_bbox
        merged_det["num_merged"] = len(group)
        merged_det["source_models"] = sorted({d["source_model"] for d in group if d.get("source_model")})
        if with_mask:
            merged_det["mask_contours"] = with_mask.get("mask_contours")
        merged.append(merged_det)
    # Deduplicate MC by spatial location (left vs right side)
    if mandibular_detections:
        # Sort by confidence
        mandibular_detections.sort(key=lambda d: d["confidence"], reverse=True)
        
        mc_merged = []
        for mc_det in mandibular_detections:
            bbox = mc_det["bbox"]
            x_center = (bbox[0] + bbox[2]) / 2
            
            # Check if this MC overlaps with any already merged MC
            is_duplicate = False
            for existing_mc in mc_merged:
                ex_bbox = existing_mc["bbox"]
                ex_x_center = (ex_bbox[0] + ex_bbox[2]) / 2
                
                # If x-centers are close (same side) and high IoU, it's a duplicate
                if abs(x_center - ex_x_center) < 300 and calculate_iou(bbox, ex_bbox) > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                mc_det_copy = mc_det.copy()
                mc_det_copy["num_merged"] = 1
                if "source_model" in mc_det_copy:
                    mc_det_copy["source_models"] = [mc_det_copy["source_model"]]
                mc_merged.append(mc_det_copy)
        
        merged.extend(mc_merged)
        merged.sort(key=lambda d: d["confidence"], reverse=True)


    return merged


def draw_annotations(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw bounding boxes and mask contours on image for all detections.
    Special handling for Mandibular Canal (mask-only, no bbox).
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated = image_bgr.copy()
    img_h, img_w = annotated.shape[:2]
    
    for det in detections:
        unified = det.get("display_name") or det.get("unified_class")
        bbox = det["bbox"]
        color_hex = CLASS_COLORS.get(unified)
        
        if not color_hex:
            continue
            
        color_bgr = hex_to_bgr(color_hex)
        
        # Special handling for Mandibular Canal - mask only, no bbox
        if unified == "MANDIBULAR CANAL":
            mask_contours = det.get("mask_contours")
            if mask_contours:
                for contour in mask_contours:
                    pts = np.array([[p["x"], p["y"]] for p in contour["points"]], np.int32)
                    
                    # Clip coordinates to image bounds to prevent OpenCV crashes
                    pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)
                    
                    pts = pts.reshape((-1, 1, 2))
                    
                    # Draw filled mask
                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, [pts], color_bgr)
                    annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)
                    
                    # Draw outline
                    cv2.polylines(annotated, [pts], True, color_bgr, 5)
            continue  # Skip bbox drawing for Mandibular Canal

        # Draw bounding box for all other classes
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 3)
        
        # Draw mask if present (for other segmentation classes)
        mask_contours = det.get("mask_contours")
        if mask_contours:
            for contour in mask_contours:
                pts = np.array([[p["x"], p["y"]] for p in contour["points"]], np.int32)
                
                # Clip coordinates to image bounds
                pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)
                
                pts = pts.reshape((-1, 1, 2))
                
                # Draw filled mask
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [pts], color_bgr)
                annotated = cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0)
                
                # Draw outline
                cv2.polylines(annotated, [pts], True, color_bgr, 2)
    
    # Convert back to RGB for PIL
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    
    return annotated_rgb
    return annotated_rgb




def read_image_bytes(image_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))

def encode_image_np_to_base64(image_np: np.ndarray) -> str:
    pil = Image.fromarray(image_np)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)
