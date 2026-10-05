"""
V3-A Real-World Engineering Validation

This script evaluates the V3-A best checkpoint against the same 11
validation_samples images used for V2-B real-world validation.

IMPORTANT:
  - Engineering validation ONLY — not an official accuracy/generalization claim.
  - Does NOT modify production inference code, face_detector.py, or predictor.py.
  - Does NOT use the official test set.
  - Does NOT add these images to training.
  - Uses the exact same Haar cascade detection, largest-face selection,
    0% tight crop, grayscale → 48×48 → /255 pipeline as production.
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from PIL import Image
from datetime import datetime

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.cnn_v2b import ResidualCNN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

MANUAL_LABELS = {
    "IMG_20250609_122532.jpg": "happy",
    "IMG_20250829_140834.jpg": "neutral",
    "IMG_20260410_152237.jpg": "neutral",
    "IMG-20241029-WA0009(1).jpg": "neutral",
    "IMG-20241128-WA0036.jpg": "happy",
    "IMG-20241207-WA0021.jpg": "sad",
    "IMG-20250830-WA0048.jpg": "neutral",
    "IMG20200413144756.jpg": "sad",
    "IMG_20260430_211801_813.webp": "fear",
    "IMG_20250903_074442_988.jpg": "no_face",
    "IMG_20240926_133913.jpg": "no_face",
}

# V2-B 0% tight-crop results (from validated REAL_WORLD_VALIDATION_0PCT_REPORT.md)
V2B_RESULTS = {
    "IMG_20250609_122532.jpg":        {"pred": "sad",     "conf": 0.8704, "correct": False},
    "IMG_20250829_140834.jpg":        {"pred": "neutral", "conf": 0.5673, "correct": True},
    "IMG_20260410_152237.jpg":        {"pred": "neutral", "conf": 0.7655, "correct": True},
    "IMG-20241029-WA0009(1).jpg":     {"pred": "neutral", "conf": 0.9915, "correct": True},
    "IMG-20241128-WA0036.jpg":        {"pred": "neutral", "conf": 0.6114, "correct": False},
    "IMG-20241207-WA0021.jpg":        {"pred": "neutral", "conf": 0.8269, "correct": False},
    "IMG-20250830-WA0048.jpg":        {"pred": "neutral", "conf": 0.9974, "correct": True},
    "IMG20200413144756.jpg":          {"pred": "sad",     "conf": 0.7902, "correct": True},
    "IMG_20260430_211801_813.webp":   {"pred": "neutral", "conf": 0.4965, "correct": False},
    "IMG_20240926_133913.jpg":        {"pred": "neutral", "conf": 0.8988, "detect": "false_positive"},
    "IMG_20250903_074442_988.jpg":    {"pred": None,      "conf": None,   "detect": "true_negative"},
}

EXPERIMENT_DIR = SCRIPT_DIR
VALIDATION_DIR = os.path.join(PROJECT_ROOT, 'validation_samples')
CHECKPOINT_PATH = os.path.join(EXPERIMENT_DIR, 'checkpoint', 'best_model.pth')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ---------------------------------------------------------------------------
# Detection (exact production replica — Haar, largest face, 0% tight crop)
# ---------------------------------------------------------------------------
def detect_and_crop(image_path):
    """
    Replicates the exact production face detection pipeline:
      - Haar cascade (haarcascade_frontalface_default.xml)
      - scaleFactor=1.1, minNeighbors=5, minSize=(30,30)
      - Largest face by area
      - 0% tight bounding-box crop
      - Boundary clamping
    Returns (cropped_pil_rgb, bbox) or raises on no detection.
    """
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None, None

    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest

    # 0% padding — tight bounding box
    img_h, img_w = img_np.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    cropped = img_np[y1:y2, x1:x2]
    return Image.fromarray(cropped), [int(x), int(y), int(w), int(h)]


# ---------------------------------------------------------------------------
# Inference (grayscale → 48×48 → /255 — exact production preprocessing)
# ---------------------------------------------------------------------------
def predict(model, pil_crop, device):
    """
    Exact production CNN preprocessing:
      1. Convert to grayscale
      2. Resize to 48×48 (bilinear)
      3. Normalize /255
      4. Shape [1, 1, 48, 48]
    """
    gray = pil_crop.convert('L')
    resized = gray.resize((48, 48), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(7)},
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    # ---- Load V3-A checkpoint ---
    device = torch.device('cpu')
    print(f"Loading V3-A checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model = ResidualCNN(num_classes=7)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', '?')
        best_f1 = checkpoint.get('best_val_macro_f1', '?')
    else:
        model.load_state_dict(checkpoint)
        best_epoch = '?'
        best_f1 = '?'
    model.to(device)
    model.eval()
    print(f"Best epoch: {best_epoch}  |  Best val macro F1: {best_f1}")

    # ---- Evaluate each image ----
    per_image = []
    total_images = 0
    faces_detected = 0
    no_face_detected = 0
    false_positive_detector = 0
    errors = 0
    confidences = []
    prediction_counts = {c: 0 for c in CLASSES}

    face_correct = 0
    face_total = 0
    comparisons = []

    for filename, true_label in MANUAL_LABELS.items():
        total_images += 1
        filepath = os.path.join(VALIDATION_DIR, filename)

        if not os.path.exists(filepath):
            per_image.append({
                "filename": filename,
                "true_label": true_label,
                "status": "file_not_found",
            })
            errors += 1
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            per_image.append({
                "filename": filename,
                "true_label": true_label,
                "status": "unsupported_format",
            })
            errors += 1
            continue

        try:
            cropped, bbox = detect_and_crop(filepath)
        except Exception as e:
            per_image.append({
                "filename": filename,
                "true_label": true_label,
                "status": "detection_error",
                "error": str(e),
            })
            errors += 1
            continue

        if cropped is None:
            # No face detected
            no_face_detected += 1
            is_correct_no_detect = (true_label == "no_face")
            per_image.append({
                "filename": filename,
                "true_label": true_label,
                "face_detected": False,
                "detector_correct": is_correct_no_detect,
            })
            continue

        # Face detected
        faces_detected += 1

        if true_label == "no_face":
            false_positive_detector += 1
            # Still run inference to report what the model would say
            result = predict(model, cropped, device)
            per_image.append({
                "filename": filename,
                "true_label": true_label,
                "face_detected": True,
                "detector_correct": False,
                "false_positive": True,
                "bbox": bbox,
                **result,
            })
            confidences.append(result["confidence"])
            prediction_counts[result["predicted_class"]] += 1
            continue

        # Normal face evaluation
        result = predict(model, cropped, device)
        is_correct = (result["predicted_class"] == true_label)
        face_total += 1
        if is_correct:
            face_correct += 1

        confidences.append(result["confidence"])
        prediction_counts[result["predicted_class"]] += 1

        per_image.append({
            "filename": filename,
            "true_label": true_label,
            "face_detected": True,
            "bbox": bbox,
            "correct": is_correct,
            **result,
        })

        # Build V2-B comparison
        v2b = V2B_RESULTS.get(filename, {})
        v2b_correct = v2b.get("correct", None)

        if v2b_correct is not None:
            if is_correct and v2b_correct:
                change = "maintained_correct"
            elif is_correct and not v2b_correct:
                change = "improved"
            elif not is_correct and v2b_correct:
                change = "regressed"
            else:
                change = "maintained_incorrect"
        else:
            change = "n/a"

        comparisons.append({
            "filename": filename,
            "true_label": true_label,
            "v2b_pred": v2b.get("pred"),
            "v2b_conf": v2b.get("conf"),
            "v2b_correct": v2b_correct,
            "v3a_pred": result["predicted_class"],
            "v3a_conf": result["confidence"],
            "v3a_correct": is_correct,
            "change": change,
        })

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # ---- Summary ----
    summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": CHECKPOINT_PATH,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "total_images": total_images,
        "faces_detected": faces_detected,
        "no_face_detected": no_face_detected,
        "false_positive_detector": false_positive_detector,
        "errors": errors,
        "average_confidence": avg_conf,
        "prediction_distribution": prediction_counts,
        "face_classification_correct": face_correct,
        "face_classification_total": face_total,
        "face_classification_accuracy": face_correct / face_total if face_total > 0 else 0.0,
    }

    output = {
        "summary": summary,
        "per_image": per_image,
        "v2b_vs_v3a_comparisons": comparisons,
    }

    # ---- Save JSON ----
    json_path = os.path.join(EXPERIMENT_DIR, 'v3a_real_world_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"\nJSON results saved to {json_path}")

    # ---- Generate Markdown Report ----
    improved = sum(1 for c in comparisons if c["change"] == "improved")
    regressed = sum(1 for c in comparisons if c["change"] == "regressed")
    maintained_correct = sum(1 for c in comparisons if c["change"] == "maintained_correct")
    maintained_incorrect = sum(1 for c in comparisons if c["change"] == "maintained_incorrect")

    md = []
    md.append("# V3-A Real-World Engineering Validation Report")
    md.append("")
    md.append("*Engineering validation only — not an official accuracy/generalization claim.*")
    md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    md.append("")

    md.append("## 1. Experiment Context")
    md.append("")
    md.append("| Parameter | Value |")
    md.append("|-----------|-------|")
    md.append(f"| Checkpoint | V3-A best_model.pth |")
    md.append(f"| Best Epoch | {best_epoch} |")
    md.append(f"| Best Val Macro F1 | {best_f1} |")
    md.append(f"| V2-B Val Macro F1 (baseline) | 0.5489 |")
    md.append(f"| Preprocessing | Haar → 0% tight crop → grayscale → 48×48 → /255 |")
    md.append("")

    md.append("## 2. Detection Summary")
    md.append("")
    md.append("| Metric | Count |")
    md.append("|--------|-------|")
    md.append(f"| Total Images | {total_images} |")
    md.append(f"| Faces Detected | {faces_detected} |")
    md.append(f"| No Face Detected | {no_face_detected} |")
    md.append(f"| False Positive (Detector) | {false_positive_detector} |")
    md.append(f"| Errors | {errors} |")
    md.append("")

    md.append("## 3. Classification Summary (Manually Labeled Faces)")
    md.append("")
    md.append("| Metric | V2-B (0% crop) | V3-A |")
    md.append("|--------|----------------|------|")
    md.append(f"| Correct / Total | 5 / 9 | {face_correct} / {face_total} |")
    v2b_acc_str = f"{5/9:.1%}"
    v3a_acc_str = f"{face_correct/face_total:.1%}" if face_total > 0 else "N/A"
    md.append(f"| Accuracy | {v2b_acc_str} | {v3a_acc_str} |")
    md.append(f"| Average Confidence | 0.7816 | {avg_conf:.4f} |")
    md.append("")

    md.append("## 4. Prediction Distribution")
    md.append("")
    md.append("| Expression | V3-A Count |")
    md.append("|------------|------------|")
    for cls in CLASSES:
        md.append(f"| {cls} | {prediction_counts[cls]} |")
    md.append("")

    md.append("## 5. Per-Image V2-B vs V3-A Comparison")
    md.append("")
    md.append("| Filename | True Label | V2-B Pred (Conf) | V3-A Pred (Conf) | V2-B | V3-A | Change |")
    md.append("|----------|------------|------------------|------------------|------|------|--------|")

    for c in comparisons:
        v2b_str = f"{c['v2b_pred']} ({c['v2b_conf']:.2%})" if c['v2b_pred'] else "—"
        v3a_str = f"{c['v3a_pred']} ({c['v3a_conf']:.2%})"
        v2b_mark = "✓" if c['v2b_correct'] else "✗"
        v3a_mark = "✓" if c['v3a_correct'] else "✗"
        change_emoji = {
            "improved": "⬆ Improved",
            "regressed": "⬇ Regressed",
            "maintained_correct": "= Correct",
            "maintained_incorrect": "= Incorrect",
        }.get(c["change"], c["change"])
        md.append(f"| `{c['filename']}` | {c['true_label']} | {v2b_str} | {v3a_str} | {v2b_mark} | {v3a_mark} | {change_emoji} |")
    md.append("")

    # No-face / detector rows
    md.append("### Detector-Only Cases")
    md.append("")
    md.append("| Filename | True Label | Detection | V3-A Pred (if any) |")
    md.append("|----------|------------|-----------|-------------------|")
    for r in per_image:
        if r["true_label"] == "no_face":
            detected = r.get("face_detected", False)
            pred_str = r.get("predicted_class", "—")
            conf_str = f" ({r['confidence']:.2%})" if r.get("confidence") else ""
            status = "False Positive" if detected else "True Negative ✓"
            md.append(f"| `{r['filename']}` | no_face | {status} | {pred_str}{conf_str} |")
    md.append("")

    md.append("## 6. Change Summary")
    md.append("")
    md.append("| Category | Count |")
    md.append("|----------|-------|")
    md.append(f"| ⬆ Improved | {improved} |")
    md.append(f"| ⬇ Regressed | {regressed} |")
    md.append(f"| = Maintained Correct | {maintained_correct} |")
    md.append(f"| = Maintained Incorrect | {maintained_incorrect} |")
    md.append("")

    md.append("## 7. Limitations")
    md.append("")
    md.append("- This evaluation uses only 11 manually labeled images (9 faces + 2 no-face).")
    md.append("- The sample is far too small for statistical significance or generalization claims.")
    md.append("- Manual labels may not perfectly reflect ground-truth psychological expressions.")
    md.append("- This is an engineering diagnostic to compare V2-B and V3-A preprocessing behavior.")
    md.append("- The official test set has NOT been used.")
    md.append("")

    report_path = os.path.join(EXPERIMENT_DIR, 'V3_A_REAL_WORLD_VALIDATION_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
    print(f"Report saved to {report_path}")

    # ---- Terminal summary ----
    print(f"\n{'='*60}")
    print(f"V3-A REAL-WORLD ENGINEERING VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images:             {total_images}")
    print(f"Faces detected:           {faces_detected}")
    print(f"No face detected:         {no_face_detected}")
    print(f"False positive detector:  {false_positive_detector}")
    print(f"Errors:                   {errors}")
    print(f"Average confidence:       {avg_conf:.4f}")
    print(f"Face classification:      {face_correct}/{face_total}")
    print(f"Prediction distribution:  {prediction_counts}")
    print(f"{'='*60}")
    print(f"V2-B vs V3-A comparison:")
    print(f"  Improved:              {improved}")
    print(f"  Regressed:             {regressed}")
    print(f"  Maintained correct:    {maintained_correct}")
    print(f"  Maintained incorrect:  {maintained_incorrect}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
