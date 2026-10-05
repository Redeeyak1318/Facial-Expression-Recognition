import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import sys

# Add project root to sys.path to resolve imports properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.cnn_v2b import ResidualCNN
EXPRESSION_LABELS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

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
    "IMG_20240926_133913.jpg": "no_face"
}

idx_to_label = EXPRESSION_LABELS

def get_face_bbox(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, img
    
    # Select largest face
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces_sorted[0], img

def apply_padding(bbox, img_shape, pad_pct):
    x, y, w, h = bbox
    img_h, img_w = img_shape[:2]
    
    pad_w = int(w * pad_pct)
    pad_h = int(h * pad_pct)
    
    new_x = max(0, x - pad_w)
    new_y = max(0, y - pad_h)
    new_w = min(img_w - new_x, w + 2 * pad_w)
    new_h = min(img_h - new_y, h + 2 * pad_h)
    
    return [new_x, new_y, new_w, new_h]

def run_diagnostic():
    device = torch.device('cpu')
    model = ResidualCNN(num_classes=7)
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'V2', 'V2_B_residual_cnn', 'checkpoint', 'best_model.pth'))
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    validation_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'validation_samples'))
    
    paddings = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    results = {}
    
    for filename, manual_label in MANUAL_LABELS.items():
        if manual_label == "no_face":
            continue
            
        filepath = os.path.join(validation_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        bbox, img_bgr = get_face_bbox(filepath)
        if bbox is None:
            results[filename] = {"error": "no_face_detected_by_detector"}
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        results[filename] = {
            "manual_label": manual_label,
            "crops": {}
        }
        
        for p in paddings:
            p_bbox = apply_padding(bbox, img_bgr.shape, p)
            crop_img = pil_img.crop((p_bbox[0], p_bbox[1], p_bbox[0]+p_bbox[2], p_bbox[1]+p_bbox[3]))
            crop_gray = crop_img.convert("L")
            crop_resized = crop_gray.resize((48, 48), Image.BILINEAR)
            img_array = np.array(crop_resized, dtype=np.float32) / 255.0
            tensor_img = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(tensor_img)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()
                
            pred_idx = np.argmax(probs)
            pred_class = idx_to_label[pred_idx]
            conf = probs[pred_idx]
            
            prob_dict = {idx_to_label[i]: float(probs[i]) for i in range(7)}
            
            results[filename]["crops"][f"{int(p*100)}%"] = {
                "predicted_class": pred_class,
                "confidence": float(conf),
                "probabilities": prob_dict,
                "correct": (pred_class == manual_label)
            }

    pad_keys = ["0%", "10%", "20%", "30%", "40%"]
    metrics = {k: {"correct": 0, "incorrect": 0, "total": 0, "conf_correct": [], "conf_incorrect": []} for k in pad_keys}
    
    for filename, data in results.items():
        if "error" in data:
            continue
        for k in pad_keys:
            if k in data["crops"]:
                metrics[k]["total"] += 1
                c = data["crops"][k]
                if c["correct"]:
                    metrics[k]["correct"] += 1
                    metrics[k]["conf_correct"].append(c["confidence"])
                else:
                    metrics[k]["incorrect"] += 1
                    metrics[k]["conf_incorrect"].append(c["confidence"])

    for k in pad_keys:
        m = metrics[k]
        m["accuracy"] = m["correct"] / m["total"] if m["total"] > 0 else 0
        all_conf = m["conf_correct"] + m["conf_incorrect"]
        m["avg_conf"] = sum(all_conf) / len(all_conf) if all_conf else 0
        m["avg_conf_correct"] = sum(m["conf_correct"]) / len(m["conf_correct"]) if m["conf_correct"] else 0
        m["avg_conf_incorrect"] = sum(m["conf_incorrect"]) / len(m["conf_incorrect"]) if m["conf_incorrect"] else 0
        
        del m["conf_correct"]
        del m["conf_incorrect"]

    output = {
        "metrics": metrics,
        "results": results
    }
    
    with open(os.path.join(validation_dir, "preprocessing_diagnostic.json"), "w") as f:
        json.dump(output, f, indent=4)
        
    md = [
        "# Preprocessing Diagnostic Report",
        "\n*Diagnostic only \u2014 not an official model evaluation.*",
        "\n## 1. Method",
        "This diagnostic evaluates how variations in face-crop padding affect the V2-B model predictions. "
        "For each detected face, multiple crops were generated using 0%, 10%, 20% (current production baseline), 30%, and 40% padding around the bounding box. "
        "The model weights were not changed, and the diagnostic was run against a small manual validation set of engineering samples.",
        "\n## 2. Crop Strategies",
        "- **0% Padding:** Tight crop exactly bounding the face.",
        "- **10% Padding:** Modest context inclusion.",
        "- **20% Padding:** Current baseline padding.",
        "- **30% Padding:** Broad context.",
        "- **40% Padding:** Very broad context.",
        "\n## 3. Per-Image Comparison Table",
        "| Filename | Manual Label | 0% Pred | 10% Pred | 20% Pred | 30% Pred | 40% Pred |",
        "|----------|--------------|---------|----------|----------|----------|----------|"
    ]
    
    stability_issues = []
    
    for filename, data in results.items():
        if "error" in data:
            md.append(f"| {filename} | {MANUAL_LABELS.get(filename)} | {data['error']} | | | | |")
            continue
        
        lbl = data["manual_label"]
        preds = []
        confs = []
        for k in pad_keys:
            preds.append(data["crops"][k]["predicted_class"])
            confs.append(data["crops"][k]["confidence"])
            
        md.append(f"| {filename} | {lbl} | " + " | ".join([f"{p} ({c:.2f})" for p, c in zip(preds, confs)]) + " |")
        
        unique_preds = set(preds)
        if len(unique_preds) > 1:
            stability_issues.append(f"- **{filename}**: Prediction varied across paddings (predicted: {', '.join(unique_preds)}).")
        
        conf_range = max(confs) - min(confs)
        if conf_range > 0.15:
            stability_issues.append(f"- **{filename}**: Confidence fluctuated significantly (range: {conf_range:.2f}).")
            
        if lbl in unique_preds and len(unique_preds) > 1:
            stability_issues.append(f"- **{filename}**: Correct prediction appears at some paddings but is lost at others.")

    md.extend([
        "\n## 4. Accuracy by Crop Strategy",
        "| Strategy | Correct | Incorrect | Accuracy |",
        "|----------|---------|-----------|----------|"
    ])
    for k in pad_keys:
        m = metrics[k]
        md.append(f"| {k} | {m['correct']} | {m['incorrect']} | {m['accuracy']:.2%} |")

    md.extend([
        "\n## 5. Average Confidence by Crop Strategy",
        "| Strategy | Avg Conf (All) | Avg Conf (Correct) | Avg Conf (Incorrect) |",
        "|----------|----------------|--------------------|----------------------|"
    ])
    for k in pad_keys:
        m = metrics[k]
        md.append(f"| {k} | {m['avg_conf']:.4f} | {m['avg_conf_correct']:.4f} | {m['avg_conf_incorrect']:.4f} |")
        
    md.extend([
        "\n## 6. Prediction Stability Observations"
    ])
    if stability_issues:
        md.extend(stability_issues)
    else:
        md.append("- All predictions were perfectly stable across crop paddings.")
        
    md.extend([
        "\n## 7. Limitations",
        "- - The diagnostic uses only 9 manually labeled face samples and is too small to support reliable generalization claims.",
        "- Labels were assigned manually and may not accurately reflect true psychological expressions.",
        "\nNext decision requires review of this diagnostic together with broader real-world validation."
    ])
    
    with open(os.path.join(validation_dir, "PREPROCESSING_DIAGNOSTIC_REPORT.md"), "w") as f:
        f.write("\n".join(md))

if __name__ == "__main__":
    run_diagnostic()
