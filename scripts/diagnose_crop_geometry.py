import os
import sys
import json
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Add project root to sys.path
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
    "IMG_20260430_211801_813.webp": "fear"
}

def predict_crop(model, device, crop_pil):
    img = crop_pil.convert('L')
    img = img.resize((48, 48), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, 1))
    input_tensor = torch.tensor(img_array).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).tolist()
        
    prob_dict = {EXPRESSION_LABELS[i]: probabilities[i] for i in range(7)}
    max_idx = int(torch.argmax(logits, dim=1).item())
    predicted_class = EXPRESSION_LABELS[max_idx]
    confidence = probabilities[max_idx]
    
    return predicted_class, confidence, prob_dict

def get_square_crop(img_np, x, y, w, h):
    img_h, img_w = img_np.shape[:2]
    side = max(w, h)
    cx = x + w // 2
    cy = y + h // 2
    
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side
    
    # Shift if out of bounds
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h
        
    # Clamp just in case (e.g. image is smaller than bounding box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    
    return img_np[y1:y2, x1:x2]

def run_diagnostic():
    device = torch.device("cpu")
    model = ResidualCNN(num_classes=7)
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'V2', 'V2_B_residual_cnn', 'checkpoint', 'best_model.pth'))
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    validation_dir = os.path.join(os.path.dirname(__file__), '..', 'validation_samples')
    
    results = {}
    metrics = {
        "rect_20": {"correct": 0, "incorrect": 0, "avg_conf": 0.0, "preds": {}},
        "rect_0": {"correct": 0, "incorrect": 0, "avg_conf": 0.0, "preds": {}},
        "square_norm": {"correct": 0, "incorrect": 0, "avg_conf": 0.0, "preds": {}}
    }
    
    contact_sheet_data = []
    
    for filename, true_label in MANUAL_LABELS.items():
        img_path = os.path.join(validation_dir, filename)
        if not os.path.exists(img_path):
            continue
            
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            continue
            
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        img_h, img_w = img_np.shape[:2]
        
        results[filename] = {"manual_label": true_label, "crops": {}}
        crops = []
        
        # 1. 20% padded rectangular crop
        pad_w = int(w * 0.20)
        pad_h = int(h * 0.20)
        x1_20 = max(0, x - pad_w)
        y1_20 = max(0, y - pad_h)
        x2_20 = min(img_w, x + w + pad_w)
        y2_20 = min(img_h, y + h + pad_h)
        crop_20_np = img_np[y1_20:y2_20, x1_20:x2_20]
        crop_20_pil = Image.fromarray(crop_20_np)
        pred_20, conf_20, probs_20 = predict_crop(model, device, crop_20_pil)
        
        results[filename]["crops"]["rect_20"] = {
            "predicted_class": pred_20,
            "confidence": conf_20,
            "probabilities": probs_20,
            "correct": pred_20 == true_label,
            "width": crop_20_np.shape[1],
            "height": crop_20_np.shape[0],
            "aspect_ratio": crop_20_np.shape[1] / max(1, crop_20_np.shape[0])
        }
        crops.append(crop_20_pil)
        
        # 2. 0% tight rectangular crop
        x1_0 = max(0, x)
        y1_0 = max(0, y)
        x2_0 = min(img_w, x + w)
        y2_0 = min(img_h, y + h)
        crop_0_np = img_np[y1_0:y2_0, x1_0:x2_0]
        crop_0_pil = Image.fromarray(crop_0_np)
        pred_0, conf_0, probs_0 = predict_crop(model, device, crop_0_pil)
        
        results[filename]["crops"]["rect_0"] = {
            "predicted_class": pred_0,
            "confidence": conf_0,
            "probabilities": probs_0,
            "correct": pred_0 == true_label,
            "width": crop_0_np.shape[1],
            "height": crop_0_np.shape[0],
            "aspect_ratio": crop_0_np.shape[1] / max(1, crop_0_np.shape[0])
        }
        crops.append(crop_0_pil)
        
        # 3. 0% tight square crop
        crop_sq_np = get_square_crop(img_np, x, y, w, h)
        crop_sq_pil = Image.fromarray(crop_sq_np)
        pred_sq, conf_sq, probs_sq = predict_crop(model, device, crop_sq_pil)
        
        results[filename]["crops"]["square_norm"] = {
            "predicted_class": pred_sq,
            "confidence": conf_sq,
            "probabilities": probs_sq,
            "correct": pred_sq == true_label,
            "width": crop_sq_np.shape[1],
            "height": crop_sq_np.shape[0],
            "aspect_ratio": crop_sq_np.shape[1] / max(1, crop_sq_np.shape[0])
        }
        crops.append(crop_sq_pil)
        
        contact_sheet_data.append((filename, crops))
        
        # Update metrics
        for k, pred, conf in [("rect_20", pred_20, conf_20), ("rect_0", pred_0, conf_0), ("square_norm", pred_sq, conf_sq)]:
            metrics[k]["preds"][pred] = metrics[k]["preds"].get(pred, 0) + 1
            metrics[k]["avg_conf"] += conf
            if pred == true_label:
                metrics[k]["correct"] += 1
            else:
                metrics[k]["incorrect"] += 1
                
    num_faces = len(results)
    if num_faces > 0:
        for k in metrics:
            metrics[k]["avg_conf"] /= num_faces
            metrics[k]["accuracy"] = metrics[k]["correct"] / num_faces

    # Generate Report
    report = []
    report.append("# Crop Geometry Diagnostic Report\n")
    report.append("*Diagnostic only — not an official model evaluation.*\n")
    
    report.append("## 1. Crop Strategies Evaluated")
    report.append("- **rect_20:** 20% padded rectangular crop (historical baseline)")
    report.append("- **rect_0:** 0% tight rectangular crop (production candidate)")
    report.append("- **square_norm:** square-normalized crop derived from the detected face bounding box\n")
    
    report.append("## 2. Accuracy & Average Confidence")
    report.append("| Strategy | Correct | Incorrect | Accuracy | Avg Conf |")
    report.append("|----------|---------|-----------|----------|----------|")
    for k in ["rect_20", "rect_0", "square_norm"]:
        report.append(f"| {k} | {metrics[k]['correct']} | {metrics[k]['incorrect']} | {metrics[k].get('accuracy', 0)*100:.2f}% | {metrics[k]['avg_conf']:.4f} |")
        
    report.append("\n## 3. Prediction Distribution")
    for k in ["rect_20", "rect_0", "square_norm"]:
        preds = ", ".join(f"{c}: {cnt}" for c, cnt in sorted(metrics[k]["preds"].items()))
        report.append(f"- **{k}**: {preds}")
        
    report.append("\n## 4. Per-Image Details")
    report.append("| Filename | True Label | rect_20 | rect_0 | square_norm | Correct Strategies |")
    report.append("|----------|------------|---------|--------|----------|--------------------|")
    
    for filename, data in results.items():
        r20 = data["crops"]["rect_20"]
        r0 = data["crops"]["rect_0"]
        rsq = data["crops"]["square_norm"]
        
        correct_strats = []
        if r20["correct"]: correct_strats.append("rect_20")
        if r0["correct"]: correct_strats.append("rect_0")
        if rsq["correct"]: correct_strats.append("square_norm")
        correct_str = ", ".join(correct_strats) if correct_strats else "None"
        
        s20 = f"{r20['predicted_class']} ({r20['confidence']:.2f}, {r20['width']}x{r20['height']}, AR {r20['aspect_ratio']:.2f})"
        s0 = f"{r0['predicted_class']} ({r0['confidence']:.2f}, {r0['width']}x{r0['height']}, AR {r0['aspect_ratio']:.2f})"
        ssq = f"{rsq['predicted_class']} ({rsq['confidence']:.2f}, {rsq['width']}x{rsq['height']}, AR {rsq['aspect_ratio']:.2f})"
        
        report.append(f"| {filename} | {data['manual_label']} | {s20} | {s0} | {ssq} | {correct_str} |")
        
    report.append("\n## 5. Engineering Interpretation")
    report.append("This diagnostic helps determine if the model is sensitive to crop geometry/aspect-ratio versus just background context.")
    
    with open(os.path.join(validation_dir, 'CROP_GEOMETRY_DIAGNOSTIC.md'), 'w') as f:
        f.write("\n".join(report))
        
    with open(os.path.join(validation_dir, 'crop_geometry_diagnostic.json'), 'w') as f:
        json.write_data = {"metrics": metrics, "results": results}
        json.dump(json.write_data, f, indent=4)
        
    # Generate Contact Sheet
    fig, axes = plt.subplots(len(contact_sheet_data), 3, figsize=(9, 3 * len(contact_sheet_data)))
    for i, (filename, crops) in enumerate(contact_sheet_data):
        for j, (crop, title) in enumerate(zip(crops, ["rect_20", "rect_0", "square_norm"])):
            ax = axes[i, j]
            ax.imshow(crop)
            if i == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            if j == 0:
                ax.text(-0.1, 0.5, filename[:15]+'...', rotation=90, va='center', ha='center', transform=ax.transAxes)
                
    plt.tight_layout()
    plt.savefig(os.path.join(validation_dir, 'crop_geometry_contact_sheet.png'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_diagnostic()
