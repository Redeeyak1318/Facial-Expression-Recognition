"""
Real-world image validation script.
Processes only the 11 original benchmark photos from the V4-A benchmark,
evaluating face detection, predicted expression, confidence, and bounding box.
Saves results in both JSON and CSV formats.
"""

import os
import sys
import argparse
import json
import csv
from PIL import Image, UnidentifiedImageError

# Add project root to sys.path to resolve imports properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.predictor import predict_expression_from_photo

# The 11 original photos listed in the V4-A benchmark
V4_A_BENCHMARK_IMAGES = [
    "IMG_20250609_122532.jpg",
    "IMG_20250829_140834.jpg",
    "IMG_20260410_152237.jpg",
    "IMG-20241029-WA0009(1).jpg",
    "IMG-20241128-WA0036.jpg",
    "IMG-20241207-WA0021.jpg",
    "IMG-20250830-WA0048.jpg",
    "IMG20200413144756.jpg",
    "IMG_20260430_211801_813.webp",
    "IMG_20250903_074442_988.jpg",
    "IMG_20240926_133913.jpg"
]

def validate_images(input_dir="validation_samples", image_list=None, output_json=None, output_csv=None):
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return None

    if image_list is None:
        image_list = V4_A_BENCHMARK_IMAGES

    if output_json is None:
        output_json = os.path.join(input_dir, "validation_results.json")
    if output_csv is None:
        output_csv = os.path.join(input_dir, "validation_results.csv")

    results = []
    stats = {
        "total_images": 0,
        "successful_face_detections": 0,
        "no_face_images": 0,
        "invalid_images": 0,
        "inference_errors": 0,
        "class_prediction_counts": {
            "angry": 0,
            "disgust": 0,
            "fear": 0,
            "happy": 0,
            "neutral": 0,
            "sad": 0,
            "surprise": 0
        },
        "average_confidence": 0.0
    }
    
    total_confidence = 0.0
    
    for filename in image_list:
        filepath = os.path.join(input_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Skipping {filename}: file not found in {input_dir}")
            continue

        stats["total_images"] += 1
        
        try:
            # Check if it's an image
            try:
                img = Image.open(filepath)
                width, height = img.size
            except UnidentifiedImageError:
                stats["invalid_images"] += 1
                results.append({
                    "filename": filename,
                    "detection_status": "invalid_image",
                    "face_detected": False,
                    "predicted_expression": None,
                    "predicted_class": None,
                    "confidence": None,
                    "bounding_box": None,
                    "bbox": None,
                    "probabilities": None,
                    "width": None,
                    "height": None,
                    "error": "invalid image"
                })
                continue
                
            # Run inference
            result = predict_expression_from_photo(filepath)
            
            stats["successful_face_detections"] += 1
            pred_class = result["predicted_class"]
            conf = float(result["confidence"])
            bbox = result["bbox"]
            
            stats["class_prediction_counts"][pred_class] += 1
            total_confidence += conf
            
            results.append({
                "filename": filename,
                "detection_status": "detected",
                "face_detected": True,
                "predicted_expression": pred_class,
                "predicted_class": pred_class,
                "confidence": conf,
                "bounding_box": bbox,
                "bbox": bbox,
                "probabilities": result.get("probabilities", {}),
                "width": width,
                "height": height,
                "error": None
            })
            
        except ValueError as e:
            err_msg = str(e)
            if "face" in err_msg.lower() or "detect" in err_msg.lower():
                stats["no_face_images"] += 1
                results.append({
                    "filename": filename,
                    "detection_status": "no_face",
                    "face_detected": False,
                    "predicted_expression": None,
                    "predicted_class": None,
                    "confidence": None,
                    "bounding_box": None,
                    "bbox": None,
                    "probabilities": None,
                    "width": width if 'width' in locals() else None,
                    "height": height if 'height' in locals() else None,
                    "error": err_msg
                })
            else:
                stats["inference_errors"] += 1
                results.append({
                    "filename": filename,
                    "detection_status": "error",
                    "face_detected": False,
                    "predicted_expression": None,
                    "predicted_class": None,
                    "confidence": None,
                    "bounding_box": None,
                    "bbox": None,
                    "probabilities": None,
                    "width": width if 'width' in locals() else None,
                    "height": height if 'height' in locals() else None,
                    "error": err_msg
                })
        except Exception as e:
            stats["inference_errors"] += 1
            results.append({
                "filename": filename,
                "detection_status": "error",
                "face_detected": False,
                "predicted_expression": None,
                "predicted_class": None,
                "confidence": None,
                "bounding_box": None,
                "bbox": None,
                "probabilities": None,
                "width": width if 'width' in locals() else None,
                "height": height if 'height' in locals() else None,
                "error": str(e)
            })

    if stats["successful_face_detections"] > 0:
        stats["average_confidence"] = total_confidence / stats["successful_face_detections"]
        
    print("\n=============================")
    print("VALIDATION SUMMARY (V4-A Benchmark 11 Photos)")
    print("=============================")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Successful face detections: {stats['successful_face_detections']}")
    print(f"No-face images: {stats['no_face_images']}")
    print(f"Invalid images: {stats['invalid_images']}")
    print(f"Inference errors: {stats['inference_errors']}")
    print(f"Average confidence: {stats['average_confidence']:.4f}")
    print("\nClass Prediction Counts:")
    for cls, count in stats["class_prediction_counts"].items():
        print(f"  {cls}: {count}")

    print("\nPer-Image Results:")
    print(f"{'Filename':<30} {'Status':<12} {'Expression':<12} {'Confidence':<12} {'Bounding Box'}")
    print("-" * 85)
    for r in results:
        status = r["detection_status"]
        expr = r["predicted_expression"] or "-"
        conf = f"{r['confidence']:.4f}" if r["confidence"] is not None else "-"
        bbox = str(r["bounding_box"]) if r["bounding_box"] is not None else "-"
        print(f"{r['filename']:<30} {status:<12} {expr:<12} {conf:<12} {bbox}")

    # 1. Save JSON output
    with open(output_json, 'w') as f:
        json.dump({"summary": stats, "results": results}, f, indent=4)
    print(f"\nSaved detailed JSON results to {output_json}")

    # 2. Save CSV output
    csv_columns = [
        "filename",
        "detection_status",
        "predicted_expression",
        "confidence",
        "bounding_box",
        "face_detected",
        "error"
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filename": r["filename"],
                "detection_status": r["detection_status"],
                "predicted_expression": r["predicted_expression"] or "",
                "confidence": f"{r['confidence']:.4f}" if r["confidence"] is not None else "",
                "bounding_box": json.dumps(r["bounding_box"]) if r["bounding_box"] is not None else "",
                "face_detected": r["face_detected"],
                "error": r.get("error") or ""
            })
    print(f"Saved per-image CSV results to {output_csv}")

    return {"summary": stats, "results": results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate model on real images (V4-A benchmark 11 photos).")
    parser.add_argument("--input_dir", type=str, default="validation_samples", help="Directory containing real images.")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save JSON results.")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save CSV results.")
    args = parser.parse_args()
    
    validate_images(
        input_dir=args.input_dir,
        output_json=args.output_json,
        output_csv=args.output_csv
    )
