import os
import argparse
import json
from PIL import Image, UnidentifiedImageError
import sys

# Add project root to sys.path to resolve imports properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.predictor import predict_expression_from_photo

def validate_images(input_dir):
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return

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
    
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        
        # Skip directories and non-image files if any
        if not os.path.isfile(filepath) or filename.endswith('.json'):
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
                    "error": "invalid image"
                })
                continue
                
            # Run inference
            result = predict_expression_from_photo(filepath)
            
            stats["successful_face_detections"] += 1
            pred_class = result["predicted_class"]
            conf = result["confidence"]
            
            stats["class_prediction_counts"][pred_class] += 1
            total_confidence += conf
            
            results.append({
                "filename": filename,
                "face_detected": True,
                "bbox": result["bbox"],
                "predicted_class": pred_class,
                "confidence": conf,
                "probabilities": result["probabilities"],
                "width": width,
                "height": height
            })
            
        except ValueError as e:
            if "face" in str(e).lower() or "detect" in str(e).lower():
                stats["no_face_images"] += 1
                results.append({
                    "filename": filename,
                    "face_detected": False,
                    "error": str(e)
                })
            else:
                stats["inference_errors"] += 1
                results.append({
                    "filename": filename,
                    "error": str(e)
                })
        except Exception as e:
            stats["inference_errors"] += 1
            results.append({
                "filename": filename,
                "error": str(e)
            })

    if stats["successful_face_detections"] > 0:
        stats["average_confidence"] = total_confidence / stats["successful_face_detections"]
        
    print("\n=============================")
    print("VALIDATION SUMMARY")
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
        
    output_file = os.path.join(input_dir, "validation_results.json")
    with open(output_file, 'w') as f:
        json.dump({"summary": stats, "results": results}, f, indent=4)
        
    print(f"\nSaved detailed results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate V2-B model on real images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing real images.")
    args = parser.parse_args()
    
    validate_images(args.input_dir)
