"""
Test script for the V2-B standalone face detection and inference pipeline.
"""
import sys
import os
import json
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference.face_detector import detect_face
from src.inference.predictor import predict_expression_from_photo
from src.data.split import get_test_paths

def create_mock_photo(face_path):
    """
    Pads a 48x48 dataset face onto a larger background to simulate a real photo.
    This ensures OpenCV Haar Cascades can comfortably detect it.
    """
    face = Image.open(face_path).convert("RGB")
    face = face.resize((150, 150), Image.BILINEAR) # scale up the face
    photo = Image.new('RGB', (300, 300), color=(200, 200, 200)) # gray background
    photo.paste(face, (75, 75)) # paste face in the middle
    return photo

def run_test():
    print("=" * 60)
    print("FACE DETECTION PIPELINE TEST (V2-B)")
    print("=" * 60)
    
    test_paths = get_test_paths()
    sample_image_path = test_paths['happy'][0]
    print(f"Using dataset face: {sample_image_path}")
    
    # Create mock photo
    photo = create_mock_photo(sample_image_path)
    
    # 1 & 2. Detect face and verify it's found
    print("\n1. Testing isolated detect_face()...")
    detection = detect_face(photo)
    print("Face successfully detected.")
    
    # 4. Verify cropped image is non-empty and has 0% padding
    cropped = detection["face"]
    bbox = detection["bbox"]
    assert cropped.size[0] > 0 and cropped.size[1] > 0, "Cropped image is empty"
    assert cropped.size[0] == bbox[2], f"Expected 0% padding width. Bbox w: {bbox[2]}, Crop w: {cropped.size[0]}"
    assert cropped.size[1] == bbox[3], f"Expected 0% padding height. Bbox h: {bbox[3]}, Crop h: {cropped.size[1]}"
    print(f"Cropped image size: {cropped.size} (Verified 0% padding tight crop)")
    
    # 5. End-to-end pipeline
    print("\n2. Running End-to-End Pipeline (predict_expression_from_photo)...")
    result = predict_expression_from_photo(photo)
    
    # 3. Verify bbox has 4 numeric values
    assert "bbox" in result, "Missing bbox in result"
    bbox = result["bbox"]
    assert len(bbox) == 4, "bbox must have 4 coordinates"
    assert all(isinstance(x, (int, float)) for x in bbox), "bbox coordinates must be numeric"
    print(f"Bounding Box detected: {bbox}")
    
    # Verify other fields
    assert "predicted_class" in result
    assert "confidence" in result
    assert "probabilities" in result
    
    # 6. Verify probabilities sum to ~1
    probs = result["probabilities"]
    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities do not sum to 1. Sum = {prob_sum}"
    print(f"Probabilities sum to ~1 ({prob_sum:.6f}).")
    
    print("\nAll end-to-end assertions passed!")
    
    print("\nOutput Details:")
    print(f"Predicted Expression: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nProbability Distribution:")
    for cls, p in probs.items():
        print(f"  {cls}: {p:.4f}")
        
    # 7. No-face negative test
    print("\n3. Running Negative Test (No face)...")
    black_img = Image.new('RGB', (200, 200), color='black')
    try:
        detect_face(black_img)
        print("FAILED: Detector found a face in a solid black image!")
        sys.exit(1)
    except ValueError as e:
        if "No face detected" in str(e):
            print("Successfully caught expected FaceNotDetectedError/ValueError.")
        else:
            print(f"Caught ValueError, but unexpected message: {e}")
            sys.exit(1)
            
    print("\n============================================================")
    print("TEST STATUS: SUCCESS")
    print("============================================================")

if __name__ == "__main__":
    run_test()
