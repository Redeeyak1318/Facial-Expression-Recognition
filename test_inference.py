"""
Test script for the V2-B standalone inference pipeline.
"""
import sys
import os
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference.predictor import predict_expression, _model
from src.data.split import get_test_paths
from src.data.config import CLASSES

def run_test():
    print("=" * 60)
    print("INFERENCE PIPELINE TEST (V2-B)")
    print("=" * 60)
    
    # Verify the model is not loaded initially
    import src.inference.predictor as pred
    assert pred._model is None, "Model should not be loaded before predict_expression is called."
    
    # 1. Pick a known dataset image
    test_paths = get_test_paths()
    # Pick the first image from the 'happy' class (or any class)
    sample_image_path = test_paths['happy'][0]
    print(f"Testing with image: {sample_image_path}")
    
    # 2. Run prediction
    result = predict_expression(sample_image_path)
    
    # 6. Verify model loaded successfully
    assert pred._model is not None, "Model failed to load lazily."
    print("Model loaded successfully from V2-B checkpoint.")
    
    # 3. Output shape/fields are correct
    assert "predicted_class" in result
    assert "confidence" in result
    assert "probabilities" in result
    print("Output fields are correct.")
    
    # 2. Output contains all 7 classes
    probs = result["probabilities"]
    assert len(probs) == 7
    for c in CLASSES:
        assert c in probs
    print("Output contains probabilities for all 7 classes.")
    
    # 4. Probabilities sum to ~1
    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities do not sum to 1. Sum = {prob_sum}"
    print(f"Probabilities sum to ~1 ({prob_sum:.6f}).")
    
    # 5. Confidence equals probability of predicted class
    pred_class = result["predicted_class"]
    conf = result["confidence"]
    assert probs[pred_class] == conf, f"Confidence {conf} != Probability {probs[pred_class]}"
    print("Confidence matches probability of predicted class.")
    
    print("\nAll assertions passed!")
    
    print("\nExample Result JSON:")
    print(json.dumps(result, indent=4))
    
if __name__ == "__main__":
    run_test()
