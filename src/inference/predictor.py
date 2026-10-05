import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from src.models.cnn_v2b import ResidualCNN
from src.data.config import CLASSES, NUM_CLASSES

_model = None

def get_model():
    """Lazily loads and returns the V2-B ResidualCNN."""
    global _model
    if _model is None:
        checkpoint_path = os.path.join(
    "experiments", "V3", "V3_B_moderate_augmentation",
    "checkpoint", "best_model.pth"
)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        device = torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        _model = ResidualCNN(num_classes=NUM_CLASSES)
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.to(device)
        _model.eval()
    return _model

def predict_expression(image):
    """
    Predicts the facial expression from an image using the V2-B model.
    
    Args:
        image: PIL Image or string path to an image.
        
    Returns:
        dict: Inference results matching the requested contract.
    """
    if isinstance(image, str):
        try:
            img = Image.open(image)
            img.load()
        except Exception as e:
            raise ValueError(f"Could not open image from path: {e}")
    elif isinstance(image, Image.Image):
        try:
            img = image.copy()
            img.load()
        except Exception as e:
            raise ValueError(f"Invalid PIL Image provided: {e}")
    else:
        raise ValueError("Input must be a path string or PIL Image.")

    # Preprocessing
    # 1. Grayscale
    img = img.convert('L')
    
    # 2. Resize 48x48
    img = img.resize((48, 48), Image.BILINEAR)
    
    # 3. To numpy and normalize
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    
    # 4. Shape to [1, 1, 48, 48]
    img_array = np.expand_dims(img_array, axis=(0, 1))
    input_tensor = torch.tensor(img_array)
    
    # 5. Inference
    model = get_model()
    device = torch.device("cpu")
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).tolist()
        
    # Build probability dict mapping
    prob_dict = {}
    for i, class_name in enumerate(CLASSES):
        prob_dict[class_name] = probabilities[i]
        
    # Get max probability class
    max_idx = int(torch.argmax(logits, dim=1).item())
    predicted_class = CLASSES[max_idx]
    confidence = probabilities[max_idx]
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict
    }

def predict_expression_from_photo(image):
    """
    End-to-end pipeline:
    1. Detect and crop face (tight 0% padding)
    2. Predict expression on the cropped face
    
    Args:
        image: PIL Image or string path to an image.
        
    Returns:
        dict: Inference results including bounding box.
    """
    from src.inference.face_detector import detect_face
    
    # 1. Detect face
    detection_result = detect_face(image)
    cropped_face = detection_result["face"]
    bbox = detection_result["bbox"]
    
    # 2. Predict expression on cropped face
    prediction_result = predict_expression(cropped_face)
    
    # 3. Combine results
    prediction_result["bbox"] = bbox
    
    return prediction_result

