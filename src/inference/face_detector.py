import cv2
import numpy as np
from PIL import Image

def detect_face(image):
    """
    Detects the largest face in an image, crops it with a tight 0% padding,
    and returns the cropped PIL Image and the bounding box.
    
    Args:
        image: PIL Image or string path to an image.
        
    Returns:
        dict: {
            "face": cropped_pil_image,
            "bbox": [x, y, width, height]
        }
    """
    if isinstance(image, str):
        try:
            pil_img = Image.open(image).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not open image from path: {e}")
    elif isinstance(image, Image.Image):
        try:
            pil_img = image.copy().convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid PIL Image provided: {e}")
    else:
        raise ValueError("Input must be a path string or PIL Image.")
        
    img_np = np.array(pil_img)
    # PIL is RGB. Convert to grayscale for Haar cascade
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Load Haar cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade from OpenCV.")
        
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        raise ValueError("FaceNotDetectedError: No face detected in the image.")
        
    # Select the largest face by area (w * h)
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    
    # 0% padding (tight bounding box)
    pad_w = int(w * 0.0)
    pad_h = int(h * 0.0)
    
    img_h, img_w = img_np.shape[:2]
    
    # Clamp to image boundaries
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)
    
    cropped_np = img_np[y1:y2, x1:x2]
    cropped_pil = Image.fromarray(cropped_np)
    
    return {
        "face": cropped_pil,
        "bbox": [int(x), int(y), int(w), int(h)]
    }
