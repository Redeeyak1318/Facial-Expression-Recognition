import os
import cv2
import numpy as np
from PIL import Image

_detector = None

def get_detector():
    """Lazily initializes the MediaPipe FaceDetector."""
    global _detector
    if _detector is None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        model_path = os.path.join(project_root, 'experiments', 'V4', 'V4_A_face_detector', 'blaze_face_full_range.tflite')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe Face Detector model not found at {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options)
        _detector = vision.FaceDetector.create_from_options(options)

    return _detector

def detect_face(image, min_confidence=0.7):
    """
    Detects the largest face in an image using MediaPipe BlazeFace, crops it with
    a tight 0% padding, and returns the cropped PIL Image and the bounding box.

    Args:
        image: PIL Image or string path to an image.
        min_confidence (float): Minimum confidence threshold for detection.

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
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

    detector = get_detector()
    detection_result = detector.detect(mp_image)

    if not detection_result.detections:
        raise ValueError("FaceNotDetectedError: No face detected in the image.")

    # Filter by confidence
    valid_detections = [det for det in detection_result.detections if det.categories[0].score >= min_confidence]
    if not valid_detections:
        raise ValueError(f"FaceNotDetectedError: No face detected with confidence >= {min_confidence}.")

    # Select the largest face by area
    best_det = None
    max_area = -1
    for det in valid_detections:
        bbox = det.bounding_box
        area = bbox.width * bbox.height
        if area > max_area:
            max_area = area
            best_det = det

    bboxC = best_det.bounding_box
    x = max(0, bboxC.origin_x)
    y = max(0, bboxC.origin_y)
    w = bboxC.width
    h = bboxC.height

    img_h, img_w = mp_image.height, mp_image.width

    # Clamp to image boundaries
    x1 = min(img_w, x)
    y1 = min(img_h, y)
    x2 = min(img_w, x1 + w)
    y2 = min(img_h, y1 + h)

    final_w = x2 - x1
    final_h = y2 - y1

    if final_w <= 0 or final_h <= 0:
        raise ValueError("FaceNotDetectedError: Face detected but cropped dimensions are invalid.")

    cropped_np = img_np[y1:y2, x1:x2]
    cropped_pil = Image.fromarray(cropped_np)

    return {
        "face": cropped_pil,
        "bbox": [int(x1), int(y1), int(final_w), int(final_h)]
    }
