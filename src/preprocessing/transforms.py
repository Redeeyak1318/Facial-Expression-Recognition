"""
Module for image preprocessing functions.
"""
import random
from PIL import Image
import numpy as np
from src.data.config import IMAGE_SIZE, IMAGE_MODE

def load_and_preprocess_image(image_path: str, augment: bool = False) -> np.ndarray:
    """
    Loads an image, converts it to grayscale, verifies dimensions, and normalizes.
    Optionally applies controlled augmentations.
    
    Returns:
        numpy array of shape (48, 48, 1) with values in [0, 1]
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Verify mode (do not silently convert)
        if img.mode != IMAGE_MODE:
            raise ValueError(f"Unexpected image mode {img.mode} for {image_path}. Expected {IMAGE_MODE}.")
            
        # Verify dimensions
        if img.size != IMAGE_SIZE:
            raise ValueError(f"Unexpected image size {img.size} for {image_path}. Expected {IMAGE_SIZE}.")
            
        if augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            angle = random.uniform(-10.0, 10.0)
            # fillcolor is usually 0 (black background for rotation in grayscale)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            
        # Convert to numpy array and add channel dimension
        img_array = np.array(img, dtype=np.float32)
        
        # Add channel dimension if missing (should be (48, 48) -> (48, 48, 1))
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)
            
        # Normalize to [0.0, 1.0]
        img_array = img_array / 255.0
        
        return img_array
        
    except Exception as e:
        raise IOError(f"Failed to process image {image_path}: {e}")
