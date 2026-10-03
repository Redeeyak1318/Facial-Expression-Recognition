"""
Module to explicitly validate dataset integrity without loading everything automatically.
"""
import os
from PIL import Image
from typing import List
from src.data.config import TRAIN_DIR, TEST_DIR, CLASSES, IMAGE_SIZE, IMAGE_MODE

def check_dataset_integrity() -> bool:
    """
    Performs a full pass over all files to check for issues.
    Returns True if passed, False if issues were found.
    """
    issues = []
    
    # Check directories exist
    if not os.path.exists(TRAIN_DIR):
        issues.append(f"Missing train directory: {TRAIN_DIR}")
    if not os.path.exists(TEST_DIR):
        issues.append(f"Missing test directory: {TEST_DIR}")
        
    for base_dir in [TRAIN_DIR, TEST_DIR]:
        if not os.path.exists(base_dir):
            continue
            
        # Check unexpected directories
        found_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        for d in found_dirs:
            if d not in CLASSES:
                issues.append(f"Unexpected directory found: {os.path.join(base_dir, d)}")
                
        # Check files in class directories
        for class_name in CLASSES:
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.exists(class_dir):
                issues.append(f"Missing expected class directory: {class_dir}")
                continue
                
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                if os.path.isdir(filepath):
                    continue
                    
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    issues.append(f"Unsupported file extension: {filepath}")
                    continue
                    
                # Check image validity, size, mode
                try:
                    with Image.open(filepath) as img:
                        if img.size != IMAGE_SIZE:
                            issues.append(f"Unexpected image dimensions for {filepath}: {img.size} (expected {IMAGE_SIZE})")
                        if img.mode != IMAGE_MODE:
                            issues.append(f"Unexpected image mode for {filepath}: {img.mode} (expected {IMAGE_MODE})")
                except Exception as e:
                    issues.append(f"Unreadable image: {filepath} - {str(e)}")
                    
    if issues:
        print("Data Integrity Issues Found:")
        for issue in issues[:50]:
            print(f"- {issue}")
        if len(issues) > 50:
            print(f"... and {len(issues) - 50} more issues.")
        return False
    else:
        print("Data Integrity Check Passed. No unexpected files, sizes, or modes found.")
        return True
