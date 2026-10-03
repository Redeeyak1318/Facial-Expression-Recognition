"""
Module for discovering dataset files and creating train/validation splits.
"""
import os
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from src.data.config import TRAIN_DIR, TEST_DIR, CLASSES, VALIDATION_RATIO, RANDOM_SEED

def get_image_paths(directory: str) -> Dict[str, List[str]]:
    """
    Discovers all image paths in the given directory and groups them by class.
    Assumes the directory contains subdirectories named after each class.
    """
    class_paths = defaultdict(list)
    for class_name in CLASSES:
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                class_paths[class_name].append(os.path.join(class_dir, filename))
    return dict(class_paths)

def create_train_val_split() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Creates a stratified train/validation split using the fixed random seed.
    Returns: (train_paths_by_class, val_paths_by_class)
    """
    # Use fixed random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    original_train_paths = get_image_paths(TRAIN_DIR)
    
    train_split = {}
    val_split = {}
    
    for class_name, paths in original_train_paths.items():
        # Sort paths first to ensure reproducible shuffling across different OS/environments
        sorted_paths = sorted(paths)
        random.shuffle(sorted_paths)
        
        # Calculate split index
        val_count = int(round(len(sorted_paths) * VALIDATION_RATIO))
        
        val_split[class_name] = sorted_paths[:val_count]
        train_split[class_name] = sorted_paths[val_count:]
        
    return train_split, val_split

def get_test_paths() -> Dict[str, List[str]]:
    """
    Returns the test paths grouped by class. The test set remains completely untouched.
    """
    return get_image_paths(TEST_DIR)

def verify_no_overlap(train_split: Dict[str, List[str]], 
                      val_split: Dict[str, List[str]], 
                      test_split: Dict[str, List[str]]):
    """
    Explicitly checks for any overlap between train, validation, and test splits.
    Raises ValueError if overlap is detected.
    """
    train_set = set()
    for paths in train_split.values():
        train_set.update(paths)
        
    val_set = set()
    for paths in val_split.values():
        val_set.update(paths)
        
    test_set = set()
    for paths in test_split.values():
        test_set.update(paths)
        
    train_val_overlap = train_set.intersection(val_set)
    if train_val_overlap:
        raise ValueError(f"DATA LEAKAGE DETECTED: {len(train_val_overlap)} images in both train and validation sets.")
        
    train_test_overlap = train_set.intersection(test_set)
    if train_test_overlap:
        raise ValueError(f"DATA LEAKAGE DETECTED: {len(train_test_overlap)} images in both train and test sets.")
        
    val_test_overlap = val_set.intersection(test_set)
    if val_test_overlap:
        raise ValueError(f"DATA LEAKAGE DETECTED: {len(val_test_overlap)} images in both validation and test sets.")
