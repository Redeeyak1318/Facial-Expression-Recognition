"""
Module for calculating dataset statistics and class weights.
"""
from typing import Dict, List
from src.data.config import CLASSES, NUM_CLASSES
import math

def calculate_class_weights(train_split: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculates balanced class weights using only the training split.
    Weight for class c = total_train_samples / (num_classes * samples_in_class_c)
    """
    total_train = sum(len(paths) for paths in train_split.values())
    weights = {}
    for class_name, paths in train_split.items():
        count = len(paths)
        if count > 0:
            weight = total_train / (NUM_CLASSES * count)
        else:
            weight = 0.0
        weights[class_name] = weight
    return weights

def print_dataset_statistics(train_split: Dict[str, List[str]], 
                             val_split: Dict[str, List[str]], 
                             test_split: Dict[str, List[str]]):
    """
    Prints a formatted table of dataset statistics.
    """
    print(f"\nTotal Classes: {NUM_CLASSES}")
    print(f"Classes: {', '.join(CLASSES)}\n")
    
    print(f"{'Class':<12} | {'Train':<8} | {'Validation':<12} | {'Test':<8} | {'Total':<8}")
    print("-" * 59)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for class_name in CLASSES:
        train_count = len(train_split.get(class_name, []))
        val_count = len(val_split.get(class_name, []))
        test_count = len(test_split.get(class_name, []))
        
        total = train_count + val_count + test_count
        
        total_train += train_count
        total_val += val_count
        total_test += test_count
        
        print(f"{class_name:<12} | {train_count:<8} | {val_count:<12} | {test_count:<8} | {total:<8}")
        
    print("-" * 59)
    total_all = total_train + total_val + total_test
    print(f"{'Total':<12} | {total_train:<8} | {total_val:<12} | {total_test:<8} | {total_all:<8}\n")
    
    print("Class Distribution (Original Train Set):")
    original_train_total = total_train + total_val
    for class_name in CLASSES:
        orig_count = len(train_split.get(class_name, [])) + len(val_split.get(class_name, []))
        pct = (orig_count / original_train_total) * 100 if original_train_total > 0 else 0
        print(f"  {class_name:<10}: {pct:.2f}%")
        
    print("\nClass Weights (Calculated from Train Split Only):")
    weights = calculate_class_weights(train_split)
    for class_name in CLASSES:
        print(f"  {class_name:<10}: {weights.get(class_name, 0.0):.4f}")
