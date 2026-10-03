"""
Script to verify the complete data pipeline functionality.
"""
import sys
import os
import argparse

# Add the project root to sys.path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.split import create_train_val_split, get_test_paths, verify_no_overlap
from src.data.statistics import print_dataset_statistics
from src.data.integrity import check_dataset_integrity
from src.data.config import IMAGE_SIZE, IMAGE_MODE, NORMALIZATION_RULE, CLASS_MAPPING, CLASSES
from src.preprocessing.transforms import load_and_preprocess_image

def verify_class_mapping() -> bool:
    """
    Explicitly verify the exact class mapping requirements.
    0 = angry, 1 = disgust, 2 = fear, 3 = happy, 4 = neutral, 5 = sad, 6 = surprise
    """
    expected_mapping = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise"
    }
    
    print("\nVERIFYING CLASS MAPPING...")
    failed = False
    if len(CLASS_MAPPING) != 7:
        print(f"FAILED: Expected exactly 7 classes, found {len(CLASS_MAPPING)}.")
        failed = True
        
    for key, expected_val in expected_mapping.items():
        if CLASS_MAPPING.get(key) != expected_val:
            print(f"FAILED: Class mapping mismatch at index {key}. Expected '{expected_val}', got '{CLASS_MAPPING.get(key)}'.")
            failed = True
            
    if not failed:
        print("Success: Class mapping is exactly as required.")
        return True
    return False

def verify_dataset_invariants(train_split, val_split, test_split) -> bool:
    """
    Explicit assertions for the known dataset invariants:
    - original training count = 28709
    - original test count = 7178
    - train + validation = 28709
    - test = 7178
    - exactly 7 expected classes exist
    """
    print("\nVERIFYING DATASET INVARIANTS...")
    
    total_train = sum(len(paths) for paths in train_split.values())
    total_val = sum(len(paths) for paths in val_split.values())
    total_test = sum(len(paths) for paths in test_split.values())
    
    train_plus_val = total_train + total_val
    
    failed = False
    
    if train_plus_val != 28709:
        print(f"FAILED: Expected train + validation = 28709. Actual: {train_plus_val}")
        failed = True
    else:
        print("Success: train + validation = 28709")
        
    if total_test != 7178:
        print(f"FAILED: Expected original test count = 7178. Actual: {total_test}")
        failed = True
    else:
        print("Success: test count = 7178")
        
    if len(CLASSES) != 7:
        print(f"FAILED: Expected exactly 7 classes. Actual: {len(CLASSES)}")
        failed = True
    else:
        print("Success: Exactly 7 expected classes exist")
        
    if failed:
        print("FAILED: One or more dataset invariants failed.")
        return False
    else:
        print("Success: All dataset invariants passed.")
        return True

def main():
    parser = argparse.ArgumentParser(description="Verify Facial Expression Recognition Data Pipeline.")
    parser.add_argument("--full-integrity", action="store_true", help="Run the expensive complete dataset integrity scan.")
    args = parser.parse_args()

    print("=" * 60)
    print("DATA PIPELINE VERIFICATION")
    print("=" * 60)
    
    print("\n1. CREATING SPLITS...")
    train_split, val_split = create_train_val_split()
    test_split = get_test_paths()
    
    if not verify_class_mapping():
        print("\nCRITICAL FAILURE: Class mapping verification failed. Stopping verification.")
        sys.exit(1)
        
    if not verify_dataset_invariants(train_split, val_split, test_split):
        print("\nCRITICAL FAILURE: Dataset invariants verification failed. Stopping verification.")
        sys.exit(1)
    
    print("\n2. VERIFYING NO LEAKAGE / OVERLAP...")
    try:
        verify_no_overlap(train_split, val_split, test_split)
        print("Success: No overlap detected between train, validation, and test sets.")
    except ValueError as e:
        print(f"FAILED: {e}")
        print("\nCRITICAL FAILURE: Data leakage detected. Stopping verification.")
        sys.exit(1)
        
    print("\n3. DATASET STATISTICS...")
    print_dataset_statistics(train_split, val_split, test_split)
    
    print("\n4. VERIFYING REPRODUCIBILITY...")
    train_split_2, val_split_2 = create_train_val_split()
    
    # Check if paths in train_split and train_split_2 are identical
    is_reproducible = True
    for class_name in train_split:
        if train_split[class_name] != train_split_2[class_name]:
            is_reproducible = False
            break
            
    if is_reproducible:
        print("Success: Split is reproducible across multiple calls (fixed seed used).")
    else:
        print("FAILED: Split is not reproducible.")
        print("\nCRITICAL FAILURE: Reproducibility check failed. Stopping verification.")
        sys.exit(1)
        
    print("\n5. SAMPLE PREPROCESSING VERIFICATION...")
    # Find a sample image
    sample_path = None
    for class_name, paths in train_split.items():
        if paths:
            sample_path = paths[0]
            break
            
    if sample_path:
        print(f"Testing preprocessing on: {sample_path}")
        try:
            img_array = load_and_preprocess_image(sample_path)
            print(f"Shape: {img_array.shape} (Expected: {IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}, 1)")
            print(f"Min value: {img_array.min():.4f} (Expected: >= 0.0)")
            print(f"Max value: {img_array.max():.4f} (Expected: <= 1.0)")
            
            if img_array.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1], 1) and img_array.min() >= 0.0 and img_array.max() <= 1.0:
                print("Success: Preprocessing works as expected.")
            else:
                print("FAILED: Preprocessing output is not as expected.")
                print("\nCRITICAL FAILURE: Sample preprocessing verification failed. Stopping verification.")
                sys.exit(1)
        except Exception as e:
            print(f"FAILED: Preprocessing threw an exception: {e}")
            print("\nCRITICAL FAILURE: Sample preprocessing verification failed. Stopping verification.")
            sys.exit(1)
    else:
        print("Warning: Could not find any images to test preprocessing.")
        
    print("\n6. INTEGRITY VALIDATION...")
    if args.full_integrity:
        print("Running full integrity validation. This may take a few seconds...")
        if not check_dataset_integrity():
            print("FAILED: Full integrity validation failed.")
            print("\nCRITICAL FAILURE: Integrity check failed. Stopping verification.")
            sys.exit(1)
    else:
        print("SKIPPED: Full integrity scan was not run. Use --full-integrity to run it.")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
