"""
Script to verify PyTorch installation and data pipeline compatibility.
"""
import sys
import os
import platform

# Add the project root to sys.path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from src.data.split import create_train_val_split
from src.preprocessing.transforms import load_and_preprocess_image

def main():
    print("=" * 60)
    print("PYTORCH ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    # 1. Environment Details
    print(f"Python version: {sys.version.split()[0]}")
    print(f"OS: {platform.system()} {platform.release()}")
    
    # 2. PyTorch Details
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torch availability: {bool(torch)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    print(f"CPU availability: True") # CPU is always available
    print(f"CUDA availability: {torch.cuda.is_available()}")
    
    if str(device) != "cpu":
        print("WARNING: Expected 'cpu' device for this verification, but got something else.")
        
    print("\n" + "=" * 60)
    print("TENSOR OPERATION VERIFICATION")
    print("=" * 60)
    
    try:
        # Create a small random tensor
        x = torch.rand((3, 3), device=device)
        y = torch.rand((3, 3), device=device)
        z = torch.matmul(x, y)
        print("Small random tensor multiplication:")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"Result z shape: {z.shape}")
        print("Tensor test result: SUCCESS")
    except Exception as e:
        print(f"Tensor test result: FAILED - {e}")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("DATA PIPELINE TENSOR COMPATIBILITY")
    print("=" * 60)
    
    try:
        train_split, _ = create_train_val_split()
        sample_path = None
        for class_name, paths in train_split.items():
            if paths:
                sample_path = paths[0]
                break
                
        if sample_path:
            print(f"Testing tensor conversion on: {sample_path}")
            # Preprocessing outputs (48, 48, 1) numpy array
            img_array = load_and_preprocess_image(sample_path)
            
            # PyTorch expects Channels-First format for CNNs: [C, H, W] -> [1, 48, 48]
            # Numpy array is [H, W, C] -> [48, 48, 1]
            img_tensor = torch.from_numpy(img_array)
            
            # Rearrange dimensions to [C, H, W]
            img_tensor_chw = img_tensor.permute(2, 0, 1)
            
            print(f"Original Numpy shape: {img_array.shape}")
            print(f"Converted PyTorch Tensor shape: {img_tensor_chw.shape}")
            
            if img_tensor_chw.shape == (1, 48, 48):
                print("Data-pipeline tensor compatibility result: SUCCESS")
            else:
                print("Data-pipeline tensor compatibility result: FAILED (Unexpected shape)")
                sys.exit(1)
        else:
            print("No images found to test compatibility.")
            
    except Exception as e:
        print(f"Data-pipeline tensor compatibility result: FAILED - {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
