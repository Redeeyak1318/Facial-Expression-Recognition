"""
Script to verify the Lightweight CNN architecture and its forward pass.
"""
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from src.models.cnn import LightweightCNN
from src.data.split import create_train_val_split
from src.preprocessing.transforms import load_and_preprocess_image

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main():
    print("=" * 60)
    print("CNN ARCHITECTURE VERIFICATION")
    print("=" * 60)
    
    device = torch.device("cpu")
    print(f"Device: {device}")
    
    try:
        model = LightweightCNN(num_classes=7).to(device)
        model.eval()
        print("\nModel successfully instantiated.")
    except Exception as e:
        print(f"FAILED to instantiate model: {e}")
        sys.exit(1)
        
    print("\n" + "-" * 40)
    print("MODEL ARCHITECTURE")
    print("-" * 40)
    print(model)
    
    total_params, trainable_params = count_parameters(model)
    print("\n" + "-" * 40)
    print("PARAMETER COUNTS")
    print("-" * 40)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "-" * 40)
    print("FORWARD PASS (DUMMY TENSOR)")
    print("-" * 40)
    
    # Create dummy tensor
    dummy_input = torch.randn(1, 1, 48, 48, device=device)
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        with torch.no_grad():
            # Layer-by-layer tracking
            print("\nLayer-by-layer shape flow:")
            x = dummy_input
            print(f"  Input        : {x.shape}")
            
            x = model.block1(x)
            print(f"  After Block 1: {x.shape}")
            
            x = model.block2(x)
            print(f"  After Block 2: {x.shape}")
            
            x = model.block3(x)
            print(f"  After Block 3: {x.shape}")
            
            x = model.global_avg_pool(x)
            print(f"  After GAP    : {x.shape}")
            
            x = torch.flatten(x, 1)
            print(f"  After Flatten: {x.shape}")
            
            output = model.classifier(x)
            print(f"  Final Output : {output.shape}")
            
            # Verify correctness programmatically
            if output.shape == torch.Size([1, 7]):
                print("\nForward pass result: SUCCESS")
                print("Numerical values are finite:", torch.all(torch.isfinite(output)).item())
            else:
                print(f"\nForward pass result: FAILED - Unexpected shape {output.shape}")
                sys.exit(1)
            
    except Exception as e:
        print(f"\nForward pass result: FAILED with exception - {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DATA PIPELINE COMPATIBILITY")
    print("=" * 60)
    
    try:
        train_split, _ = create_train_val_split()
        sample_path = None
        for class_name, paths in train_split.items():
            if paths:
                sample_path = paths[0]
                break
                
        if sample_path:
            print(f"Testing real data forward pass on: {sample_path}")
            # Preprocessing outputs (48, 48, 1) numpy array
            img_array = load_and_preprocess_image(sample_path)
            
            # Convert to tensor and reshape: [H, W, C] -> [C, H, W]
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            
            # Add batch dimension: [C, H, W] -> [B, C, H, W]
            # where B=1
            img_batch = img_tensor.unsqueeze(0).to(device)
            
            print(f"Prepared batch shape: {img_batch.shape}")
            
            # Forward pass
            with torch.no_grad():
                real_output = model(img_batch)
            
            if real_output.shape == torch.Size([1, 7]):
                print("Data-pipeline compatibility result: SUCCESS")
            else:
                print(f"Data-pipeline compatibility result: FAILED - Shape {real_output.shape}")
                sys.exit(1)
        else:
            print("No images found to test pipeline compatibility.")
            
    except Exception as e:
        print(f"Data-pipeline compatibility result: FAILED with exception - {e}")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE (NO TRAINING PERFORMED)")
    print("=" * 60)

if __name__ == "__main__":
    main()
