"""
Smoke test to verify training pipeline components.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.split import create_train_val_split, get_test_paths
from src.data.dataloader import get_dataloaders
from src.models.cnn import LightweightCNN
from src.training.loop import train_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("=" * 60)
    print("SMOKE TEST - NOT A FINAL RESULT")
    print("=" * 60)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    
    print("\n1. Data Split Generation...")
    train_split, val_split = create_train_val_split()
    test_split = get_test_paths()
    
    # Subset splits for smoke test (keep only 128 images from train and 128 from val)
    def subset_split(split_dict, max_total=128):
        subset = {}
        count = 0
        for class_name, paths in split_dict.items():
            for path in paths:
                if class_name not in subset:
                    subset[class_name] = []
                subset[class_name].append(path)
                count += 1
                if count >= max_total:
                    return subset
        return subset
        
    smoke_train_split = subset_split(train_split, max_total=64)
    smoke_val_split = subset_split(val_split, max_total=64)
    
    print("\n2. DataLoaders...")
    # Pass smoke_val_split as a dummy for test_split to keep smoke test isolated from real test data
    train_loader, val_loader, _ = get_dataloaders(smoke_train_split, smoke_val_split, smoke_val_split, batch_size=32)
    
    # Check first batch
    inputs, labels = next(iter(train_loader))
    print(f"Batch input shape: {inputs.shape}")
    print(f"Batch labels shape: {labels.shape}")
    
    print("\n3. Model, Loss, Optimizer...")
    model = LightweightCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n4. Running 1 Epoch (Smoke Test)...")
    try:
        results = train_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device,
            max_epochs=1, 
            patience=5,
            is_smoke_test=True
        )
        print("\nSMOKE TEST COMPLETED SUCCESSFULLY.")
    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
