"""
Smoke test to verify Experiment B3 (weighted + augmented) training pipeline components.
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
from src.data.config import CLASSES
from src.data.statistics import calculate_class_weights
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
    print("B3 SMOKE TEST - NOT A FINAL RESULT")
    print("=" * 60)
    
    set_seed(42)
    device = torch.device("cpu")
    print(f"Selected device: {device}")
    
    print("\n1. Data Split Generation...")
    train_split, val_split = create_train_val_split()
    test_split = get_test_paths()
    
    # Subset splits for smoke test (keep exactly 64 images from train and 64 from val, covering all classes)
    def subset_split(split_dict, target_total=64):
        subset = {}
        classes = list(split_dict.keys())
        base_count = target_total // len(classes)
        remainder = target_total % len(classes)
        
        for i, class_name in enumerate(classes):
            take_count = base_count + (1 if i < remainder else 0)
            subset[class_name] = split_dict[class_name][:take_count]
            
        return subset
        
    smoke_train_split = subset_split(train_split, target_total=64)
    smoke_val_split = subset_split(val_split, target_total=64)
    
    print("\nSmoke Train Subset Counts:")
    for c, p in smoke_train_split.items():
        print(f"  {c}: {len(p)}")
        
    print("\nSmoke Val Subset Counts:")
    for c, p in smoke_val_split.items():
        print(f"  {c}: {len(p)}")
    
    print("\n2. DataLoaders (with Augmentation for Training)...")
    # Pass smoke_val_split as a dummy for test_split to keep smoke test isolated from real test data
    # augment_train=True applies data augmentation selectively
    train_loader, val_loader, _ = get_dataloaders(smoke_train_split, smoke_val_split, smoke_val_split, batch_size=32, augment_train=True)
    
    # Check first batch
    inputs, labels = next(iter(train_loader))
    print(f"Batch input shape: {inputs.shape}")
    print(f"Batch labels shape: {labels.shape}")
    
    print("\n3. Class Weights Computation...")
    # Calculate weights on the actual train split as we would in full training
    weights_dict = calculate_class_weights(train_split)
    weight_tensor = torch.tensor([weights_dict[CLASSES[i]] for i in range(len(CLASSES))], dtype=torch.float32).to(device)
    print(f"Computed weights tensor: {weight_tensor}")
    
    print("\n4. Model, Loss, Optimizer...")
    model = LightweightCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n5. Running 1 Epoch (Smoke Test)...")
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
            is_smoke_test=True,
            experiment_dir=os.path.join("experiments", "B3_weighted_augmented")
        )
        print("\nB3 SMOKE TEST COMPLETED SUCCESSFULLY.")
    except Exception as e:
        print(f"\nB3 SMOKE TEST FAILED: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
