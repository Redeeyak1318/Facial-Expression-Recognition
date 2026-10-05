"""
Experiment V3-A: Aggressive Augmentation Training (V2-B Architecture).
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import json
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.data.split import create_train_val_split
from src.data.dataloader import FacialExpressionDataset
from src.data.config import CLASSES, CLASS_MAPPING
from src.data.statistics import calculate_class_weights
from src.models.cnn_v2b import ResidualCNN
from src.training.loop import train_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Note on Aggressive Augmentation:
# - RandomRotation, RandomAffine, and ColorJitter are intentionally applied to EVERY training sample.
# - RandomPerspective, GaussianBlur, and RandomErasing are probabilistic.
# This is intentional because V3-A is the aggressive augmentation experiment.
v3a_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.RandomPerspective(distortion_scale=0.2, p=0.2),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
    T.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
])

class V3ATransformDataset(Dataset):
    """
    Wraps the baseline FacialExpressionDataset to dynamically apply PyTorch transforms 
    to the generated tensors during training.
    """
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.base)
        
    def __getitem__(self, idx):
        img_tensor, label = self.base[idx]
        img_tensor = self.transform(img_tensor)
        return img_tensor, label

def main():
    print("=" * 60)
    print("STARTING EXPERIMENT V3-A - AGGRESSIVE AUGMENTATION")
    print("=" * 60)
    
    experiment_dir = os.path.dirname(__file__)
    config_dir = os.path.join(experiment_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    
    train_split, val_split = create_train_val_split()
    weights_dict = calculate_class_weights(train_split)
    
    config = {
        "experiment_name": "V3_A_augmentation",
        "model_name": "ResidualCNN (V2-B)",
        "parameter_count": 307687,
        "random_seed": 42,
        "loss": "CrossEntropyLoss(weight=class_weights)",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 30,
        "patience": 5,
        "augmentation": "Aggressive V3-A Tensor Pipeline",
        "class_weights": weights_dict,
        "device": "cpu"
    }
    
    set_seed(config["random_seed"])
    
    print("\n--- EXPERIMENT CONFIGURATION ---")
    config_path = os.path.join(config_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Setup DataLoaders (Test set is intentionally NOT loaded for V3-A)
    # 1. Base datasets (no production augmentation)
    train_base_dataset = FacialExpressionDataset(train_split, augment=False)
    val_dataset = FacialExpressionDataset(val_split, augment=False)
    
    # 2. V3-A Training wrapper
    train_dataset = V3ATransformDataset(train_base_dataset, v3a_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    print("\n--- PRE-TRAINING VERIFICATION ---")
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"Train Split Size: {train_size} (Expected 25837)")
    print(f"Val Split Size: {val_size}   (Expected 2872)")
    
    assert train_size == 25837, f"Unexpected train split size: {train_size}"
    assert val_size == 2872, f"Unexpected val split size: {val_size}"
    assert val_dataset.base_dataset.augment == False, "Validation augmentation must be disabled!"
    assert train_base_dataset.base_dataset.augment == False, "Production augmentation on train base must be disabled!"
    
    print("Verification Passed: Strict isolation and sizes confirmed.")
    print("Test split is completely isolated from the training/evaluation logic.")
    
    print("\n--- V3-A AUGMENTATION SMOKE TEST ---")
    try:
        sample_batch, sample_labels = next(iter(train_loader))
        print(f"Batch Tensor Shape: {sample_batch.shape} (Expected [64, 1, 48, 48])")
        assert sample_batch.shape == (config["batch_size"], 1, 48, 48), f"Invalid batch shape: {sample_batch.shape}"
        
        min_val = sample_batch.min().item()
        max_val = sample_batch.max().item()
        has_nan = torch.isnan(sample_batch).any().item()
        has_inf = torch.isinf(sample_batch).any().item()
        
        print(f"Value Range: min={min_val:.4f}, max={max_val:.4f} (Expected ~0 to 1.x)")
        print(f"Contains NaN: {has_nan}")
        print(f"Contains Inf: {has_inf}")
        
        assert not has_nan, "NaN found in augmented tensor!"
        assert not has_inf, "Inf found in augmented tensor!"
        
        print("Smoke Test Passed! The augmentation pipeline produces valid tensors.")
        
    except Exception as e:
        print(f"Smoke Test Failed: {e}")
        sys.exit(1)
        
    print("\n[REVIEW STOP PASSED] Pre-training verification and smoke test successful.")
    print("Proceeding to full 30-epoch training loop.")
    
    # ---------------------------------------------------------
    # ACTUAL TRAINING (Disabled by sys.exit(0) above)
    # ---------------------------------------------------------
        
    device = torch.device(config["device"])
    weight_tensor = torch.tensor([weights_dict[CLASSES[i]] for i in range(len(CLASSES))], dtype=torch.float32).to(device)
    
    model = ResidualCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    print("\nStarting full training...")
    results = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=device,
        max_epochs=config["max_epochs"], 
        patience=config["patience"],
        experiment_dir=experiment_dir,
        is_smoke_test=False
    )
    
    print("\nTraining completed. Best Checkpoint saved.")

if __name__ == "__main__":
    main()
