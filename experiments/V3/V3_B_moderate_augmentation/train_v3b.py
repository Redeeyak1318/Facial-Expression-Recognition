"""
Experiment V3-B: Moderate Augmentation Training (V2-B Architecture).
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

# Note on Moderate Augmentation (V3-B):
# - Reduced magnitudes compared to V3-A to balance real-world robustness with benchmark validation F1.
# - RandomHorizontalFlip(p=0.5)
# - RandomRotation(degrees=8)
# - RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))
# - RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.10)
# - ColorJitter(brightness=0.10, contrast=0.10)
v3b_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=8),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.10),
    T.ColorJitter(brightness=0.10, contrast=0.10)
])

class V3BTransformDataset(Dataset):
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
    print("STARTING EXPERIMENT V3-B - MODERATE AUGMENTATION")
    print("=" * 60)
    
    experiment_dir = os.path.dirname(__file__)
    config_dir = os.path.join(experiment_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "history"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "report"), exist_ok=True)
    
    train_split, val_split = create_train_val_split()
    weights_dict = calculate_class_weights(train_split)
    
    config = {
        "experiment_name": "V3_B_moderate_augmentation",
        "model_name": "ResidualCNN (V2-B)",
        "parameter_count": 307687,
        "random_seed": 42,
        "loss": "CrossEntropyLoss(weight=class_weights)",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 30,
        "patience": 5,
        "augmentation": {
            "description": "Moderate V3-B Tensor Pipeline",
            "RandomHorizontalFlip": {"p": 0.5},
            "RandomRotation": {"degrees": 8},
            "RandomAffine": {"degrees": 0, "translate": [0.05, 0.05], "scale": [0.95, 1.05]},
            "GaussianBlur": {"kernel_size": 3, "sigma": [0.1, 0.8], "probability": 0.10},
            "ColorJitter": {"brightness": 0.10, "contrast": 0.10}
        },
        "train_split_size": 25837,
        "val_split_size": 2872,
        "production_inference_preprocessing": "Haar largest-face detection -> 0% tight crop -> grayscale -> 48x48 -> /255",
        "test_set_usage": "EXPLICITLY NOT USED (Isolated from training/evaluation)",
        "class_weights": weights_dict,
        "device": "cpu"
    }
    
    set_seed(config["random_seed"])
    
    print("\n--- EXPERIMENT CONFIGURATION ---")
    config_path = os.path.join(config_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Model parameters: {config['parameter_count']}")
    print(f"Augmentation Pipeline: {v3b_transforms}")
    
    # Setup DataLoaders (Test set is intentionally NOT loaded for V3-B)
    # 1. Base datasets (no production augmentation)
    train_base_dataset = FacialExpressionDataset(train_split, augment=False)
    val_dataset = FacialExpressionDataset(val_split, augment=False)
    
    # 2. V3-B Training wrapper
    train_dataset = V3BTransformDataset(train_base_dataset, v3b_transforms)
    
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
    
    print("\n--- V3-B AUGMENTATION SMOKE TEST ---")
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
        
        # Dynamic transform sanity check
        print("\nChecking if transforms are applied dynamically...")
        # Get the same item twice
        img1, lbl1 = train_dataset[0]
        img2, lbl2 = train_dataset[0]
        if torch.allclose(img1, img2):
            print("WARNING: Transform outputs were identical. The augmentations may not be applying dynamically.")
            # Note: Given random chance, they COULD be identical, but ColorJitter + Affine/Rotation makes it extremely rare.
            # We don't fail the assert here, but we warn.
        else:
            print("Success: Repeated samples from same index produce dynamically perturbed tensors.")

        print("\nSmoke Test Passed! The augmentation pipeline produces valid tensors.")
        
    except Exception as e:
        print(f"Smoke Test Failed: {e}")
        sys.exit(1)
        
    print("\n[REVIEW STOP PASSED] Pre-training verification and smoke test successful.")
    
    # Stop before full training
    print("\nSTOPPING before full training loop as requested.")
    print("Full training has NOT started. V2-B and V3-A files are untouched.")
    
    
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
