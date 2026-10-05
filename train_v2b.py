"""
Experiment V2-B: Lightweight Residual CNN Training.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.split import create_train_val_split, get_test_paths
from src.data.dataloader import get_dataloaders
from src.data.config import CLASSES
from src.data.statistics import calculate_class_weights
from src.models.cnn_v2b import ResidualCNN
from src.training.loop import train_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("=" * 60)
    print("STARTING EXPERIMENT V2-B - RESIDUAL CNN")
    print("=" * 60)
    
    experiment_dir = os.path.join("experiments", "V2", "V2_B_residual_cnn")
    config_dir = os.path.join(experiment_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    
    print("\n1. Data Split Generation...")
    train_split, val_split = create_train_val_split()
    test_split = get_test_paths()
    
    # Calculate class weights from training split only
    weights_dict = calculate_class_weights(train_split)
    
    # 1. Configuration
    config = {
        "experiment_name": "V2_B_residual_cnn",
        "model_name": "ResidualCNN",
        "parameter_count": 307687,
        "random_seed": 42,
        "loss": "CrossEntropyLoss(weight=class_weights)",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 30,
        "patience": 5,
        "augmentation": "None",
        "class_weights": weights_dict,
        "device": "cpu"
    }
    
    config_path = os.path.join(config_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    set_seed(config["random_seed"])
    device = torch.device("cpu")
    print(f"Configured Device: {device}")
    
    print("\nClass weights computed from train split:")
    for c_name, w in weights_dict.items():
        print(f"  {c_name}: {w:.4f}")
        
    # Convert weights to tensor
    weight_tensor = torch.tensor([weights_dict[CLASSES[i]] for i in range(len(CLASSES))], dtype=torch.float32).to(device)
    
    print("\n2. DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        train_split, val_split, test_split, batch_size=config["batch_size"], augment_train=False
    )
    print(f"Training samples:   {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    print("\n3. Model, Loss, Optimizer...")
    model = ResidualCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameter Count: {total_params:,}")

    print("\n4. Running Full V2-B Training...")
    try:
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
        
        print("\n" + "=" * 60)
        print("EXPERIMENT V2-B RESULTS")
        print("=" * 60)
        print(f"Experiment Name: {config['experiment_name']}")
        print(f"Actual class weights used:")
        for c_name, w in weights_dict.items():
            print(f"  {c_name}: {w:.4f}")
        print(f"Parameter count: {total_params:,}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Epochs completed: {results['epochs_completed']}")
        print(f"Best epoch: {results['best_epoch']}")
        print(f"Best Val Macro F1: {results['best_val_macro_f1']:.4f}")
        
        # Get metrics at best epoch from history
        best_history = next(item for item in results['history'] if item["epoch"] == results['best_epoch'])
        print(f"Val Accuracy at best epoch: {best_history['val_accuracy']:.4f}")
        print(f"Val Loss at best epoch: {best_history['val_loss']:.4f}")
        print(f"Total training time: {results['total_time_seconds']:.2f} seconds")
        print(f"Best checkpoint path: {os.path.join(experiment_dir, 'checkpoint', 'best_model.pth')}")
        print(f"History path: {os.path.join(experiment_dir, 'history', 'history.json')}")
        
    except Exception as e:
        print(f"\nEXPERIMENT V2-B FAILED: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
