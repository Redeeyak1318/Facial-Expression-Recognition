"""
Module for the PyTorch training loop.
"""
import torch
import json
import os
import time
from src.training.metrics import calculate_accuracy_and_macro_f1
from src.data.config import CLASS_MAPPING

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())
        
    epoch_loss = total_loss / len(dataloader.dataset)
    
    preds_tensor = torch.tensor(all_preds)
    targets_tensor = torch.tensor(all_targets)
    epoch_acc, epoch_f1 = calculate_accuracy_and_macro_f1(preds_tensor, targets_tensor, num_classes=len(CLASS_MAPPING))
    
    return epoch_loss, epoch_acc, epoch_f1

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
            
    epoch_loss = total_loss / len(dataloader.dataset)
    
    preds_tensor = torch.tensor(all_preds)
    targets_tensor = torch.tensor(all_targets)
    epoch_acc, epoch_f1 = calculate_accuracy_and_macro_f1(preds_tensor, targets_tensor, num_classes=len(CLASS_MAPPING))
    
    return epoch_loss, epoch_acc, epoch_f1

def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                max_epochs=30, patience=5, experiment_dir="experiments/B0_baseline",
                is_smoke_test=False):
    """
    Main training function handling epochs, early stopping, and checkpointing.
    """
    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    start_time = time.time()
    
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    history_dir = os.path.join(experiment_dir, "history")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    
    for epoch in range(1, max_epochs + 1):
        if is_smoke_test and epoch > 1:
            break
            
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion, device)
        
        # Log metrics
        print(f"Epoch {epoch}/{max_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}   | Val Macro F1: {val_f1:.4f}")
        
        # Save history
        epoch_history = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        }
        history.append(epoch_history)
        
        # Early Stopping & Checkpointing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            
            print(f"Best validation macro F1 improved to {best_val_f1:.4f}")
            
            if not is_smoke_test:
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_macro_f1': best_val_f1,
                    'class_mapping': CLASS_MAPPING,
                    'experiment_name': os.path.basename(experiment_dir)
                }, checkpoint_path)
                print(f"Saved best checkpoint to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} consecutive epochs.")
            
        print("-" * 60)
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
    total_time = time.time() - start_time
    
    if not is_smoke_test:
        history_path = os.path.join(history_dir, "history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Saved training history to {history_path}")
        
    return {
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "total_time_seconds": total_time,
        "history": history
    }
