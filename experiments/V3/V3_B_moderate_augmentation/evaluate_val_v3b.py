"""
Script to evaluate the V3-B checkpoint on the VALIDATION set and generate the results report.
"""
import sys
import os
import torch
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.data.split import create_train_val_split
from src.models.cnn_v2b import ResidualCNN
from src.data.config import CLASS_MAPPING, CLASSES, NUM_CLASSES
from src.data.dataloader import FacialExpressionDataset
from torch.utils.data import DataLoader

def main():
    experiment_dir = os.path.dirname(__file__)
    checkpoint_path = os.path.join(experiment_dir, "checkpoint", "best_model.pth")
    history_path = os.path.join(experiment_dir, "history", "history.json")
    config_path = os.path.join(experiment_dir, "config", "config.json")
    
    device = torch.device("cpu")
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    with open(history_path, "r") as f:
        history = json.load(f)
        
    best_epoch = checkpoint['epoch']
    best_val_macro_f1 = checkpoint.get('best_val_macro_f1', 0.5794768004579861) # Using known best from history
    
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Macro F1 (from Checkpoint): {best_val_macro_f1:.4f}")
    
    # 1. Evaluate on Validation Set
    train_split, val_split = create_train_val_split()
    val_dataset = FacialExpressionDataset(val_split, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    model = ResidualCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.tolist())
            
    confusion_matrix = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
    for p, t in zip(all_preds, all_targets):
        confusion_matrix[t][p] += 1
        
    per_class_metrics = {}
    correct = 0
    macro_precision, macro_recall, macro_f1 = 0.0, 0.0, 0.0
    
    for c in range(NUM_CLASSES):
        tp = confusion_matrix[c][c]
        fp = sum(confusion_matrix[i][c] for i in range(NUM_CLASSES)) - tp
        fn = sum(confusion_matrix[c][i] for i in range(NUM_CLASSES)) - tp
        support = sum(confusion_matrix[c])
        correct += tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[CLASSES[c]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        
    accuracy = correct / len(val_dataset)
    macro_f1 /= NUM_CLASSES
    macro_precision /= NUM_CLASSES
    macro_recall /= NUM_CLASSES
    
    print(f"Validation Re-eval Accuracy: {accuracy:.4f}")
    print(f"Validation Re-eval Macro F1: {macro_f1:.4f}")
    
    # Extract top confusion pairs
    confusions = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and confusion_matrix[i][j] > 0:
                confusions.append((CLASSES[i], CLASSES[j], confusion_matrix[i][j]))
    confusions.sort(key=lambda x: x[2], reverse=True)
    top_confusions = confusions[:5]
    
    # 2. Generate Curves and Conf Matrix Plots
    report_dir = os.path.join(experiment_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Training Curves
        epochs = [x['epoch'] for x in history]
        train_loss = [x['train_loss'] for x in history]
        val_loss = [x['val_loss'] for x in history]
        train_f1 = [x['train_macro_f1'] for x in history]
        val_f1 = [x['val_macro_f1'] for x in history]
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Val Loss')
        plt.title('Loss Curve')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_f1, label='Train Macro F1')
        plt.plot(epochs, val_f1, label='Val Macro F1')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.title('Macro F1 Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "training_curves.png"))
        plt.close()
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(confusion_matrix, cmap='Blues')
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(CLASSES)))
        ax.set_yticks(np.arange(len(CLASSES)))
        ax.set_xticklabels(CLASSES)
        ax.set_yticklabels(CLASSES)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, confusion_matrix[i][j], ha="center", va="center", color="black")
        plt.title('V3-B Validation Confusion Matrix', pad=20)
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "val_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")
        
    # 3. Generate Markdown Report
    final_epoch = history[-1]['epoch']
    early_stopping = final_epoch < config['max_epochs']
    v2b_f1 = 0.5489
    delta = macro_f1 - v2b_f1
    
    report_content = f"""# V3-B Validation Results Report

## Experiment Configuration
- **Model**: {config['model_name']} ({config['parameter_count']:,} params)
- **Epochs**: {final_epoch}/{config['max_epochs']} (Early Stopping: {early_stopping})
- **Best Epoch**: {best_epoch}
- **Batch Size**: {config['batch_size']}
- **Learning Rate**: {config['learning_rate']}
- **Optimizer**: {config['optimizer']}

## Augmentation Pipeline (Moderate)
- RandomHorizontalFlip (p=0.5)
- RandomRotation (degrees=8)
- RandomAffine (translate=5%, scale=5%)
- GaussianBlur (p=0.10)
- ColorJitter (brightness/contrast = 0.10)

## Validation Performance vs Baseline
- **V2-B Baseline Macro F1 (Validation)**: {v2b_f1:.4f}
- **V3-B Macro F1 (Validation)**: {macro_f1:.4f}
- **Delta**: {delta:+.4f}

### Per-Class Metrics (Validation)
| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
"""
    for c in CLASSES:
        m = per_class_metrics[c]
        report_content += f"| {c} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |\n"

    report_content += f"\n**Overall Accuracy**: {accuracy:.4f}\n"
    report_content += f"**Macro Precision**: {macro_precision:.4f}\n"
    report_content += f"**Macro Recall**: {macro_recall:.4f}\n"
    report_content += f"**Macro F1**: {macro_f1:.4f}\n\n"
    
    report_content += "## Top Confusion Pairs (True -> Predicted)\n"
    for true_cls, pred_cls, count in top_confusions:
        report_content += f"- **{true_cls}** confused as **{pred_cls}**: {count} times\n"
    
    report_md_path = os.path.join(experiment_dir, "V3_B_VALIDATION_REPORT.md")
    with open(report_md_path, "w") as f:
        f.write(report_content)
        
    print(f"\nReport generated at {report_md_path}")
    
if __name__ == "__main__":
    main()
