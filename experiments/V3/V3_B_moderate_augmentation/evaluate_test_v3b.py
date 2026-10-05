"""
Script to evaluate the V2-B Residual CNN on the test set.
"""
import sys
import os
import torch
import json
import csv
import time
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
sys.path.insert(0, PROJECT_ROOT)

from src.data.split import get_test_paths, create_train_val_split, verify_no_overlap
from src.data.dataloader import get_dataloaders
from src.models.cnn_v2b import ResidualCNN
from src.data.config import CLASS_MAPPING, CLASSES, NUM_CLASSES

def evaluate():
    print("=" * 50)
    print("V3-B OFFICIAL TEST EVALUATION")
    print("=" * 50)
    
    start_time = time.time()
    
    device = torch.device("cpu")
    print(f"Device: {device}")
    
    checkpoint_path = os.path.join(
    "experiments", "V3", "V3_B_moderate_augmentation",
    "checkpoint", "best_model.pth"
    )
    if not os.path.exists(checkpoint_path):
        print(f"FAILED: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    model = ResidualCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    test_paths = get_test_paths()
    
    # Integrity Check
    test_count = sum(len(paths) for paths in test_paths.values())
    if test_count != 7178:
        print(f"FAILED INTEGRITY CHECK: Expected 7178 test images, found {test_count}")
        sys.exit(1)
        
    flat_test_paths = []
    for paths in test_paths.values():
        flat_test_paths.extend(paths)
        
    unique_test_paths = set(flat_test_paths)
    if len(unique_test_paths) != 7178:
        duplicate_count = 7178 - len(unique_test_paths)
        print(f"FAILED INTEGRITY CHECK: Found {duplicate_count} duplicate paths in test set.")
        sys.exit(1)
        
    train_split, val_split = create_train_val_split()
    try:
        verify_no_overlap(train_split, val_split, test_paths)
    except Exception as e:
        print(f"FAILED INTEGRITY CHECK (Leakage): {e}")
        sys.exit(1)
        
    print("Test-set leakage and uniqueness verified.")
        
    print(f"Test samples: {test_count}")
    
    from src.data.dataloader import FacialExpressionDataset
    from torch.utils.data import DataLoader
    test_dataset = FacialExpressionDataset(test_paths)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_paths = []
    
    # Manually extract paths for tracking
    test_flat_paths = test_loader.dataset.base_dataset.samples
    
    print("Running evaluation...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.tolist())
            
    eval_time = time.time() - start_time
    avg_inference = (eval_time / test_count) * 1000
    
    # Calculate metrics
    confusion_matrix = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
    for p, t in zip(all_preds, all_targets):
        confusion_matrix[t][p] += 1
        
    per_class_metrics = {}
    correct = 0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    
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
        
    accuracy = correct / test_count
    macro_precision /= NUM_CLASSES
    macro_recall /= NUM_CLASSES
    macro_f1 /= NUM_CLASSES
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    print("\nPer-class results:")
    print(f"{'Class':<12} {'Precision':<11} {'Recall':<9} {'F1':<5} {'Support'}")
    print("-" * 48)
    for c in range(NUM_CLASSES):
        class_name = CLASSES[c]
        m = per_class_metrics[class_name]
        print(f"{class_name:<12} {m['precision']:<11.4f} {m['recall']:<9.4f} {m['f1']:<5.4f} {m['support']}")
        
    print("\nConfusion Matrix:")
    header = "          " + " ".join([f"{c[:3]:>4}" for c in CLASSES])
    print(header)
    for i, row in enumerate(confusion_matrix):
        row_str = " ".join([f"{val:>4}" for val in row])
        print(f"{CLASSES[i]:<10} {row_str}")
        
    print(f"\nBest V3-B checkpoint: {checkpoint_path}")
    print(f"Total evaluation time: {eval_time:.2f} seconds ({avg_inference:.2f} ms/image)")
    
    # Save outputs
    eval_dir = os.path.join(
    "experiments", "V3", "V3_B_moderate_augmentation", "evaluation"
    )
    os.makedirs(eval_dir, exist_ok=True)
    
    # 1. metrics.json
    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "total_test_samples": test_count,
        "evaluation_time_seconds": eval_time
    }
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # 2. confusion_matrix.json
    with open(os.path.join(eval_dir, "confusion_matrix.json"), "w") as f:
        json.dump({"matrix": confusion_matrix, "classes": CLASSES}, f, indent=4)
        
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(confusion_matrix, cmap='Blues')
        fig.colorbar(cax)
        
        ax.set_xticks(np.arange(len(CLASSES)))
        ax.set_yticks(np.arange(len(CLASSES)))
        ax.set_xticklabels(CLASSES)
        ax.set_yticklabels(CLASSES)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                text = ax.text(j, i, confusion_matrix[i][j],
                               ha="center", va="center", color="black")
                               
        plt.title('V3-B Residual CNN Confusion Matrix', pad=20)
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"))
        plt.close()
    except ImportError:
        print("\nmatplotlib not installed. Skipping visual confusion_matrix.png generation.")
        
    # 3. predictions.csv & error analysis
    errors = []
    with open(os.path.join(eval_dir, "predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "predicted_label"])
        
        for i in range(test_count):
            path = test_flat_paths[i][0]
            true_label = CLASS_MAPPING[all_targets[i]]
            pred_label = CLASS_MAPPING[all_preds[i]]
            writer.writerow([path, true_label, pred_label])
            
            if true_label != pred_label:
                errors.append({
                    "image_path": path,
                    "true_class": true_label,
                    "predicted_class": pred_label
                })
                
    # 4. Error analysis file
    with open(os.path.join(eval_dir, "error_analysis.json"), "w") as f:
        json.dump(errors, f, indent=4)
        
    print(f"\nSaved evaluation outputs to {eval_dir}")
    print(f"Number of incorrect predictions: {len(errors)}")

if __name__ == "__main__":
    evaluate()
