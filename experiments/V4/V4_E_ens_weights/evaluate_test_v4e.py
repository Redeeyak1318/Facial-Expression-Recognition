"""
Evaluation script for V4-E: Effective Number of Samples (ENS) Weighting.
Evaluates the best V4-E checkpoint on the untouched official test set.
Compares results against the frozen V3-B test baseline.
"""
import sys
import os
import torch
import json
import csv
import time

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data.split import get_test_paths, create_train_val_split, verify_no_overlap
from src.data.dataloader import FacialExpressionDataset
from torch.utils.data import DataLoader
from src.models.cnn_v2b import ResidualCNN
from src.data.config import CLASS_MAPPING, CLASSES, NUM_CLASSES

# V3-B Official Test Results for direct comparison
V3B_TEST_METRICS = {
    "accuracy": 0.5916689885762051,
    "macro_precision": 0.5489445513393556,
    "macro_recall": 0.5910789429553329,
    "macro_f1": 0.5575540321837488,
    "per_class": {
        "angry": {"precision": 0.495403, "recall": 0.506263, "f1": 0.500774, "support": 958},
        "disgust": {"precision": 0.384615, "recall": 0.675676, "f1": 0.490196, "support": 111},
        "fear": {"precision": 0.469534, "recall": 0.255859, "f1": 0.331226, "support": 1024},
        "happy": {"precision": 0.858838, "recall": 0.775085, "f1": 0.814815, "support": 1774},
        "neutral": {"precision": 0.537023, "recall": 0.582320, "f1": 0.558755, "support": 1233},
        "sad": {"precision": 0.443609, "recall": 0.520449, "f1": 0.478967, "support": 1247},
        "surprise": {"precision": 0.653589, "recall": 0.821901, "f1": 0.728145, "support": 831}
    }
}

def evaluate():
    print("=" * 65)
    print("V4-E OFFICIAL TEST EVALUATION (ENS WEIGHTING)")
    print("=" * 65)

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = os.path.join(SCRIPT_DIR, "checkpoint", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"FAILED: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    best_val_macro_f1 = checkpoint.get("best_val_macro_f1", "N/A")
    best_epoch = checkpoint.get("epoch", "N/A")
    print(f"Checkpoint info - Best Epoch: {best_epoch}, Best Val Macro F1: {best_val_macro_f1}")

    model = ResidualCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # ── Test Set Integrity Verification ───────────────────────────────
    print("\n--- Verifying Test Set Integrity ---")
    test_paths = get_test_paths()
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
        print(f"FAILED INTEGRITY CHECK (Data Leakage): {e}")
        sys.exit(1)

    print(f"Integrity confirmed: {test_count} unique test samples, zero leakage with train/val.")

    # ── Test DataLoader ───────────────────────────────────────────────
    test_dataset = FacialExpressionDataset(test_paths, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_targets = []
    test_flat_samples = test_dataset.base_dataset.samples

    print("\nRunning inference on official test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.tolist())

    eval_time = time.time() - start_time
    avg_inference = (eval_time / test_count) * 1000

    # ── Metrics Calculation (Manual Confusion Matrix) ─────────────────
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

    # ── Results Output ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("V4-E OFFICIAL TEST RESULTS")
    print("=" * 65)
    print(f"Test Accuracy:    {accuracy:.4f} ({correct}/{test_count})")
    print(f"Macro Precision:  {macro_precision:.4f}")
    print(f"Macro Recall:     {macro_recall:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")

    print("\n--- Per-Class Performance ---")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support'}")
    print("-" * 56)
    for c in range(NUM_CLASSES):
        cls = CLASSES[c]
        m = per_class_metrics[cls]
        print(f"{cls:<12} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']}")

    print("\n--- Confusion Matrix (Rows: True, Cols: Pred) ---")
    header = "          " + " ".join([f"{c[:3]:>5}" for c in CLASSES])
    print(header)
    for i, row in enumerate(confusion_matrix):
        row_str = " ".join([f"{val:>5}" for val in row])
        print(f"{CLASSES[i]:<10} {row_str}")

    # ── Comparison with V3-B ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("COMPARISON: V4-E (ENS Weights) vs V3-B Baseline")
    print("=" * 65)
    diff_acc = accuracy - V3B_TEST_METRICS["accuracy"]
    diff_mp = macro_precision - V3B_TEST_METRICS["macro_precision"]
    diff_mr = macro_recall - V3B_TEST_METRICS["macro_recall"]
    diff_mf1 = macro_f1 - V3B_TEST_METRICS["macro_f1"]

    print(f"{'Metric':<18} | {'V3-B':<10} | {'V4-E':<10} | {'Delta':<10}")
    print("-" * 56)
    print(f"{'Accuracy':<18} | {V3B_TEST_METRICS['accuracy']:<10.4f} | {accuracy:<10.4f} | {diff_acc:+10.4f}")
    print(f"{'Macro Precision':<18} | {V3B_TEST_METRICS['macro_precision']:<10.4f} | {macro_precision:<10.4f} | {diff_mp:+10.4f}")
    print(f"{'Macro Recall':<18} | {V3B_TEST_METRICS['macro_recall']:<10.4f} | {macro_recall:<10.4f} | {diff_mr:+10.4f}")
    print(f"{'Macro F1':<18} | {V3B_TEST_METRICS['macro_f1']:<10.4f} | {macro_f1:<10.4f} | {diff_mf1:+10.4f}")

    print("\n--- Per-Class F1 Comparison ---")
    print(f"{'Class':<12} | {'V3-B F1':<10} | {'V4-E F1':<10} | {'Delta F1':<10}")
    print("-" * 50)
    for cls in CLASSES:
        v3b_f1 = V3B_TEST_METRICS["per_class"][cls]["f1"]
        v4e_f1 = per_class_metrics[cls]["f1"]
        delta_f1 = v4e_f1 - v3b_f1
        print(f"{cls:<12} | {v3b_f1:<10.4f} | {v4e_f1:<10.4f} | {delta_f1:+10.4f}")

    print(f"\nTotal evaluation time: {eval_time:.2f} seconds ({avg_inference:.2f} ms/image)")

    # ── Save Outputs ──────────────────────────────────────────────────
    eval_dir = os.path.join(SCRIPT_DIR, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # 1. metrics.json
    metrics_data = {
        "experiment_name": "V4_E_ens_weights",
        "checkpoint_evaluated": checkpoint_path,
        "best_epoch": best_epoch,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "comparison_v3b": {
            "v3b_accuracy": V3B_TEST_METRICS["accuracy"],
            "v3b_macro_f1": V3B_TEST_METRICS["macro_f1"],
            "delta_accuracy": diff_acc,
            "delta_macro_f1": diff_mf1
        },
        "total_test_samples": test_count,
        "evaluation_time_seconds": eval_time
    }
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)

    # 2. confusion_matrix.json
    with open(os.path.join(eval_dir, "confusion_matrix.json"), "w") as f:
        json.dump({"matrix": confusion_matrix, "classes": CLASSES}, f, indent=4)

    # 3. Optional visual confusion matrix
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

        plt.title('V4-E Residual CNN (ENS Weights) Confusion Matrix', pad=20)
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Skipping plot generation: {e}")

    # 4. predictions.csv and error analysis
    errors = []
    with open(os.path.join(eval_dir, "predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "predicted_label"])

        for i in range(test_count):
            path = test_flat_samples[i][0]
            true_label = CLASS_MAPPING[all_targets[i]]
            pred_label = CLASS_MAPPING[all_preds[i]]
            writer.writerow([path, true_label, pred_label])

            if true_label != pred_label:
                errors.append({
                    "image_path": path,
                    "true_class": true_label,
                    "predicted_class": pred_label
                })

    with open(os.path.join(eval_dir, "error_analysis.json"), "w") as f:
        json.dump(errors, f, indent=4)

    print(f"\nEvaluation outputs successfully saved to: {eval_dir}")
    print(f"Total misclassifications: {len(errors)} / {test_count}")
    print("=" * 65)

if __name__ == "__main__":
    evaluate()
