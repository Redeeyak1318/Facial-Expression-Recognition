"""
Experiment V4-E: Effective Number of Samples (ENS) Weighting
Architecture: Frozen V3-B ResidualCNN
Augmentation: V3-B moderate pipeline
Loss: CrossEntropyLoss with ENS weights (β=0.999)

Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
  E_n = (1 - β^n) / (1 - β)
  weight_c = 1 / E_n_c
  Then normalize so weights sum to num_classes.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data.split import create_train_val_split
from src.data.dataloader import FacialExpressionDataset
from src.data.config import CLASSES, CLASS_MAPPING, NUM_CLASSES
from src.models.cnn_v2b import ResidualCNN
from src.training.loop import train_model

# ── Seed ──────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── ENS weight calculation ────────────────────────────────────────────
def calculate_ens_weights(train_split, beta=0.999):
    """
    Effective Number of Samples weighting.
    weight_c = (1 - β) / (1 - β^n_c)
    Normalized so weights sum to num_classes.
    """
    counts = {}
    for class_name in CLASSES:
        counts[class_name] = len(train_split.get(class_name, []))

    raw_weights = {}
    for class_name in CLASSES:
        n = counts[class_name]
        effective_num = (1.0 - beta ** n) / (1.0 - beta)
        raw_weights[class_name] = 1.0 / effective_num

    # Normalize so weights sum to num_classes
    total = sum(raw_weights.values())
    ens_weights = {}
    for class_name in CLASSES:
        ens_weights[class_name] = (raw_weights[class_name] / total) * NUM_CLASSES

    return ens_weights, counts

# ── V3-B Augmentation (identical to V3-B) ─────────────────────────────
v3b_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=8),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.10),
    T.ColorJitter(brightness=0.10, contrast=0.10)
])

class V3BTransformDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_tensor, label = self.base[idx]
        img_tensor = self.transform(img_tensor)
        return img_tensor, label

# ── Per-class F1 evaluation ───────────────────────────────────────────
def evaluate_best_model(model, val_loader, device, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    confusion = [[0]*NUM_CLASSES for _ in range(NUM_CLASSES)]
    for p, t in zip(all_preds, all_labels):
        confusion[t][p] += 1

    correct = 0
    macro_f1 = 0.0
    per_class = {}

    for c in range(NUM_CLASSES):
        tp = confusion[c][c]
        fp = sum(confusion[i][c] for i in range(NUM_CLASSES)) - tp
        fn = sum(confusion[c][i] for i in range(NUM_CLASSES)) - tp
        correct += tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[CLASSES[c]] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': sum(confusion[c])
        }
        macro_f1 += f1

    acc = correct / len(all_labels)
    macro_f1 /= NUM_CLASSES

    return acc, macro_f1, per_class

# ── Main ──────────────────────────────────────────────────────────────
def main():
    BETA = 0.999
    SEED = 42
    BATCH_SIZE = 64
    LR = 0.001
    MAX_EPOCHS = 30
    PATIENCE = 5

    print("=" * 60)
    print("V4-E: Effective Number of Samples (ENS) Weighting")
    print(f"beta = {BETA}")
    print("=" * 60)

    set_seed(SEED)

    train_split, val_split = create_train_val_split()
    ens_weights, counts = calculate_ens_weights(train_split, beta=BETA)

    print("\n--- Class Counts & ENS Weights (beta=0.999) ---")
    print(f"{'Class':<10} | {'Count':<8} | {'ENS Weight':<12}")
    print("-" * 35)
    for cls in CLASSES:
        print(f"{cls:<10} | {counts[cls]:<8} | {ens_weights[cls]:.4f}")

    # Save config
    experiment_dir = SCRIPT_DIR
    config = {
        "experiment_name": "V4_E_ens_weights",
        "model_name": "ResidualCNN (V2-B / V3-B frozen architecture)",
        "weighting": f"Effective Number of Samples (beta={BETA})",
        "loss": "CrossEntropyLoss(weight=ens_weights)",
        "optimizer": "Adam",
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "random_seed": SEED,
        "augmentation": "V3-B moderate pipeline",
        "class_counts": counts,
        "ens_weights": ens_weights,
        "beta": BETA,
        "test_set_usage": "NOT EVALUATED"
    }

    config_dir = os.path.join(experiment_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    # Datasets
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    train_base = FacialExpressionDataset(train_split, augment=False)
    val_dataset = FacialExpressionDataset(val_split, augment=False)
    train_dataset = V3BTransformDataset(train_base, v3b_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ENS weight tensor (ordered by class index)
    weight_tensor = torch.tensor(
        [ens_weights[CLASSES[i]] for i in range(NUM_CLASSES)],
        dtype=torch.float32
    ).to(device)

    print(f"\nWeight tensor: {weight_tensor.tolist()}")

    model = ResidualCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train
    os.makedirs(os.path.join(experiment_dir, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "history"), exist_ok=True)

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        experiment_dir=experiment_dir,
        is_smoke_test=False
    )

    # Evaluate best checkpoint
    best_ckpt = os.path.join(experiment_dir, "checkpoint", "best_model.pth")
    acc, macro_f1, per_class = evaluate_best_model(model, val_loader, device, best_ckpt)

    print("\n" + "=" * 60)
    print("V4-E FINAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")
    print(f"\n{'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'Support'}")
    print("-" * 55)
    for cls in CLASSES:
        m = per_class[cls]
        print(f"{cls:<10} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['support']}")

    # Save results
    eval_results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "best_epoch": results["best_epoch"],
        "epochs_completed": results["epochs_completed"],
        "training_time_seconds": results["total_time_seconds"]
    }
    with open(os.path.join(experiment_dir, "v4e_results.json"), 'w') as f:
        json.dump(eval_results, f, indent=4)

    print(f"\nResults saved to v4e_results.json")

if __name__ == "__main__":
    main()
