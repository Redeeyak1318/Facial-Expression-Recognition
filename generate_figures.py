import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np

# Data directories
consol_dir = os.path.join("experiments", "consolidated_analysis")
fig_dir = os.path.join(consol_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# 1. Load Data
with open(os.path.join(consol_dir, "consolidated_results.json"), "r") as f:
    c_res = json.load(f)

classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
exp_ids = ["B0", "B1", "B2", "B3"]

test_macro_f1s = [c_res[e]["test_macro_f1"] for e in exp_ids]
val_macro_f1s = [c_res[e]["best_val_macro_f1"] for e in exp_ids]

# 2. experiment_macro_f1.png
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(exp_ids, test_macro_f1s)
ax.set_ylim(0, 0.6)
ax.set_ylabel('Test Macro F1')
ax.set_title('Test Macro F1 by Experiment')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
plt.savefig(os.path.join(fig_dir, "experiment_macro_f1.png"), dpi=300)
plt.close()

# 3. validation_vs_test_macro_f1.png
x = np.arange(len(exp_ids))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, val_macro_f1s, width, label='Best Validation Macro F1')
rects2 = ax.bar(x + width/2, test_macro_f1s, width, label='Test Macro F1')
ax.set_ylabel('Macro F1 Score')
ax.set_title('Validation vs Test Macro F1')
ax.set_xticks(x)
ax.set_xticklabels(exp_ids)
ax.legend()
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
plt.savefig(os.path.join(fig_dir, "validation_vs_test_macro_f1.png"), dpi=300)
plt.close()

# 4. per_class_f1_comparison.png
per_class_f1 = {e: [] for e in exp_ids}
with open(os.path.join(consol_dir, "per_class_f1.csv"), "r") as f:
    reader = csv.reader(f)
    next(reader) # header
    for row in reader:
        # cls = row[0]
        per_class_f1["B0"].append(float(row[1]))
        per_class_f1["B1"].append(float(row[2]))
        per_class_f1["B2"].append(float(row[3]))
        per_class_f1["B3"].append(float(row[4]))

x = np.arange(len(classes))
width = 0.2
fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(x - 1.5*width, per_class_f1["B0"], width, label='B0')
ax.bar(x - 0.5*width, per_class_f1["B1"], width, label='B1')
ax.bar(x + 0.5*width, per_class_f1["B2"], width, label='B2')
ax.bar(x + 1.5*width, per_class_f1["B3"], width, label='B3')
ax.set_ylabel('F1 Score')
ax.set_title('Per-Class F1 Score Comparison (B0-B3)')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
plt.savefig(os.path.join(fig_dir, "per_class_f1_comparison.png"), dpi=300)
plt.close()

# 5. b1_confusion_matrix.png
with open(os.path.join("experiments", "B1_class_weighted", "evaluation", "confusion_matrix.json"), "r") as f:
    b1_cm = json.load(f)["matrix"]

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(b1_cm)
fig.colorbar(cax)
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, b1_cm[i][j], ha="center", va="center", color="black" if b1_cm[i][j] < (np.max(b1_cm)/2) else "white")
plt.title('B1 Official Test Confusion Matrix', pad=20)
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "b1_confusion_matrix.png"), dpi=300)
plt.close()

# 6. dataset_class_distribution.png
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.data.split import create_train_val_split, get_test_paths
train_split, val_split = create_train_val_split()
test_paths = get_test_paths()

train_counts = [len(train_split[c]) for c in classes]
val_counts = [len(val_split[c]) for c in classes]
test_counts = [len(test_paths[c]) for c in classes]

fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(classes))
ax.bar(classes, train_counts, label='Train', bottom=bottom)
bottom += train_counts
ax.bar(classes, val_counts, label='Validation', bottom=bottom)
bottom += val_counts
ax.bar(classes, test_counts, label='Test', bottom=bottom)

ax.set_ylabel('Number of Images')
ax.set_title('Dataset Class Distribution (Train/Val/Test)')
ax.legend()
for i in range(len(classes)):
    total = train_counts[i] + val_counts[i] + test_counts[i]
    ax.text(i, total + 200, str(total), ha='center', va='bottom', fontsize=9)

plt.savefig(os.path.join(fig_dir, "dataset_class_distribution.png"), dpi=300)
plt.close()

# 7. cnn_architecture.png
fig, ax = plt.subplots(figsize=(6, 10))
ax.axis('off')
layers = [
    "Input 48x48x1",
    "Conv 1->32, 3x3, pad=1",
    "ReLU",
    "MaxPool 2x2",
    "Conv 32->64, 3x3, pad=1",
    "ReLU",
    "MaxPool 2x2",
    "Conv 64->128, 3x3, pad=1",
    "ReLU",
    "MaxPool 2x2",
    "Global Average Pooling",
    "Dropout 0.30",
    "Linear 128->7",
    "Output (7 classes)\n\nParams: 93,575"
]

y = 0.95
for layer in layers:
    bbox = dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black", lw=1.5)
    ax.text(0.5, y, layer, ha="center", va="center", size=11, bbox=bbox)
    if y > 0.15:
        ax.annotate("", xy=(0.5, y-0.035), xytext=(0.5, y-0.015), arrowprops=dict(arrowstyle="->", lw=1.5))
    y -= 0.07

plt.title("LightweightCNN Architecture", size=14, pad=10)
plt.savefig(os.path.join(fig_dir, "cnn_architecture.png"), dpi=300, bbox_inches='tight')
plt.close()

# 8. FIGURES_README.md
readme_path = os.path.join(fig_dir, "FIGURES_README.md")
with open(readme_path, "w") as f:
    f.write("# Project Figures\n\n")
    f.write("This directory contains visualizations summarizing the experiment results and architecture.\n\n")
    f.write("- **experiment_macro_f1.png**: Shows Test Macro F1 across B0-B3 based on the consolidated JSON.\n")
    f.write("- **validation_vs_test_macro_f1.png**: Compares best validation vs test Macro F1 across B0-B3.\n")
    f.write("- **per_class_f1_comparison.png**: Displays the breakdown of F1 scores per class for B0-B3.\n")
    f.write("- **b1_confusion_matrix.png**: Confusion matrix extracted from B1's official test evaluation output.\n")
    f.write("- **dataset_class_distribution.png**: Training, validation, and test sample counts per class.\n")
    f.write("- **cnn_architecture.png**: Flow diagram representing the 93,575-parameter LightweightCNN structure.\n")
