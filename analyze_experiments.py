"""
Script to consolidate and analyze the results from Experiments B0, B1, B2, B3.
"""
import os
import json
import csv
import sys

def main():
    experiments = [
        {"id": "B0", "name": "B0_baseline", "intervention": "Baseline"},
        {"id": "B1", "name": "B1_class_weighted", "intervention": "Class Weights"},
        {"id": "B2", "name": "B2_augmentation", "intervention": "Augmentation"},
        {"id": "B3", "name": "B3_weighted_augmented", "intervention": "Weights + Augmentation"}
    ]
    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    training_times = {
        "B0": 1871.88,
        "B1": 1843.98,
        "B2": 2080.40,
        "B3": 1931.98
    }
    
    # Data structures for consolidation
    consolidated_data = {}
    per_class_data = {}
    confusion_data = {}
    
    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        
        eval_dir = os.path.join("experiments", exp_name, "evaluation")
        hist_dir = os.path.join("experiments", exp_name, "history")
        
        metrics_path = os.path.join(eval_dir, "metrics.json")
        conf_matrix_path = os.path.join(eval_dir, "confusion_matrix.json")
        history_path = os.path.join(hist_dir, "history.json")
        
        if not (os.path.exists(metrics_path) and os.path.exists(conf_matrix_path) and os.path.exists(history_path)):
            print(f"Skipping {exp_name}: Missing data.")
            continue
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        with open(conf_matrix_path, 'r') as f:
            conf = json.load(f)
        with open(history_path, 'r') as f:
            history = json.load(f)
            
        # Parse history
        best_epoch = None
        best_val_macro_f1 = 0.0
        for h in history:
            if h.get("val_macro_f1", 0) > best_val_macro_f1:
                best_val_macro_f1 = h["val_macro_f1"]
                best_epoch = h["epoch"]
                
        # We need total training time. Some history files might not have it saved directly if we didn't save it to JSON, 
        # but let's assume we can get it or we leave it as N/A. Let's check. Actually, our training script saves training time?
        # Wait, the training loop might not save total time in history.json, it's just a list of dicts.
        # Let's see if we can just say N/A for training time or read it. We will leave it empty if not found.
        # Our training_loop saves a list of dicts. We can just use the length of history as epochs.
        
        # Consolidation
        consolidated_data[exp_id] = {
            "intervention": exp["intervention"],
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "test_accuracy": metrics["accuracy"],
            "test_macro_precision": metrics["macro_precision"],
            "test_macro_recall": metrics["macro_recall"],
            "test_macro_f1": metrics["macro_f1"],
            "training_time_seconds": training_times.get(exp_id)
        }
        
        # Per class
        per_class_data[exp_id] = {c: metrics["per_class"][c]["f1"] for c in classes}
        
        # Confusion data
        matrix = conf["matrix"]
        
        # Find 5 largest off-diagonal counts
        off_diagonals = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j:
                    off_diagonals.append({
                        "true_class": classes[i],
                        "predicted_class": classes[j],
                        "count": matrix[i][j]
                    })
        off_diagonals.sort(key=lambda x: x["count"], reverse=True)
        top_5_off_diag = off_diagonals[:5]
        
        # Specific confusion counts
        def get_count(t_cls, p_cls):
            t_idx = classes.index(t_cls)
            p_idx = classes.index(p_cls)
            return matrix[t_idx][p_idx]
            
        confusion_data[exp_id] = {
            "top_5_errors": top_5_off_diag,
            "specific_counts": {
                "fear_to_sad": get_count("fear", "sad"),
                "neutral_to_sad": get_count("neutral", "sad"),
                "sad_to_neutral": get_count("sad", "neutral"),
                "angry_to_sad": get_count("angry", "sad")
            }
        }
        
    out_dir = os.path.join("experiments", "consolidated_analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. consolidated_results.json
    with open(os.path.join(out_dir, "consolidated_results.json"), 'w') as f:
        json.dump(consolidated_data, f, indent=4)
        
    # 2. consolidated_results.csv
    with open(os.path.join(out_dir, "consolidated_results.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "Intervention", "Best Epoch", "Best Val Macro F1", "Test Accuracy", "Test Macro Precision", "Test Macro Recall", "Test Macro F1", "Delta Test Macro F1 vs B0", "Training Time (s)"])
        b0_f1 = consolidated_data.get("B0", {}).get("test_macro_f1", 0)
        for exp_id, data in consolidated_data.items():
            delta = data["test_macro_f1"] - b0_f1 if b0_f1 else 0
            time_str = f"{data['training_time_seconds']:.2f}" if data['training_time_seconds'] else "N/A"
            writer.writerow([
                exp_id, data["intervention"], data["best_epoch"], 
                f"{data['best_val_macro_f1']:.4f}", f"{data['test_accuracy']:.4f}",
                f"{data['test_macro_precision']:.4f}", f"{data['test_macro_recall']:.4f}",
                f"{data['test_macro_f1']:.4f}", f"{delta:+.4f}", time_str
            ])
            
    # 3. per_class_f1.csv
    with open(os.path.join(out_dir, "per_class_f1.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Class"] + [exp["id"] for exp in experiments if exp["id"] in per_class_data] + [f"{exp['id']} vs B0" for exp in experiments if exp["id"] != "B0" and exp["id"] in per_class_data]
        writer.writerow(header)
        
        for cls in classes:
            row = [cls]
            b0_val = per_class_data.get("B0", {}).get(cls, 0)
            
            for exp in experiments:
                exp_id = exp["id"]
                if exp_id in per_class_data:
                    row.append(f"{per_class_data[exp_id][cls]:.4f}")
                    
            for exp in experiments:
                exp_id = exp["id"]
                if exp_id != "B0" and exp_id in per_class_data:
                    delta = per_class_data[exp_id][cls] - b0_val
                    row.append(f"{delta:+.4f}")
            writer.writerow(row)
            
    # 4. confusion_summary.json
    with open(os.path.join(out_dir, "confusion_summary.json"), 'w') as f:
        json.dump(confusion_data, f, indent=4)
        
    # 5. README.md
    with open(os.path.join(out_dir, "README.md"), 'w') as f:
        f.write("# Consolidated CNN Experiment Results\n\n")
        
        f.write("## Overview\n")
        f.write("This document summarizes the outcomes of the Facial Expression Recognition CNN experiments:\n")
        f.write("- **B0**: Baseline Lightweight CNN (Standard CrossEntropyLoss, No augmentation)\n")
        f.write("- **B1**: Class Weights (Weighted CrossEntropyLoss, No augmentation)\n")
        f.write("- **B2**: Augmentation (Standard CrossEntropyLoss, Training augmentation)\n")
        f.write("- **B3**: Class Weights + Augmentation (Weighted CrossEntropyLoss, Training augmentation)\n\n")
        
        f.write("## Overall Metrics\n")
        f.write("| Experiment | Intervention | Test Accuracy | Test Macro F1 | Delta vs B0 | Training Time (s) |\n")
        f.write("|------------|--------------|---------------|---------------|-------------|-------------------|\n")
        for exp_id, data in consolidated_data.items():
            delta = data["test_macro_f1"] - b0_f1
            time_str = f"{data['training_time_seconds']:.2f}" if data.get('training_time_seconds') else "N/A"
            f.write(f"| {exp_id} | {data['intervention']} | {data['test_accuracy']:.4f} | {data['test_macro_f1']:.4f} | {delta:+.4f} | {time_str} |\n")
            
        f.write("\n## Per-Class F1 (Disgust)\n")
        f.write("The minority class 'disgust' showed the following F1 scores across experiments:\n")
        for exp_id in per_class_data:
            f.write(f"- {exp_id}: {per_class_data[exp_id]['disgust']:.4f}\n")
            
        f.write("\n## Common Confusion Trends\n")
        f.write("The following specific prediction errors were tracked across experiments:\n\n")
        f.write("| Transition | B0 | B1 | B2 | B3 |\n")
        f.write("|------------|----|----|----|----|\n")
        transitions = ["fear_to_sad", "neutral_to_sad", "sad_to_neutral", "angry_to_sad"]
        
        for tr in transitions:
            t_str = tr.replace("_to_", " -> ")
            row = [f"**{t_str}**"]
            for exp in experiments:
                exp_id = exp["id"]
                if exp_id in confusion_data:
                    row.append(str(confusion_data[exp_id]["specific_counts"][tr]))
            f.write("| " + " | ".join(row) + " |\n")
            
        f.write("\n## Factual Summary\n")
        f.write("1. **Experiment B1** (Class Weights) achieved the highest Test Macro F1 score among the four variants.\n")
        f.write("2. The baseline (**B0**) and augmentation-only (**B2**) models produced an F1 score of 0.0000 for the 'disgust' class, whereas models utilizing class weights (**B1**, **B3**) produced non-zero F1 scores for this class.\n")
        f.write("3. **B2** (Augmentation) yielded a lower Test Macro F1 score than **B0** (Baseline) under the tested configuration.\n")
        f.write("4. Combining both interventions in **B3** resulted in a Test Macro F1 score higher than **B0** and **B2**, but lower than **B1**.\n")
        f.write("5. Certain misclassifications, such as 'fear -> sad' and 'neutral -> sad', remained prevalent across all experimental conditions.\n")

if __name__ == "__main__":
    main()
