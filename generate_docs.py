import os
import json
import csv

def main():
    out_dir = os.path.join("experiments", "consolidated_analysis")
    
    # Check if FINAL_ERROR_ANALYSIS.md needs fixes
    with open(os.path.join(out_dir, "FINAL_ERROR_ANALYSIS.md"), "r") as f:
        content = f.read()
        
    old_class_weighting = "- **Effect of Class Weighting (B1)**: Improved Macro F1 significantly (+0.0513 vs B0) by forcing the model to recognize minority classes (especially disgust) at the cost of a slight reduction in overall accuracy (-0.0078 vs B0)."
    new_class_weighting = "- **Effect of Class Weighting (B1)**: Test Macro F1 increased by +0.0513 relative to B0. The disgust class also changed from F1 = 0.0000 in B0 to F1 = 0.3806 in B1, while overall accuracy decreased by 0.0078."
    
    old_confusion = "**Confusion Behavior**: Without class weights (B0, B2), disgust predictions were almost entirely absent, resulting in 0.0000 precision and recall. Introducing class weights (B1, B3) activated disgust predictions, producing measurable precision and recall, though still limited by the overall rarity of the class."
    new_confusion = "**Confusion Behavior**: In B0 and B2, the model produced no correct disgust predictions, resulting in precision, recall, and F1 of 0.0000. B1 and B3 produced non-zero precision, recall, and F1 for disgust."
    
    if old_class_weighting in content:
        content = content.replace(old_class_weighting, new_class_weighting)
    if old_confusion in content:
        content = content.replace(old_confusion, new_confusion)
        
    with open(os.path.join(out_dir, "FINAL_ERROR_ANALYSIS.md"), "w") as f:
        f.write(content)
        
    # Generate RESULTS_AND_DISCUSSION.md
    with open(os.path.join(out_dir, "consolidated_results.json"), "r") as f:
        c_res = json.load(f)
        
    with open(os.path.join(out_dir, "confusion_summary.json"), "r") as f:
        c_conf = json.load(f)
        
    per_class_lines = []
    with open(os.path.join(out_dir, "per_class_f1.csv"), "r") as f:
        reader = csv.reader(f)
        per_class_lines = list(reader)
        
    b1_metrics = c_res["B1"]
    
    with open(os.path.join(out_dir, "RESULTS_AND_DISCUSSION.md"), "w") as f:
        f.write("# Results and Discussion\n\n")
        f.write("## 1. Experimental Design\n")
        f.write("Four controlled CNN experiments were conducted:\n")
        f.write("- **B0 baseline**: Standard CrossEntropyLoss, no training augmentation.\n")
        f.write("- **B1 class weighting**: Weighted CrossEntropyLoss, no training augmentation.\n")
        f.write("- **B2 augmentation**: Standard CrossEntropyLoss, training augmentation.\n")
        f.write("- **B3 class weighting + augmentation**: Weighted CrossEntropyLoss, training augmentation.\n\n")
        f.write("All four experiments use the same LightweightCNN architecture and controlled training setup, with only the specified intervention changed.\n\n")
        
        f.write("## 2. Overall Results\n")
        f.write("| Experiment | Best Val Macro F1 | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Best Epoch | Training Time (s) |\n")
        f.write("|------------|-------------------|---------------|----------------------|-------------------|---------------|------------|-------------------|\n")
        for exp_id in ["B0", "B1", "B2", "B3"]:
            data = c_res[exp_id]
            f.write(f"| {exp_id} | {data['best_val_macro_f1']:.4f} | {data['test_accuracy']:.4f} | {data['test_macro_precision']:.4f} | {data['test_macro_recall']:.4f} | {data['test_macro_f1']:.4f} | {data['best_epoch']} | {data['training_time_seconds']:.2f} |\n")
            
        f.write("\n## 3. Effect of Class Weighting\n")
        f.write("Comparing B1 vs B0, test Macro F1 increased from {:.4f} to {:.4f}. Test accuracy changed from {:.4f} to {:.4f}. For the disgust class, F1 changed from 0.0000 to 0.3806.\n\n".format(c_res["B0"]["test_macro_f1"], c_res["B1"]["test_macro_f1"], c_res["B0"]["test_accuracy"], c_res["B1"]["test_accuracy"]))
        
        f.write("## 4. Effect of Augmentation\n")
        f.write("Comparing B2 vs B0 under the tested augmentation (horizontal flip p=0.5, rotation [-10°, +10°]), test Macro F1 changed from {:.4f} to {:.4f}.\n\n".format(c_res["B0"]["test_macro_f1"], c_res["B2"]["test_macro_f1"]))
        
        f.write("## 5. Combined Intervention\n")
        f.write("Comparing B3 relative to the other configurations, the B3 Test Macro F1 was 0.4391. This was higher than B0 ({:.4f}) and B2 ({:.4f}), but lower than B1 (0.4738).\n\n".format(c_res["B0"]["test_macro_f1"], c_res["B2"]["test_macro_f1"]))
        
        f.write("## 6. Per-Class Results\n")
        headers = per_class_lines[0]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["-"*len(h) for h in headers]) + "|\n")
        for row in per_class_lines[1:]:
            f.write("| " + " | ".join(row) + " |\n")
            
        f.write("\n## 7. Error Patterns\n")
        f.write("The following recurring confusion transitions were recorded across B0-B3:\n\n")
        f.write("| Transition | B0 | B1 | B2 | B3 |\n")
        f.write("|------------|----|----|----|----|\n")
        transitions = ["fear_to_sad", "neutral_to_sad", "sad_to_neutral", "angry_to_sad"]
        for tr in transitions:
            t_str = tr.replace("_to_", " -> ")
            row = [f"**{t_str}**"]
            for exp_id in ["B0", "B1", "B2", "B3"]:
                row.append(str(c_conf[exp_id]["specific_counts"][tr]))
            f.write("| " + " | ".join(row) + " |\n")
            
        f.write("\n## 8. Limitations\n")
        f.write("- 93,575-parameter lightweight model\n")
        f.write("- severe class imbalance\n")
        f.write("- one augmentation configuration\n")
        f.write("- CPU-only training\n")
        f.write("- limited experimental scope\n\n")
        
        f.write("## 9. Selected Checkpoint\n")
        f.write("Based on the highest observed Validation Macro F1 among B0-B3, B1 (class-weighted) is the selected candidate.\n\n")
        f.write("- Checkpoint path: `experiments/B1_class_weighted/checkpoint/best_model.pth`\n")
        f.write(f"- Best Validation Macro F1: {b1_metrics['best_val_macro_f1']:.4f}\n\n")
        f.write("### Held-Out Test Metrics\n")
        f.write(f"- Test Accuracy: {b1_metrics['test_accuracy']:.4f}\n")
        f.write(f"- Test Macro Precision: {b1_metrics['test_macro_precision']:.4f}\n")
        f.write(f"- Test Macro Recall: {b1_metrics['test_macro_recall']:.4f}\n")
        f.write(f"- Test Macro F1: {b1_metrics['test_macro_f1']:.4f}\n")
        f.write("\nThis configuration had the highest observed Validation Macro F1 among the four tested configurations. The test set was used only for final held-out evaluation and was not used to select the model configuration.\n")
        
    # Generate MODEL_SELECTION.md
    with open(os.path.join(out_dir, "MODEL_SELECTION.md"), "w") as f:
        f.write("# Model Selection\n\n")
        f.write("- **Selected experiment**: B1_class_weighted\n")
        f.write("- **Checkpoint path**: `experiments/B1_class_weighted/checkpoint/best_model.pth`\n")
        f.write("- **Selection criterion**: Validation Macro F1\n")
        f.write(f"- **Best validation Macro F1**: {b1_metrics['best_val_macro_f1']:.4f}\n\n")
        f.write("### Held-Out Test Metrics\n")
        f.write(f"- **Test Accuracy**: {b1_metrics['test_accuracy']:.4f}\n")
        f.write(f"- **Test Macro Precision**: {b1_metrics['test_macro_precision']:.4f}\n")
        f.write(f"- **Test Macro Recall**: {b1_metrics['test_macro_recall']:.4f}\n")
        f.write(f"- **Test Macro F1**: {b1_metrics['test_macro_f1']:.4f}\n\n")
        f.write("The test set was used only for final held-out evaluation and was not used to select the model configuration.\n\n")
        f.write("The model selection is limited strictly to the four controlled configurations tested in this experimental series (B0, B1, B2, B3).\n")

if __name__ == "__main__":
    main()
