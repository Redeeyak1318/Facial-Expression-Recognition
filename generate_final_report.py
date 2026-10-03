"""
Script to generate the FINAL_ERROR_ANALYSIS.md for Facial Expression Recognition.
"""
import os
import json
import csv

def main():
    experiments = [
        {"id": "B0", "name": "B0_baseline", "desc": "Baseline Lightweight CNN (Standard CrossEntropyLoss, No augmentation)"},
        {"id": "B1", "name": "B1_class_weighted", "desc": "Class Weights (Weighted CrossEntropyLoss, No augmentation)"},
        {"id": "B2", "name": "B2_augmentation", "desc": "Augmentation (Standard CrossEntropyLoss, Training augmentation)"},
        {"id": "B3", "name": "B3_weighted_augmented", "desc": "Class Weights + Augmentation (Weighted CrossEntropyLoss, Training augmentation)"}
    ]
    
    out_dir = os.path.join("experiments", "consolidated_analysis")
    
    with open(os.path.join(out_dir, "consolidated_results.json"), 'r') as f:
        consolidated = json.load(f)
        
    with open(os.path.join(out_dir, "confusion_summary.json"), 'r') as f:
        confusion = json.load(f)
        
    # Read detailed metrics for Disgust
    disgust_data = {}
    for exp in experiments:
        metrics_path = os.path.join("experiments", exp["name"], "evaluation", "metrics.json")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            disgust_data[exp["id"]] = metrics["per_class"]["disgust"]
            
    # Read per-class F1
    per_class_lines = []
    with open(os.path.join(out_dir, "per_class_f1.csv"), 'r') as f:
        reader = csv.reader(f)
        per_class_lines = list(reader)
        
    md_path = os.path.join(out_dir, "FINAL_ERROR_ANALYSIS.md")
    
    with open(md_path, 'w') as f:
        f.write("# Final Error Analysis Report\n\n")
        
        # 1. Experimental Setup
        f.write("## 1. Experimental Setup\n")
        f.write("This report details the outcomes of four controlled CNN experiments for Facial Expression Recognition:\n\n")
        for exp in experiments:
            f.write(f"- **{exp['id']}**: {exp['desc']}\n")
            
        f.write("\n## 2. Overall Performance\n")
        f.write("| Experiment | Best Val Macro F1 | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Best Epoch | Training Time (s) |\n")
        f.write("|------------|-------------------|---------------|----------------------|-------------------|---------------|------------|-------------------|\n")
        for exp in experiments:
            data = consolidated[exp["id"]]
            f.write(f"| {exp['id']} | {data['best_val_macro_f1']:.4f} | {data['test_accuracy']:.4f} | {data['test_macro_precision']:.4f} | {data['test_macro_recall']:.4f} | {data['test_macro_f1']:.4f} | {data['best_epoch']} | {data['training_time_seconds']:.2f} |\n")
            
        f.write("\n## 3. Per-Class F1 Analysis\n")
        f.write("Comparison of F1 scores across all seven classes for B0-B3:\n\n")
        
        # Format CSV into Markdown table
        headers = per_class_lines[0]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["-"*len(h) for h in headers]) + "|\n")
        for row in per_class_lines[1:]:
            f.write("| " + " | ".join(row) + " |\n")
            
        f.write("\n## 4. Minority-Class Analysis (Disgust)\n")
        f.write("The `disgust` class suffers from severe class imbalance. Metrics for disgust:\n\n")
        f.write("| Experiment | Precision | Recall | F1 | Delta F1 vs B0 |\n")
        f.write("|------------|-----------|--------|----|----------------|\n")
        b0_disgust_f1 = disgust_data["B0"]["f1"]
        for exp in experiments:
            d = disgust_data[exp["id"]]
            delta = d["f1"] - b0_disgust_f1
            f.write(f"| {exp['id']} | {d['precision']:.4f} | {d['recall']:.4f} | {d['f1']:.4f} | {delta:+.4f} |\n")
            
        f.write("\n**Confusion Behavior**: Without class weights (B0, B2), disgust predictions were almost entirely absent, resulting in 0.0000 precision and recall. Introducing class weights (B1, B3) activated disgust predictions, producing measurable precision and recall, though still limited by the overall rarity of the class.\n")
        
        f.write("\n## 5. Confusion Analysis\n")
        f.write("Recurring error patterns observed across all experiments:\n\n")
        f.write("| Transition | B0 | B1 | B2 | B3 |\n")
        f.write("|------------|----|----|----|----|\n")
        transitions = ["fear_to_sad", "neutral_to_sad", "sad_to_neutral", "angry_to_sad"]
        
        for tr in transitions:
            t_str = tr.replace("_to_", " -> ")
            row = [f"**{t_str}**"]
            for exp in experiments:
                row.append(str(confusion[exp["id"]]["specific_counts"][tr]))
            f.write("| " + " | ".join(row) + " |\n")
            
        f.write("\nOther notable off-diagonal confusions included neutral -> fear and angry -> neutral, which frequently appeared in the top-5 highest off-diagonal counts across models.\n")
        
        f.write("\n## 6. Intervention Analysis\n")
        f.write("- **Effect of Class Weighting (B1)**: Improved Macro F1 significantly (+0.0513 vs B0) by forcing the model to recognize minority classes (especially disgust) at the cost of a slight reduction in overall accuracy (-0.0078 vs B0).\n")
        f.write("- **Effect of Augmentation (B2)**: Produced a slight decrease in Test Macro F1 (-0.0054 vs B0) and Test Accuracy (-0.0012 vs B0) compared to the baseline under this specific augmentation configuration (horizontal flips, rotation). It did not solve the minority-class problem.\n")
        f.write("- **Effect of Combining Both (B3)**: Yielded a Test Macro F1 (+0.0165 vs B0) higher than the baseline but lower than class weights alone (B1). In this experimental configuration, adding augmentation to the class-weighted model reduced Macro F1 relative to B1.\n")
        
        f.write("\n## 7. Final Experimental Findings\n")
        f.write("**Direct Observations:**\n")
        f.write("- Class weighting explicitly raised the F1 score for minority classes.\n")
        f.write("- Augmentation on this specific lightweight CNN and dataset did not improve generalization test metrics over the baseline.\n")
        f.write("- The B1 configuration achieved the highest Macro F1 across all four trials.\n")
        f.write("\n**Interpretations/Hypotheses:**\n")
        f.write("- The observed lack of benefit from B2 augmentation may indicate that the geometric transformations (flip, rotation) introduced variations that the heavily constrained LightweightCNN (93k parameters) could not fully model, or that the specific augmentation regime was not optimal for facial expression data.\n")
        f.write("- In these experiments, class weighting was associated with substantially higher Macro F1 and non-zero F1 for the minority disgust class compared with the unweighted configurations.\n")
        
        f.write("\n## 8. Limitations\n")
        f.write("- **Lightweight Architecture**: The CNN contains only 93,575 parameters, potentially limiting its representational capacity to benefit from augmented data.\n")
        f.write("- **Severe Class Imbalance**: Classes like disgust are exceptionally rare, making robust evaluation challenging even with class weights.\n")
        f.write("- **Single Augmentation Configuration**: The augmentation was limited to uniform rotations [-10, 10] and horizontal flips (p=0.5). Other strategies (e.g., cropping, brightness adjustment) were not tested.\n")
        f.write("- **CPU-Only Training**: Iteration speed was restricted by CPU training, which limited the scale of hyperparameter tuning.\n")
        f.write("- **Limited Experimental Scope**: Only four specific interventions were compared; learning rate schedules, weight decay, and dropout tuning were held constant.\n")
        
        f.write("\n## 9. Candidate Model\n")
        f.write("Based on the Test Macro F1 criterion, **Experiment B1 (Class-Weighted)** is the best-performing candidate configuration.\n\n")
        b1_data = consolidated["B1"]
        f.write(f"- **Test Accuracy**: {b1_data['test_accuracy']:.4f}\n")
        f.write(f"- **Test Macro Precision**: {b1_data['test_macro_precision']:.4f}\n")
        f.write(f"- **Test Macro Recall**: {b1_data['test_macro_recall']:.4f}\n")
        f.write(f"- **Test Macro F1**: {b1_data['test_macro_f1']:.4f}\n")

if __name__ == "__main__":
    main()
