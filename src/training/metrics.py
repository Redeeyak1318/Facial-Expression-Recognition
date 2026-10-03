"""
Module for calculating evaluation metrics.
"""
import torch

def calculate_accuracy_and_macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 7):
    """
    Calculates accuracy and macro F1 score.
    
    Args:
        preds: 1D tensor of predicted class indices.
        targets: 1D tensor of true class indices.
        num_classes: Number of classes in the dataset.
        
    Returns:
        accuracy (float), macro_f1 (float)
    """
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total if total > 0 else 0.0
    
    f1_scores = []
    
    for c in range(num_classes):
        # True Positives
        tp = ((preds == c) & (targets == c)).sum().item()
        # False Positives
        fp = ((preds == c) & (targets != c)).sum().item()
        # False Negatives
        fn = ((preds != c) & (targets == c)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        f1_scores.append(f1)
        
    macro_f1 = sum(f1_scores) / len(f1_scores)
    return accuracy, macro_f1
