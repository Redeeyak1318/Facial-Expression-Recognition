# Facial Expression Recognition Using a Lightweight CNN

## Abstract
This report documents a systematic experimental study of Facial Expression Recognition using a highly constrained, 93,575-parameter Lightweight Convolutional Neural Network (CNN). The project classified 48×48 grayscale facial images into seven discrete emotion categories. The study rigorously evaluates four controlled configurations—a baseline (B0), a class-weighted variant (B1), an augmentation-only variant (B2), and a combined variant (B3)—to isolate the impacts of interventions on minority-class recognition and generalization. The B1 configuration (class weighting alone) achieved the highest Validation Macro F1 and was formally selected, with a substantial increase in the minority-class 'disgust' F1 relative to the baseline.

## 1. Introduction
Facial expression recognition is a challenging computer vision task due to high intra-class variance, subtle inter-class differences, and pervasive class imbalance in naturalistic datasets. This project aims to systematically evaluate the performance of a minimal CNN architecture under strict experimental controls, observing how different training interventions (such as loss weighting and data augmentation) affect the network's ability to learn across imbalanced classes. 

## 2. Problem Statement
Given 48×48 grayscale images of human faces, the system must classify each image into one of seven emotional states: angry, disgust, fear, happy, neutral, sad, and surprise. The dataset suffers from severe class imbalance, particularly for the 'disgust' class, leading standard learning algorithms to disproportionately favor majority classes while entirely ignoring minority classes.

## 3. Objectives
- Establish a reproducible baseline (B0) using a minimal CNN architecture.
- Conduct strictly controlled, single-variable experiments to evaluate the impact of class weighting (B1) and data augmentation (B2).
- Evaluate the combined effect of both interventions (B3).
- Utilize a strict separation of validation data for model selection and untouched test data for final held-out evaluation.
- Identify the best-performing intervention for minority-class activation using Validation Macro F1.

## 4. Dataset
The dataset utilizes a fixed split to ensure strict experimental control.
- **Total images**: 35,887
- **Original training images**: 28,709
- **Untouched test images**: 7,178
- **Validation split**: A 90/10 stratified split from the original training set
- **Random seed**: 42

**Final split configuration:**
- Training set: 25,837
- Validation set: 2,872
- Test set: 7,178

The problem targets seven expression classes: angry, disgust, fear, happy, neutral, sad, and surprise.

![Figure 1: Dataset Class Distribution](figures/dataset_class_distribution.png)
*Figure 1: Class distribution across train, validation, and test sets. This highlights the severe class imbalance, particularly for the 'disgust' class.*

## 5. Data Preprocessing
To standardize inputs for the CNN, the following preprocessing pipeline was applied:
- Grayscale conversion
- Verification of 48×48 pixel dimensions
- Normalization of pixel values to the range [0, 1]
- Lazy/path-based image loading via a custom PyTorch Dataset
- Fixed, programmatic mapping of string class names to integer labels

## 6. CNN Architecture
The network is a custom `LightweightCNN` designed to deliberately constrain capacity, featuring only 93,575 trainable parameters.

Architecture sequence:
- Input: 48×48×1
- Conv2d (1 → 32), 3×3, padding=1
- ReLU
- MaxPool 2×2
- Conv2d (32 → 64), 3×3, padding=1
- ReLU
- MaxPool 2×2
- Conv2d (64 → 128), 3×3, padding=1
- ReLU
- MaxPool 2×2
- Global Average Pooling
- Dropout (p=0.30)
- Linear (128 → 7)

![Figure 2: CNN Architecture](figures/cnn_architecture.png)
*Figure 2: Flow diagram of the LightweightCNN architecture showing the sequential convolutional blocks, pooling, and dropout layers.*

## 7. Training Methodology
All models were trained using identical hyperparameter configurations to maintain strict control:
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Max Epochs**: 30
- **Early Stopping Patience**: 5
- **Hardware**: CPU (explicitly set for fixed random seed execution)
- **Random Seed**: 42
- **Model Selection**: The best checkpoint per experiment was selected using the highest Validation Macro F1 score.

## 8. Controlled Experiments
Four experimental variants were evaluated. The same architecture and training setup were retained across all runs, with only the specified intervention changed:

1. **B0 (Baseline)**: Standard CrossEntropyLoss, no training augmentation.
2. **B1 (Class Weighting)**: Weighted CrossEntropyLoss (programmatically calculated from the training split), no training augmentation.
3. **B2 (Augmentation)**: Standard CrossEntropyLoss, training augmentation (horizontal flip p=0.5, rotation [-10°, +10°]).
4. **B3 (Weighting + Augmentation)**: Weighted CrossEntropyLoss, training augmentation.

## 9. Experimental Results
The table below details the performance of each experimental configuration. 

| Experiment | Best Val Macro F1 | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Best Epoch | Training Time (s) |
|------------|-------------------|---------------|----------------------|-------------------|---------------|------------|-------------------|
| B0 | 0.4230 | 0.5116 | 0.4249 | 0.4294 | 0.4225 | 25 | 1871.88 |
| B1 | 0.4948 | 0.5038 | 0.4769 | 0.4862 | 0.4738 | 29 | 1843.98 |
| B2 | 0.4139 | 0.5104 | 0.4205 | 0.4205 | 0.4172 | 29 | 2080.40 |
| B3 | 0.4497 | 0.4967 | 0.4376 | 0.4493 | 0.4391 | 29 | 1931.98 |

![Figure 3: Test Macro F1 Comparison](figures/experiment_macro_f1.png)
*Figure 3: Test Macro F1 across B0-B3. The B1 configuration with class weights achieved the highest held-out test Macro F1.*

![Figure 4: Validation vs Test Macro F1](figures/validation_vs_test_macro_f1.png)
*Figure 4: Comparison of Best Validation Macro F1 vs Test Macro F1 across experiments. This illustrates the relationship between the metric used for model selection (Validation F1) and the final evaluation metric (Test F1).*

## 10. Per-Class Analysis
The following table highlights the breakdown of Test F1 scores per class.

| Class | B0 | B1 | B2 | B3 | B1 vs B0 | B2 vs B0 | B3 vs B0 |
|-----|--|--|--|--|--------|--------|--------|
| angry | 0.3475 | 0.3974 | 0.3492 | 0.3667 | +0.0499 | +0.0016 | +0.0191 |
| disgust | 0.0000 | 0.3806 | 0.0000 | 0.2458 | +0.3806 | +0.0000 | +0.2458 |
| fear | 0.2892 | 0.2603 | 0.2817 | 0.2214 | -0.0288 | -0.0075 | -0.0678 |
| happy | 0.7337 | 0.7115 | 0.7248 | 0.7032 | -0.0223 | -0.0090 | -0.0306 |
| neutral | 0.4932 | 0.4783 | 0.4916 | 0.4763 | -0.0149 | -0.0015 | -0.0168 |
| sad | 0.4330 | 0.4318 | 0.4079 | 0.4141 | -0.0013 | -0.0252 | -0.0189 |
| surprise | 0.6611 | 0.6568 | 0.6650 | 0.6461 | -0.0043 | +0.0039 | -0.0150 |

![Figure 5: Per-Class F1 Score Comparison](figures/per_class_f1_comparison.png)
*Figure 5: Per-class F1 score comparison across B0-B3. This visually demonstrates the activation of the 'disgust' class when class weighting is applied (B1 and B3).*

## 11. Error Analysis
In B0 and B2, the model produced no correct disgust predictions, resulting in precision, recall, and F1 of 0.0000. B1 and B3 produced non-zero precision, recall, and F1 for disgust.

The following recurring confusion transitions were recorded across B0-B3:

| Transition | B0 | B1 | B2 | B3 |
|------------|----|----|----|----|
| **fear -> sad** | 278 | 317 | 255 | 263 |
| **neutral -> sad** | 273 | 363 | 222 | 253 |
| **sad -> neutral** | 233 | 176 | 283 | 236 |
| **angry -> sad** | 254 | 246 | 202 | 190 |

![Figure 6: B1 Official Test Confusion Matrix](figures/b1_confusion_matrix.png)
*Figure 6: Confusion matrix for the selected B1 model on the official test set. The majority of off-diagonal errors are concentrated in confusions with the 'sad' class.*

## 12. Discussion
- **Effect of Class Weighting (B1)**: Test Macro F1 increased by +0.0513 relative to B0. The disgust class also changed from F1 = 0.0000 in B0 to F1 = 0.3806 in B1, while overall accuracy decreased slightly by 0.0078.
- **Effect of Augmentation (B2)**: Comparing B2 vs B0 under the tested augmentation (horizontal flip p=0.5, rotation [-10°, +10°]), Test Macro F1 changed from 0.4225 to 0.4172.
- **Combined Intervention (B3)**: Comparing B3 relative to the other configurations, the B3 Test Macro F1 was 0.4391. This was higher than B0 (0.4225) and B2 (0.4172), but lower than B1 (0.4738). In this experimental configuration, adding augmentation to the class-weighted model reduced Macro F1 relative to B1.

## 13. Model Selection
Based on the defined selection criteria, the class-weighted configuration was chosen as the final candidate.
- **Selected Candidate**: B1_class_weighted
- **Selection Criterion**: Highest Validation Macro F1 among B0-B3
- **Best Validation Macro F1**: 0.4948
- **Checkpoint Path**: `experiments/B1_class_weighted/checkpoint/best_model.pth`

### Held-Out Test Metrics
- **Test Accuracy**: 0.5038
- **Test Macro Precision**: 0.4769
- **Test Macro Recall**: 0.4862
- **Test Macro F1**: 0.4738

The test set was used only for final held-out evaluation and was not used to select the model configuration. This configuration had the highest observed Validation Macro F1 among the four tested configurations.

## 14. Limitations
- **Lightweight Architecture**: The network utilizes only 93,575 parameters.
- **Severe Class Imbalance**: Naturalistic class frequencies hindered learning on specific minority subsets.
- **One Augmentation Configuration**: Only horizontal flips and ±10° uniform rotations were tested.
- **CPU-Only Training**: Hardware constraints dictated CPU execution for fixed random seed execution, increasing training times.
- **Limited Experimental Scope**: The study was strictly limited to four controlled variables without expanding to hyperparameter tuning over dropout or learning rate schedules.

## 15. Conclusion
Through strict experimental controls on a lightweight convolutional neural network, class weighting was associated with improved recognition of the minority `disgust` class and produced the highest Validation Macro F1 and highest held-out Test Macro F1 among the four evaluated configurations.

## 16. Future Work
Future investigative directions built upon these findings may include:
- Broader augmentation studies incorporating spatial and color jitters.
- Architecture comparisons utilizing high-capacity transfer learning (e.g., MobileNet, ResNet).
- Advanced learning-rate scheduling algorithms (e.g., Cosine Annealing, ReduceLROnPlateau).
- Additional imbalance-handling strategies such as focal loss or minority class oversampling.

## 17. Reproducibility
The project uses a fixed experimental configuration with random seed 42. To reproduce:
1. Activate the Python virtual environment: `.\.venv\Scripts\Activate.ps1`
2. Verify PyTorch environment: `python verify_pytorch.py`
3. Verify the data pipeline: `python verify_pipeline.py`
4. Experiment checkpoints and configuration JSONs are preserved in the `experiments/` directory. (Note: Retraining from scratch requires significant CPU time and should be done intentionally via `train_b*.py` scripts).

## 18. Project Artifacts
Key artifacts documenting these experiments are preserved in `experiments/consolidated_analysis/`:
- `RESULTS_AND_DISCUSSION.md`
- `FINAL_ERROR_ANALYSIS.md`
- `MODEL_SELECTION.md`
- `FINAL_PROJECT_REPORT_WITH_FIGURES.md`
- `consolidated_results.json` & `per_class_f1.csv`
- `/figures/` (Generated high-resolution PNGs)
