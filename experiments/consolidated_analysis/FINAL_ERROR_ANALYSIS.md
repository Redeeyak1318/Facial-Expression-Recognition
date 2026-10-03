# Final Error Analysis Report

## 1. Experimental Setup
This report details the outcomes of four controlled CNN experiments for Facial Expression Recognition:

- **B0**: Baseline Lightweight CNN (Standard CrossEntropyLoss, No augmentation)
- **B1**: Class Weights (Weighted CrossEntropyLoss, No augmentation)
- **B2**: Augmentation (Standard CrossEntropyLoss, Training augmentation)
- **B3**: Class Weights + Augmentation (Weighted CrossEntropyLoss, Training augmentation)

## 2. Overall Performance
| Experiment | Best Val Macro F1 | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Best Epoch | Training Time (s) |
|------------|-------------------|---------------|----------------------|-------------------|---------------|------------|-------------------|
| B0 | 0.4230 | 0.5116 | 0.4249 | 0.4294 | 0.4225 | 25 | 1871.88 |
| B1 | 0.4948 | 0.5038 | 0.4769 | 0.4862 | 0.4738 | 29 | 1843.98 |
| B2 | 0.4139 | 0.5104 | 0.4205 | 0.4205 | 0.4172 | 29 | 2080.40 |
| B3 | 0.4497 | 0.4967 | 0.4376 | 0.4493 | 0.4391 | 29 | 1931.98 |

## 3. Per-Class F1 Analysis
Comparison of F1 scores across all seven classes for B0-B3:

| Class | B0 | B1 | B2 | B3 | B1 vs B0 | B2 vs B0 | B3 vs B0 |
|-----|--|--|--|--|--------|--------|--------|
| angry | 0.3475 | 0.3974 | 0.3492 | 0.3667 | +0.0499 | +0.0016 | +0.0191 |
| disgust | 0.0000 | 0.3806 | 0.0000 | 0.2458 | +0.3806 | +0.0000 | +0.2458 |
| fear | 0.2892 | 0.2603 | 0.2817 | 0.2214 | -0.0288 | -0.0075 | -0.0678 |
| happy | 0.7337 | 0.7115 | 0.7248 | 0.7032 | -0.0223 | -0.0090 | -0.0306 |
| neutral | 0.4932 | 0.4783 | 0.4916 | 0.4763 | -0.0149 | -0.0015 | -0.0168 |
| sad | 0.4330 | 0.4318 | 0.4079 | 0.4141 | -0.0013 | -0.0252 | -0.0189 |
| surprise | 0.6611 | 0.6568 | 0.6650 | 0.6461 | -0.0043 | +0.0039 | -0.0150 |

## 4. Minority-Class Analysis (Disgust)
The `disgust` class suffers from severe class imbalance. Metrics for disgust:

| Experiment | Precision | Recall | F1 | Delta F1 vs B0 |
|------------|-----------|--------|----|----------------|
| B0 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| B1 | 0.3456 | 0.4234 | 0.3806 | +0.3806 |
| B2 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| B3 | 0.2320 | 0.2613 | 0.2458 | +0.2458 |

**Confusion Behavior**: In B0 and B2, the model produced no correct disgust predictions, resulting in precision, recall, and F1 of 0.0000. B1 and B3 produced non-zero precision, recall, and F1 for disgust.

## 5. Confusion Analysis
Recurring error patterns observed across all experiments:

| Transition | B0 | B1 | B2 | B3 |
|------------|----|----|----|----|
| **fear -> sad** | 278 | 317 | 255 | 263 |
| **neutral -> sad** | 273 | 363 | 222 | 253 |
| **sad -> neutral** | 233 | 176 | 283 | 236 |
| **angry -> sad** | 254 | 246 | 202 | 190 |

Other notable off-diagonal confusions included neutral -> fear and angry -> neutral, which frequently appeared in the top-5 highest off-diagonal counts across models.

## 6. Intervention Analysis
- **Effect of Class Weighting (B1)**: Test Macro F1 increased by +0.0513 relative to B0. The disgust class also changed from F1 = 0.0000 in B0 to F1 = 0.3806 in B1, while overall accuracy decreased by 0.0078.
- **Effect of Augmentation (B2)**: Produced a slight decrease in Test Macro F1 (-0.0054 vs B0) and Test Accuracy (-0.0012 vs B0) compared to the baseline under this specific augmentation configuration (horizontal flips, rotation). It did not solve the minority-class problem.
- **Effect of Combining Both (B3)**: Yielded a Test Macro F1 (+0.0165 vs B0) higher than the baseline but lower than class weights alone (B1). In this experimental configuration, adding augmentation to the class-weighted model reduced Macro F1 relative to B1.

## 7. Final Experimental Findings
**Direct Observations:**
- Class weighting explicitly raised the F1 score for minority classes.
- Augmentation on this specific lightweight CNN and dataset did not improve generalization test metrics over the baseline.
- The B1 configuration achieved the highest Macro F1 across all four trials.

**Interpretations/Hypotheses:**
- The observed lack of benefit from B2 augmentation may indicate that the geometric transformations (flip, rotation) introduced variations that the heavily constrained LightweightCNN (93k parameters) could not fully model, or that the specific augmentation regime was not optimal for facial expression data.
- In these experiments, class weighting was associated with substantially higher Macro F1 and non-zero F1 for the minority disgust class compared with the unweighted configurations.

## 8. Limitations
- **Lightweight Architecture**: The CNN contains only 93,575 parameters, potentially limiting its representational capacity to benefit from augmented data.
- **Severe Class Imbalance**: Classes like disgust are exceptionally rare, making robust evaluation challenging even with class weights.
- **Single Augmentation Configuration**: The augmentation was limited to uniform rotations [-10, 10] and horizontal flips (p=0.5). Other strategies (e.g., cropping, brightness adjustment) were not tested.
- **CPU-Only Training**: Iteration speed was restricted by CPU training, which limited the scale of hyperparameter tuning.
- **Limited Experimental Scope**: Only four specific interventions were compared; learning rate schedules, weight decay, and dropout tuning were held constant.

## 9. Candidate Model
Based on the Test Macro F1 criterion, **Experiment B1 (Class-Weighted)** is the best-performing candidate configuration.

- **Test Accuracy**: 0.5038
- **Test Macro Precision**: 0.4769
- **Test Macro Recall**: 0.4862
- **Test Macro F1**: 0.4738
