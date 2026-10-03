# Consolidated CNN Experiment Results

## Overview
This document summarizes the outcomes of the Facial Expression Recognition CNN experiments:
- **B0**: Baseline Lightweight CNN (Standard CrossEntropyLoss, No augmentation)
- **B1**: Class Weights (Weighted CrossEntropyLoss, No augmentation)
- **B2**: Augmentation (Standard CrossEntropyLoss, Training augmentation)
- **B3**: Class Weights + Augmentation (Weighted CrossEntropyLoss, Training augmentation)

## Overall Metrics
| Experiment | Intervention | Test Accuracy | Test Macro F1 | Delta vs B0 | Training Time (s) |
|------------|--------------|---------------|---------------|-------------|-------------------|
| B0 | Baseline | 0.5116 | 0.4225 | +0.0000 | 1871.88 |
| B1 | Class Weights | 0.5038 | 0.4738 | +0.0513 | 1843.98 |
| B2 | Augmentation | 0.5104 | 0.4172 | -0.0054 | 2080.40 |
| B3 | Weights + Augmentation | 0.4967 | 0.4391 | +0.0165 | 1931.98 |

## Per-Class F1 (Disgust)
The minority class 'disgust' showed the following F1 scores across experiments:
- B0: 0.0000
- B1: 0.3806
- B2: 0.0000
- B3: 0.2458

## Common Confusion Trends
The following specific prediction errors were tracked across experiments:

| Transition | B0 | B1 | B2 | B3 |
|------------|----|----|----|----|
| **fear -> sad** | 278 | 317 | 255 | 263 |
| **neutral -> sad** | 273 | 363 | 222 | 253 |
| **sad -> neutral** | 233 | 176 | 283 | 236 |
| **angry -> sad** | 254 | 246 | 202 | 190 |

## Factual Summary
1. **Experiment B1** (Class Weights) achieved the highest Test Macro F1 score among the four variants.
2. The baseline (**B0**) and augmentation-only (**B2**) models produced an F1 score of 0.0000 for the 'disgust' class, whereas models utilizing class weights (**B1**, **B3**) produced non-zero F1 scores for this class.
3. **B2** (Augmentation) yielded a lower Test Macro F1 score than **B0** (Baseline) under the tested configuration.
4. Combining both interventions in **B3** resulted in a Test Macro F1 score higher than **B0** and **B2**, but lower than **B1**.
5. Certain misclassifications, such as 'fear -> sad' and 'neutral -> sad', remained prevalent across all experimental conditions.
