# V3-A Validation Results Report

## Experiment Configuration
- **Model**: ResidualCNN (V2-B) (307,687 params)
- **Epochs**: 24/30 (Early Stopping: True)
- **Best Epoch**: 19
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Optimizer**: Adam

## Augmentation Pipeline
- RandomHorizontalFlip (p=0.5)
- RandomRotation (degrees=15)
- RandomAffine (translate=10%, scale=10%)
- RandomPerspective (p=0.2)
- ColorJitter (brightness/contrast = 0.2)
- GaussianBlur (p=0.2)
- RandomErasing (p=0.1)

## Validation Performance vs Baseline
- **V2-B Baseline Macro F1 (Validation)**: 0.5489
- **V3-A Macro F1 (Validation)**: 0.5252
- **Delta**: -0.0237

### Per-Class Metrics (Validation)
| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| angry | 0.4449 | 0.4950 | 0.4686 | 400 |
| disgust | 0.3833 | 0.5227 | 0.4423 | 44 |
| fear | 0.3986 | 0.2878 | 0.3343 | 410 |
| happy | 0.8562 | 0.7091 | 0.7758 | 722 |
| neutral | 0.5831 | 0.4456 | 0.5051 | 496 |
| sad | 0.3989 | 0.5839 | 0.4739 | 483 |
| surprise | 0.6150 | 0.7508 | 0.6761 | 317 |

**Overall Accuracy**: 0.5543
