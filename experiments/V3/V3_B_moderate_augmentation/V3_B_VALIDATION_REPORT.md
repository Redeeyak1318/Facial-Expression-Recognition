# V3-B Validation Results Report

## Experiment Configuration
- **Model**: ResidualCNN (V2-B) (307,687 params)
- **Epochs**: 30/30 (Early Stopping: False)
- **Best Epoch**: 26
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Optimizer**: Adam

## Augmentation Pipeline (Moderate)
- RandomHorizontalFlip (p=0.5)
- RandomRotation (degrees=8)
- RandomAffine (translate=5%, scale=5%)
- GaussianBlur (p=0.10)
- ColorJitter (brightness/contrast = 0.10)

## Validation Performance vs Baseline
- **V2-B Baseline Macro F1 (Validation)**: 0.5489
- **V3-B Macro F1 (Validation)**: 0.5795
- **Delta**: +0.0306

### Per-Class Metrics (Validation)
| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| angry | 0.5216 | 0.5125 | 0.5170 | 400 |
| disgust | 0.4342 | 0.7500 | 0.5500 | 44 |
| fear | 0.5066 | 0.2805 | 0.3611 | 410 |
| happy | 0.8670 | 0.7673 | 0.8141 | 722 |
| neutral | 0.5637 | 0.6331 | 0.5964 | 496 |
| sad | 0.4640 | 0.5197 | 0.4902 | 483 |
| surprise | 0.6264 | 0.8675 | 0.7275 | 317 |

**Overall Accuracy**: 0.6083
**Macro Precision**: 0.5691
**Macro Recall**: 0.6186
**Macro F1**: 0.5795

## Top Confusion Pairs (True -> Predicted)
- **fear** confused as **sad**: 95 times
- **sad** confused as **neutral**: 91 times
- **fear** confused as **surprise**: 82 times
- **neutral** confused as **sad**: 77 times
- **angry** confused as **sad**: 66 times
