# V3 Final Model Selection

## Final Selected Model: V3-B (Moderate Augmentation)

- **Architecture**: ResidualCNN (V2-B architecture)
- **Parameters**: 307,687
- **Best Epoch**: 26
- **Selection Criterion**: Highest Validation Macro F1
- **Status**: Current Production Model

## Evaluation Results

### Validation Set (Unaugmented)
- **Accuracy**: 0.6083
- **Macro Precision**: 0.5691
- **Macro Recall**: 0.6186
- **Macro F1**: 0.5795

### Official Held-out Test Set
- **Accuracy**: 0.5917
- **Macro Precision**: 0.5489
- **Macro Recall**: 0.5911
- **Macro F1**: 0.5576

*Note: The official test set was strictly held out and untouched during training and model selection. It was evaluated exactly once for the final held-out evaluation of the V3-B model.*

## V3-B Augmentation Pipeline (Moderate)

The V3-B experiment improved validation performance while testing a controlled reduction in augmentation severity. The exact pipeline applied to the training split is:

1. `RandomHorizontalFlip(p=0.5)`
2. `RandomRotation(degrees=[-8°, +8°])`
3. `RandomAffine(degrees=0, translate=(5%, 5%), scale=(0.95, 1.05))`
4. `GaussianBlur(kernel_size=3, sigma=(0.1, 0.8), p=0.10)`
5. `ColorJitter(brightness=0.10, contrast=0.10)`

## Real-World Engineering Validation

A small set of 11 manually labeled real-world photos was used to diagnose preprocessing geometry and observe qualitative robustness in-the-wild.
* **Important Note**: These images were strictly used for qualitative engineering validation and preprocessing tuning (0% tight crop vs 20% padded). They were **NOT** used for training, hyperparameter tuning, or model selection.

The real-world photos were used only as qualitative engineering validation. Their manually assigned expression labels are not treated as reliable ground truth and were not used for training, tuning, or model selection.
