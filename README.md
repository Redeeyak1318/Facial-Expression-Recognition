# Facial Expression Recognition using a CNN

## Project Overview
This is a 7-class facial-expression classification project utilizing a custom Convolutional Neural Network (CNN) trained on grayscale facial images. The current production model is **V3-B Moderate Augmentation** using the **ResidualCNN (V2-B)** architecture.

Classes:
- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

## Dataset
- original train = 28,709
- original test = 7,178
- total = 35,887
- 90/10 stratified train/validation split from the original training set
- seed = 42
- test set kept untouched

Final split:
- train = 25,837
- validation = 2,872
- test = 7,178

## Production Preprocessing & Inference
- Haar cascade face detection (`haarcascade_frontalface_default.xml`)
- selection of the largest detected face
- 0% tight bounding-box crop
- conversion to grayscale
- resize to 48×48 using bilinear interpolation
- normalization to [0,1]
- lazy/path-based image loading during training

## Production Model Architecture (V3-B)
The current production model uses the **ResidualCNN** architecture (introduced in V2-B).
- Architecture: Residual blocks with skip connections, batch normalization, and dropout.
- Parameters: 307,687

## Final Model Selection (V3-B)
- **Selected experiment**: V3-B Moderate Augmentation
- **Selection criterion**: Highest validation Macro F1
- **Best validation Macro F1**: 0.5795
- **Checkpoint**: `experiments/V3/V3_B_moderate_augmentation/checkpoint/best_model.pth`

Held-out V3-B test metrics:
- Accuracy = 0.5917
- Macro Precision = 0.5489
- Macro Recall = 0.5911
- Macro F1 = 0.5576

The official test set was used only for final held-out evaluation and was not used to select the model configuration. A separate set of 11 manually labeled real-world photos was used for qualitative engineering validation only and were not used for training, tuning, or model selection.

For detailed information on the V3-B model selection, augmentation pipeline, and engineering validation, see:
[docs/V3_FINAL_MODEL_SELECTION.md](docs/V3_FINAL_MODEL_SELECTION.md)

---

## Historical Experiments (V1 / B0-B3 / V2)

### Historical Model Architecture (LightweightCNN)
The original `LightweightCNN` architecture used in early V1 (B0-B3) experiments:
Input 48×48×1 → Conv2d 1→32 → ReLU → MaxPool 2×2 → Conv2d 32→64 → ReLU → MaxPool 2×2 → Conv2d 64→128 → ReLU → MaxPool 2×2 → Global Average Pooling → Dropout 0.30 → Linear 128→7
Parameters: 93,575

### Historical Training Configuration (V1)
- Adam, LR = 0.001, batch size = 64, max epochs = 30, early stopping patience = 5

### Historical Controlled Experiments (V1)
| Experiment | Best Val Macro F1 | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 |
|------------|-------------------|---------------|----------------------|-------------------|---------------|
| B0 (Baseline) | 0.4230 | 0.5116 | 0.4249 | 0.4294 | 0.4225 |
| B1 (Class Weights) | 0.4948 | 0.5038 | 0.4769 | 0.4862 | 0.4738 |
| B2 (Augmentation) | 0.4139 | 0.5104 | 0.4205 | 0.4205 | 0.4172 |
| B3 (Weights + Augmentation) | 0.4497 | 0.4967 | 0.4376 | 0.4493 | 0.4391 |

*(Subsequent V2 experiments introduced the ResidualCNN and V3 introduced robust augmentations. V3-B superseded B1.)*

## Repository Structure
- `src/data`: Dataset split, statistics, config, and dataloader logic.
- `src/preprocessing`: Transform pipelines and image handling.
- `src/models`: Neural network architecture definitions (`cnn.py`, `cnn_v2b.py`).
- `src/training`: The core PyTorch training loop (`loop.py`).
- `src/inference`: Production predictor and face detection (`predictor.py`, `face_detector.py`).
- `experiments/`: Saved experiment checkpoints, configs, history, and evaluations.
- `docs/`: Project documentation and planning.

Note: The raw dataset directory (`CNN_DataSet`) is intentionally kept local and not committed to source control due to size limits.

## Reproducibility
To inspect or reproduce the environment:
1. activate the virtual environment: `.\.venv\Scripts\Activate.ps1`
2. verify PyTorch environment: `python verify_pytorch.py`
3. verify the data pipeline: `python verify_pipeline.py`

## Results Artifacts
Final analytical reports and consolidated tables for historical experiments:
`experiments/consolidated_analysis/`
