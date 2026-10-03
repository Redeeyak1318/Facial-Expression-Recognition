# Facial Expression Recognition using a Lightweight CNN

## Project Overview
This is a 7-class facial-expression classification project utilizing a custom lightweight Convolutional Neural Network (CNN) trained on 48×48 grayscale facial images. 

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

## Preprocessing
- grayscale
- 48×48
- normalization to [0,1]
- lazy/path-based image loading

## Model Architecture
The exact `LightweightCNN` architecture used in all experiments:

Input 48×48×1
→ Conv2d 1→32, 3×3, padding=1
→ ReLU
→ MaxPool 2×2
→ Conv2d 32→64, 3×3, padding=1
→ ReLU
→ MaxPool 2×2
→ Conv2d 64→128, 3×3, padding=1
→ ReLU
→ MaxPool 2×2
→ Global Average Pooling
→ Dropout 0.30
→ Linear 128→7

Parameters:
93,575

## Training Configuration
- Adam
- LR = 0.001
- batch size = 64
- max epochs = 30
- early stopping patience = 5
- seed = 42
- CPU
- checkpoint selected using validation Macro F1

## Controlled Experiments
The following experimental interventions were tested against the baseline (B0).

| Experiment | Best Val Macro F1 | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 |
|------------|-------------------|---------------|----------------------|-------------------|---------------|
| B0 (Baseline) | 0.4230 | 0.5116 | 0.4249 | 0.4294 | 0.4225 |
| B1 (Class Weights) | 0.4948 | 0.5038 | 0.4769 | 0.4862 | 0.4738 |
| B2 (Augmentation) | 0.4139 | 0.5104 | 0.4205 | 0.4205 | 0.4172 |
| B3 (Weights + Augmentation) | 0.4497 | 0.4967 | 0.4376 | 0.4493 | 0.4391 |

## Final Model Selection
- Selected experiment = B1_class_weighted
- Selection criterion = highest validation Macro F1 among B0-B3
- Best validation Macro F1 = 0.4948
- checkpoint = `experiments/B1_class_weighted/checkpoint/best_model.pth`

Held-out B1 test metrics:
- Accuracy = 0.5038
- Macro Precision = 0.4769
- Macro Recall = 0.4862
- Macro F1 = 0.4738

The test set was used only for final held-out evaluation and was not used to select the model configuration.

## Key Findings
- class weighting substantially improved Macro F1 relative to B0
- disgust F1 improved from 0.0000 in B0 to 0.3806 in B1
- augmentation-only B2 did not improve over B0 under the tested configuration
- B3 was better than B0/B2 but below B1
- recurring confusions included fear→sad and neutral→sad

## Repository Structure
- `src/data`: Dataset split, statistics, config, and dataloader logic.
- `src/preprocessing`: Transform pipelines and image handling.
- `src/models`: Neural network architecture definitions (`cnn.py`).
- `src/training`: The core PyTorch training loop (`loop.py`).
- `experiments/`: Saved experiment checkpoints, configs, history, and evaluations.
- `evaluate_b*.py`: Evaluation scripts corresponding to the experiments.
- `smoke_test*.py`: Smoke tests used for dry-run verification before full training.
- `train_b*.py`: Execution scripts for running experiments.

Note: The raw dataset directory (`CNN_DataSet`) is intentionally kept local and not committed to source control due to size limits.

## Reproducibility
To inspect or reproduce the environment:
1. activate the virtual environment: `.\.venv\Scripts\Activate.ps1`
2. verify PyTorch environment: `python verify_pytorch.py`
3. verify the data pipeline: `python verify_pipeline.py`
4. run the completed experiment scripts only if needed (e.g. `python evaluate_b1.py`). *Do not rerun expensive full training experiments (`train_b*.py`) without backing up previous outputs.*

## Results Artifacts
Final analytical reports and consolidated tables:
`experiments/consolidated_analysis/`
Key files:
- `RESULTS_AND_DISCUSSION.md`
- `FINAL_ERROR_ANALYSIS.md`
- `MODEL_SELECTION.md`
- `consolidated_results.json`, `per_class_f1.csv`
