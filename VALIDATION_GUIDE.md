# Real-World Image Validation Guide

This directory (`validation_samples/`) is designed as a local-only engineering validation set to test the end-to-end V2-B inference and face detection pipeline on real-world user photographs.

## Important Privacy Notice
**USER PHOTOS MUST NEVER BE COMMITTED TO GIT.** 
The `validation_samples/` directory is explicitly ignored in `.gitignore` to prevent accidental commits of personal or sensitive images. Do NOT save uploaded or user images anywhere else in this repository.

## How to use
1. Place your test images directly into the `validation_samples/` folder.
2. Run the validation harness using the provided script:
   ```bash
   python scripts/validate_real_images.py --input_dir validation_samples
   ```
3. A `validation_results.json` file will be generated in `validation_samples/` containing detailed per-image metrics.

## Recommended Image Variety
To ensure robust validation of the pipeline, it is recommended to include the following variety of images:
- Standard frontal face portraits
- Faces with a slight left or right pose
- Images taken under different lighting conditions
- Subjects with and without glasses
- Images of varying resolutions and aspect ratios
- Various facial expressions (happy, sad, neutral, etc.)
- At least one image containing multiple people (to verify the largest face selection)
- At least one image containing no face at all (to verify error handling)

## Note
This is exclusively an **engineering validation set** to ensure the pipeline behaves robustly. It is **NOT** a new training or test set for tuning model weights or architectural hyperparameters.
