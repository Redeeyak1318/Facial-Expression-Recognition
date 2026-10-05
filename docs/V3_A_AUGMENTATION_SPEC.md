# V3-A Aggressive Augmentation Specification

## Goal
Test whether stronger training-time augmentation reduces the real-world domain gap while keeping the V2-B architecture unchanged.

## Target Architecture & Pipeline
- **Model:** V2-B (Residual CNN, 307,687 params)
- **Input:** 48x48 Grayscale Tensors (`[1, 48, 48]`, values `[0, 1]`).
- **Production Preprocessing:** Unchanged (0% tight crop).

## Augmentation Pipeline
The following transformations will be applied dynamically during training (probability applied per-image):

1. **Random Horizontal Flip** (`p = 0.5`)
   - Standard augmentation to simulate symmetrical expressions.
2. **Random Rotation** (`degrees = [-15, +15]`)
   - Modest bounded rotation to simulate head tilt.
3. **Random Affine** (`degrees = 0`, `translate = (0.1, 0.1)`, `scale = (0.9, 1.1)`)
   - Small translation/scale variation to simulate slight misalignment in bounding box cropping and distance from camera.
4. **Random Perspective** (`distortion_scale = 0.2`, `p = 0.2`)
   - Low distortion probability to simulate non-frontal angles without turning a 48x48 face into unrecognizable noise.
5. **Color Jitter** (`brightness = 0.2`, `contrast = 0.2`)
   - Brightness and contrast variance. (Saturation and Hue are excluded because the data is strictly 1-channel grayscale).
6. **Random Gaussian Blur** (`kernel_size = 3`, `sigma = (0.1, 1.0)`, `p = 0.2`)
   - Mild sharpness variation/blur simulating motion or poor camera focus.
7. **Random Erasing** (`p = 0.1`, `scale = (0.02, 0.1)`, `ratio = (0.3, 3.3)`, `value = 0`)
   - Randomly zeros out a small rectangle to simulate occlusions like glasses frames, shadows, or hair.

## Isolation
The augmentation is implemented via PyTorch `torchvision.transforms` applied dynamically to the tensors inside the isolated `V3_A_augmentation` training loop. The global `src/preprocessing` and production inference logic remains 100% untouched.
