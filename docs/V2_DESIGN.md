# V2 Design Document: Facial Expression Recognition

## 1. V2 Goals
- **Improve balanced classification performance**: Specifically focus on achieving higher overall macro metrics across all 7 emotions.
- **Improve minority-class recognition**: Enhance the model's ability to consistently identify heavily underrepresented classes, particularly "disgust" and "fear".
- **Maintain reasonable computational cost**: Ensure the network remains lightweight enough for efficient CPU execution and rapid training iteration.
- **Support real-image inference**: Establish a pipeline architecture that eventually accepts a user-uploaded image and predicts expressions robustly.

## 2. V1 Baseline
- **Architecture**: `LightweightCNN` (3 Convolutional blocks with Max Pooling → Global Average Pooling → Dropout → Linear)
- **Parameter Count**: 93,575
- **B1 Validation Macro F1**: 0.4948
- **B1 Held-out Test Macro F1**: 0.4738
- **Contract**: Input: 1 × 48 × 48 grayscale -> Output: 7 logits (0: angry, 1: disgust, 2: fear, 3: happy, 4: neutral, 5: sad, 6: surprise)

## 3. Proposed V2 Architecture Candidates

### Candidate A: Enhanced Standard CNN
A stronger CNN utilizing repeated `Conv2d → BatchNorm2d → ReLU` blocks with controlled, systematic channel growth.
- **Expected Advantages**: Batch Normalization will stabilize and accelerate training, reducing internal covariate shift. Standard structural blocks are easy to implement.
- **Expected Disadvantages**: Higher parameter count than V1; diminishing returns on feature extraction without skip connections.
- **Approximate Complexity**: Moderate (~200k - 500k parameters depending on depth).
- **Suitability for CPU Inference**: Excellent. Pure sequential convolutions are highly optimized on CPUs.

### Candidate B: Lightweight Residual CNN (ResNet-style)
A miniature residual network employing skip connections around convolutional blocks.
- **Expected Advantages**: Mitigates the vanishing gradient problem, allowing the network to be trained deeper for richer feature extraction on nuanced facial expressions.
- **Expected Disadvantages**: More complex implementation. Skip connections require strict dimension matching.
- **Approximate Complexity**: Moderate to High (~300k - 1M parameters).
- **Suitability for CPU Inference**: Very Good. Slightly higher memory bandwidth required for residual additions, but fundamentally fast on CPU.

### Candidate C: MobileNet-style Architecture
A lightweight architecture utilizing Depthwise Separable Convolutions.
- **Expected Advantages**: Drastically reduces parameter count and computational operations (FLOPs) while maintaining competitive representational capacity. 
- **Expected Disadvantages**: Can be trickier to tune on extremely small inputs (48x48) compared to large ImageNet-scale inputs.
- **Approximate Complexity**: Very Low (~100k - 200k parameters).
- **Suitability for CPU Inference**: Excellent. Explicitly designed for constrained, edge-device inference.

## 4. V2 Evaluation Strategy
- **Primary metric**: Validation Macro F1
- **Secondary metrics**: Macro Precision, Macro Recall, Accuracy, per-class F1 (especially the disgust class), and inference latency.
- **Test Set**: The original 7,178-image test set remains strictly untouched until a final V2 configuration is formally selected.

## 5. V2 Experiment Rules
Every V2 experiment must strictly adhere to the following controls:
1. **Data Split**: Use the same fixed train/validation split generated from the seed `42` configuration.
2. **Initialization**: Start from a fresh model initialization.
3. **Selection Criterion**: Use Validation Macro F1 for checkpoint selection.
4. **Isolation**: Have its own dedicated experiment directory under `experiments/V2/`.
5. **No Cross-Pollination**: Never load a V1 checkpoint unless explicitly testing transfer learning.
6. **No Data Leakage**: Never use or evaluate against the test set during model development and tuning.

## 6. Inference Requirements
A future inference API must be implemented with the following contract:

`predict_expression(image)`

**Expected output:**
```json
{
    "predicted_class": "...",
    "confidence": 0.00,
    "probabilities": {
       "angry": 0.00,
       "disgust": 0.00,
       "fear": 0.00,
       "happy": 0.00,
       "neutral": 0.00,
       "sad": 0.00,
       "surprise": 0.00
    }
}
```

## 7. Face Detection Requirement
The V1 and V2 CNN pipelines mathematically expect a tightly cropped 48×48 face image. For arbitrary, real-world uploaded photos, a dedicated face detection and cropping stage is strictly required prior to inference processing.
