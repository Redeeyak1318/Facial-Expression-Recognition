# V3 Domain Gap Resolution Plan

## 1. Evidence of Remaining Real-World Domain Gap
Despite isolating the optimal preprocessing strategy (0% tight crop), the V2-B model still exhibits a material domain gap on real-world photographs:
- **Prediction Collapse:** The model's predictions on the engineering validation set collapse almost entirely into `neutral` and `sad`, heavily ignoring active expressions like `happy`, `fear`, and `surprise`.
- **Performance Ceiling:** On the 9-image engineering validation set, accuracy topped out at 55.5%. 
- The model remains highly sensitive to deviations from the specific framing, lighting, and exaggerated expressions present in the original training set.

## 2. Dataset Mismatch vs. Model Capacity
- **Model Capacity (Sufficient):** The current V2-B architecture (ResidualCNN) has over 300,000 parameters and successfully achieved a Macro F1 score of ~0.55 on the strict benchmark test set. It has mathematically proven it has the capacity to learn non-linear expression mappings.
- **Dataset Mismatch (Root Cause):** The domain gap is almost certainly a dataset mismatch issue. Real-world user photographs contain complex lighting, off-angle faces, subtle/natural micro-expressions, and visual occlusions (hair, glasses, shadows) that the highly curated and constrained training dataset fails to adequately cover.

## 3. Candidate V3 Interventions
*(Ordered by priority)*

1. **Aggressive Data Augmentation (Highest Priority)**
   - Introduce severe photometric augmentations (ColorJitter, RandomAdjustSharpness) to simulate real-world lighting variances.
   - Introduce geometric augmentations (RandomAffine, RandomPerspective, RandomRotation) to simulate off-angle or poorly aligned real-world faces.
   - Implement RandomErasing to simulate occlusions.

2. **Training Data Preprocessing Alignment**
   - Ensure that the training dataset itself is tightly cropped (0% padding) to precisely mirror the production inference environment. If the dataset images inherently contain background padding, the model may be learning to rely on it.

3. **Label Smoothing / Loss Function Tuning**
   - Utilize label smoothing or Focal Loss to prevent the model from over-indexing on majority classes (`neutral`/`sad`) and to properly penalize hard-to-classify examples without collapsing predictions.

4. **Architectural Tweaks (Lowest Priority)**
   - Increase network capacity (V3 architecture) or add attention mechanisms. (Only pursue if augmentation fails).

## 4. First Intervention to Test
**Test First: Aggressive Data Augmentation.**
Before introducing architectural complexity or loss tuning, we must evaluate whether severe data augmentation can force the existing V2-B architecture to learn more robust, generalized features. This directly addresses the dataset mismatch by artificially inflating the variance of the training domain.

## 5. Validation Isolation Strategy
- The official benchmark test set (`CNN_DataSet/test/`) will remain strictly isolated for unbiased model evaluation.
- The `validation_samples/` directory will continue to serve exclusively as a real-world engineering diagnostic set. Under no circumstances will these images be added to the training set, preserving their utility as a zero-shot generalization gauge.

## 6. V3 Success Criteria
To justify replacing V2-B, V3 must achieve:
1. **Benchmark Set:** A Validation Macro F1 score that strictly exceeds the V2-B baseline (0.5489).
2. **Real-World Engineering Set:** A manual accuracy that exceeds the 5/9 (55.5%) baseline, accompanied by a diversified prediction distribution (i.e., breaking the `neutral`/`sad` prediction collapse).
