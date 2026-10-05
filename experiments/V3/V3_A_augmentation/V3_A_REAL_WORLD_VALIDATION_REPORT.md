# V3-A Real-World Engineering Validation Report

*Engineering validation only — not an official accuracy/generalization claim.*
*Generated: 2026-10-05 01:36:50*

## 1. Experiment Context

| Parameter | Value |
|-----------|-------|
| Checkpoint | V3-A best_model.pth |
| Best Epoch | 19 |
| Best Val Macro F1 | 0.5251729631896371 |
| V2-B Val Macro F1 (baseline) | 0.5489 |
| Preprocessing | Haar → 0% tight crop → grayscale → 48×48 → /255 |

## 2. Detection Summary

| Metric | Count |
|--------|-------|
| Total Images | 11 |
| Faces Detected | 10 |
| No Face Detected | 1 |
| False Positive (Detector) | 1 |
| Errors | 0 |

## 3. Classification Summary (Manually Labeled Faces)

| Metric | V2-B (0% crop) | V3-A |
|--------|----------------|------|
| Correct / Total | 5 / 9 | 6 / 9 |
| Accuracy | 55.6% | 66.7% |
| Average Confidence | 0.7816 | 0.4562 |

## 4. Prediction Distribution

| Expression | V3-A Count |
|------------|------------|
| angry | 2 |
| disgust | 0 |
| fear | 0 |
| happy | 1 |
| neutral | 5 |
| sad | 2 |
| surprise | 0 |

## 5. Per-Image V2-B vs V3-A Comparison

| Filename | True Label | V2-B Pred (Conf) | V3-A Pred (Conf) | V2-B | V3-A | Change |
|----------|------------|------------------|------------------|------|------|--------|
| `IMG_20250609_122532.jpg` | happy | sad (87.04%) | angry (36.83%) | ✗ | ✗ | = Incorrect |
| `IMG_20250829_140834.jpg` | neutral | neutral (56.73%) | neutral (63.70%) | ✓ | ✓ | = Correct |
| `IMG_20260410_152237.jpg` | neutral | neutral (76.55%) | neutral (47.02%) | ✓ | ✓ | = Correct |
| `IMG-20241029-WA0009(1).jpg` | neutral | neutral (99.15%) | neutral (78.17%) | ✓ | ✓ | = Correct |
| `IMG-20241128-WA0036.jpg` | happy | neutral (61.14%) | happy (26.81%) | ✗ | ✓ | ⬆ Improved |
| `IMG-20241207-WA0021.jpg` | sad | neutral (82.69%) | neutral (44.98%) | ✗ | ✗ | = Incorrect |
| `IMG-20250830-WA0048.jpg` | neutral | neutral (99.74%) | neutral (60.49%) | ✓ | ✓ | = Correct |
| `IMG20200413144756.jpg` | sad | sad (79.02%) | sad (29.69%) | ✓ | ✓ | = Correct |
| `IMG_20260430_211801_813.webp` | fear | neutral (49.65%) | sad (35.61%) | ✗ | ✗ | = Incorrect |

### Detector-Only Cases

| Filename | True Label | Detection | V3-A Pred (if any) |
|----------|------------|-----------|-------------------|
| `IMG_20250903_074442_988.jpg` | no_face | True Negative ✓ | — |
| `IMG_20240926_133913.jpg` | no_face | False Positive | angry (32.87%) |

## 6. Change Summary

| Category | Count |
|----------|-------|
| ⬆ Improved | 1 |
| ⬇ Regressed | 0 |
| = Maintained Correct | 5 |
| = Maintained Incorrect | 3 |

## 7. Limitations

- This evaluation uses only 11 manually labeled images (9 faces + 2 no-face).
- The sample is far too small for statistical significance or generalization claims.
- Manual labels may not perfectly reflect ground-truth psychological expressions.
- This is an engineering diagnostic to compare V2-B and V3-A preprocessing behavior.
- The official test set has NOT been used.
