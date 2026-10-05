# V3-B Real-World Engineering Validation Report

*Engineering validation only — not an official accuracy/generalization claim.*
*Generated: 2026-10-05 20:24:13*

## 1. Experiment Context

| Parameter | Value |
|-----------|-------|
| Checkpoint | V3-B best_model.pth |
| Best Epoch | 26 |
| Best Val Macro F1 | 0.5794768004579861 |
| V2-B Val Macro F1 (baseline) | 0.5489 |
| V3-A Val Macro F1 | 0.5252 |
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

| Metric | V2-B | V3-A | V3-B |
|--------|------|------|------|
| Correct / Total | 5 / 9 | 6 / 9 | 5 / 9 |
| Accuracy | 55.6% | 66.7% | 55.6% |
| Average Confidence | 0.7816 | 0.4562 | 0.5959 |

## 4. Prediction Distribution (V3-B)

| Expression | Count |
|------------|-------|
| angry | 1 |
| disgust | 0 |
| fear | 0 |
| happy | 1 |
| neutral | 6 |
| sad | 1 |
| surprise | 1 |

## 5. Per-Image Comparison (V2-B vs V3-A vs V3-B)

| Filename | True Label | V2-B Pred | V3-A Pred | V3-B Pred | V3-B Change vs V2-B |
|----------|------------|-----------|-----------|-----------|---------------------|
| `IMG_20250609_122532.jpg` | happy | sad (0.87) ✗ | angry (0.37) ✗ | **surprise** (0.35) ✗ | = Incorrect |
| `IMG_20250829_140834.jpg` | neutral | neutral (0.57) ✓ | neutral (0.64) ✓ | **neutral** (0.85) ✓ | = Correct |
| `IMG_20260410_152237.jpg` | neutral | neutral (0.77) ✓ | neutral (0.47) ✓ | **neutral** (0.41) ✓ | = Correct |
| `IMG-20241029-WA0009(1).jpg` | neutral | neutral (0.99) ✓ | neutral (0.78) ✓ | **neutral** (0.86) ✓ | = Correct |
| `IMG-20241128-WA0036.jpg` | happy | neutral (0.61) ✗ | happy (0.27) ✓ | **happy** (0.58) ✓ | ⬆ Improved |
| `IMG-20241207-WA0021.jpg` | sad | neutral (0.83) ✗ | neutral (0.45) ✗ | **neutral** (0.63) ✗ | = Incorrect |
| `IMG-20250830-WA0048.jpg` | neutral | neutral (1.00) ✓ | neutral (0.60) ✓ | **neutral** (0.74) ✓ | = Correct |
| `IMG20200413144756.jpg` | sad | sad (0.79) ✓ | sad (0.30) ✓ | **neutral** (0.51) ✗ | ⬇ Regressed |
| `IMG_20260430_211801_813.webp` | fear | neutral (0.50) ✗ | sad (0.36) ✗ | **sad** (0.42) ✗ | = Incorrect |

### Detector-Only Cases

| Filename | True Label | Detection | V3-B Pred (if any) |
|----------|------------|-----------|-------------------|
| `IMG_20250903_074442_988.jpg` | no_face | True Negative ✓ | — |
| `IMG_20240926_133913.jpg` | no_face | False Positive | angry (62.37%) |

## 6. Change Summary (V3-B vs V2-B Baseline)

| Category | Count |
|----------|-------|
| ⬆ Improved | 1 |
| ⬇ Regressed | 1 |
| = Maintained Correct | 4 |
| = Maintained Incorrect | 3 |

## 7. Limitations

- This evaluation uses only 11 manually labeled images (9 faces + 2 no-face).
- The sample is far too small for statistical significance or generalization claims.
- Manual labels may not perfectly reflect ground-truth psychological expressions.
- This is an engineering diagnostic to compare preprocessing and augmentation behavior.
- The official test set has NOT been used.
