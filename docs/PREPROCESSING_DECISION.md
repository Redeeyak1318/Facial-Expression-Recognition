# Inference Preprocessing Decision

## Final Decision
- **Keep 0% tight face cropping for production inference.**
- Do NOT implement square normalization.
- Do NOT test additional crop-padding variants.
- Do NOT retrain V2-B.

## Diagnostic Evidence
- **20% padding:** 2/9 correct, avg confidence 0.5767
- **0% tight crop:** 5/9 correct, avg confidence 0.7686
- **square_norm:** 5/9 correct, avg confidence 0.7686

The `square_norm` strategy produced identical predictions and confidences to the 0% tight crop on all 9 samples. This confirmed that the OpenCV Haar cascade naturally returns square bounding boxes. The significant performance improvement seen when moving from 20% to 0% padding is entirely attributable to the removal of background context noise, not due to any geometric correction.

The 0% tight crop safely isolates the facial features the model was trained on without injecting artificial background signal.
