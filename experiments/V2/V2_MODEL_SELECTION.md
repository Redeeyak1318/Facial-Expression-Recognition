# V2 Model Selection

## Selected V2 Candidate
**V2_B_residual_cnn**

## Selection Criterion
Highest Validation Macro F1

## Selected Validation Macro F1
0.5489

## Selected Checkpoint
`experiments/V2/V2_B_residual_cnn/checkpoint/best_model.pth`

## Held-Out Test Results
- Accuracy = 0.5585
- Macro Precision = 0.5521
- Macro Recall = 0.5355
- Macro F1 = 0.5408

The test set was used only for final held-out evaluation and was not used to select the V2 configuration.

## Architectural Trade-Offs
- V2-B uses 307,687 parameters.
- V2-A uses 390,599 parameters.
- V2-B has fewer parameters while achieving higher Macro F1.

It is important to note that V2-B did not outperform V2-A on every metric. V2-A achieved higher Macro Precision (0.6115 vs 0.5521), while V2-B achieved higher Accuracy, Macro Recall, and Macro F1.
