# Repository Audit

| File/Path | Status | Reason |
|-----------|--------|--------|
| `.gitignore` | KEEP | Necessary for ignoring `.venv`, `CNN_DataSet`, and cache directories |
| `.venv/` | KEEP | Local python virtual environment (ignored by Git) |
| `CNN_DataSet/` | KEEP | Raw dataset (ignored by Git) |
| `README.md` | KEEP | Primary project documentation |
| `requirements.txt` | KEEP | Project dependency list |
| `src/` | KEEP | Core source code (preprocessing, models, data, training) |
| `experiments/` | KEEP | All checkpoint artifacts, history logs, evaluation results, and analysis |
| `train_b0.py` - `train_b3.py` | KEEP | Entry points for training experiments |
| `evaluate_b0.py` - `evaluate_b3.py` | KEEP | Scripts used to evaluate the held-out test set |
| `smoke_test.py`, `smoke_test_b1.py`, etc. | REVIEW | Initial pipeline verification scripts. These can be kept as integration tests but are somewhat redundant now. |
| `verify_cnn.py`, `verify_pipeline.py`, `verify_pytorch.py` | REVIEW | Setup verification scripts. Good for onboarding but not strictly necessary for final runtime. |
| `analyze_experiments.py` | KEEP | Source logic for building the consolidated tables |
| `generate_docs.py`, `generate_final_report.py` | KEEP | Reproducibility scripts for the final markdown documents |

*Note: No experimental data, models, or core python scripts have been marked for removal.*
