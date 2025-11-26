
FeverCast360 — Quick Run Guide
===============================

Command example:

python ml_model.py   --input tmp_input_for_ml.csv   --models_dir models   --output_dir outputs   --region_col Region   --label_outbreak outbreak_label   --label_type fever_type   --threshold 0.5

Optional: add --use_xgboost (requires xgboost installed)

CSV Expectations
----------------
- Must contain columns: Region, outbreak_label, fever_type
- Features = all other columns (numeric and/or categorical)
- outbreak_label: 0/1 binary
- fever_type: string labels like Dengue, Typhoid, Viral

Outputs
-------
- models/outbreak_model.pkl, models/fever_type_model.pkl
- outputs/predicted_output.csv with columns: Region, P_Outbreak, Fever_Type, P_Type, Severity_Index
- outputs/metrics_*.txt and plots in outputs/plots/

Notes
-----
- Stage 2 is trained primarily on rows where outbreak_label == 1. If too few rows, it will fallback to all rows where fever type is available.
- Decision threshold for Stage 1 is configurable via --threshold.
- Severity Index = P(Outbreak) * P(Fever_Type)
