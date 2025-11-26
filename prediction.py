#!/usr/bin/env python3
"""
FeverCast360 — Two‑Stage ML Pipeline

Stage 1: Logistic Regression to predict outbreak probability (binary: 0/1)
Stage 2: RandomForest (default) or XGBoost (optional) to classify fever type (multiclass)

Outputs
-------
1) models/outbreak_model.pkl
2) models/fever_type_model.pkl
3) outputs/predicted_output.csv  -> [Region, P_Outbreak, Fever_Type, P_Type, Severity_Index]
4) outputs/metrics_stage1.txt, outputs/metrics_stage2.txt
5) outputs/plots: ROC curve, Confusion Matrix, Feature Importances
6) outputs/readme.txt (quick run guide & assumptions)

Assumptions (customizable via CLI)
----------------------------------
- Region column name: "Region"
- Outbreak label column: "outbreak_label"  (0/1)
- Fever type label column: "fever_type"    (values like Dengue/Typhoid/Viral)
- Features = all columns except region + labels
- Threshold for Stage 1 to trigger Stage 2: 0.5 (customizable)

Example
-------
python ml_model.py \
  --input data/preprocessed_data.csv \
  --models_dir models \
  --output_dir outputs \
  --region_col Region \
  --label_outbreak outbreak_label \
  --label_type fever_type \
  --threshold 0.5 \
  --use_xgboost  # optional

"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


@dataclass
class Config:
    input_path: str
    models_dir: str = "models"
    output_dir: str = "outputs"
    region_col: str = "Region"
    label_outbreak: str = "outbreak_label"
    label_type: str = "fever_type"
    threshold: float = 0.5
    use_xgboost: bool = False
    test_size: float = 0.2
    random_state: int = 42


def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "plots"), exist_ok=True)


def split_features(df: pd.DataFrame, cfg: Config) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature lists (exclude region + labels)."""
    drop_cols = {cfg.region_col, cfg.label_outbreak, cfg.label_type}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre


def train_stage1_logreg(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    cfg: Config,
) -> Tuple[Pipeline, dict]:
    model = LogisticRegression(max_iter=200, n_jobs=None, solver="lbfgs")
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", model),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    pipe.fit(X_train, y_train)

    # Evaluation
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= cfg.threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }

    # Plots
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("Stage 1 ROC Curve (LogReg)")
    roc_path = os.path.join(cfg.output_dir, "plots", "stage1_roc.png")
    fig.savefig(roc_path, bbox_inches="tight", dpi=160)
    plt.close(fig)

    # Confusion matrix at decision threshold
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title("Stage 1 Confusion Matrix (threshold={:.2f})".format(cfg.threshold))
    cm_path = os.path.join(cfg.output_dir, "plots", "stage1_confusion.png")
    fig.savefig(cm_path, bbox_inches="tight", dpi=160)
    plt.close(fig)

    return pipe, metrics


def train_stage2_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    cfg: Config,
) -> Tuple[Pipeline, dict, Optional[np.ndarray]]:
    if cfg.use_xgboost and _HAS_XGB:
        clf = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=cfg.random_state,
            tree_method="hist",
        )
        model_name = "XGBoost"
    else:
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=cfg.random_state,
        )
        model_name = "RandomForest"

    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", clf),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = None
    try:
        y_proba = pipe.predict_proba(X_test)
    except Exception:
        pass

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "report": classification_report(y_test, y_pred, output_dict=False),
        "model": model_name,
    }

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title(f"Stage 2 Confusion Matrix ({model_name})")
    cm_path = os.path.join(cfg.output_dir, "plots", "stage2_confusion.png")
    fig.savefig(cm_path, bbox_inches="tight", dpi=160)
    plt.close(fig)

    # Feature importances (if available)
    importances = None
    try:
        # Extract feature names after preprocessing
        pre: ColumnTransformer = pipe.named_steps["pre"]
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        num_cols = pre.transformers_[0][2]  # from build_preprocessor order
        cat_cols = pre.transformers_[1][2]
        cat_feature_names = []
        if hasattr(ohe, "get_feature_names_out"):
            cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names = list(num_cols) + list(cat_feature_names)

        base_clf = pipe.named_steps["clf"]
        if hasattr(base_clf, "feature_importances_"):
            importances = np.asarray(base_clf.feature_importances_)
            # Plot
            idx = np.argsort(importances)[-20:][::-1]  # top 20
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1])
            ax.set_title(f"Stage 2 Feature Importances ({model_name}) — Top 20")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            fi_path = os.path.join(cfg.output_dir, "plots", "stage2_feature_importances.png")
            fig.savefig(fi_path, bbox_inches="tight", dpi=160)
            plt.close(fig)
    except Exception:
        pass

    return pipe, metrics, importances


def save_metrics(metrics: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        # Keep human‑readable first
        for k, v in metrics.items():
            if k == "report":
                f.write("\nCLASSIFICATION REPORT\n")
                f.write(str(v))
                f.write("\n")
            else:
                f.write(f"{k}: {v}\n")
        f.write("\n\nJSON:\n")
        json.dump(metrics, f, indent=2)


def write_quick_readme(cfg: Config, path: str) -> None:
    text = f"""
FeverCast360 — Quick Run Guide
===============================

Command example:

python ml_model.py \
  --input {cfg.input_path} \
  --models_dir {cfg.models_dir} \
  --output_dir {cfg.output_dir} \
  --region_col {cfg.region_col} \
  --label_outbreak {cfg.label_outbreak} \
  --label_type {cfg.label_type} \
  --threshold {cfg.threshold}

Optional: add --use_xgboost (requires xgboost installed)

CSV Expectations
----------------
- Must contain columns: {cfg.region_col}, {cfg.label_outbreak}, {cfg.label_type}
- Features = all other columns (numeric and/or categorical)
- {cfg.label_outbreak}: 0/1 binary
- {cfg.label_type}: string labels like Dengue, Typhoid, Viral

Outputs
-------
- models/outbreak_model.pkl, models/fever_type_model.pkl
- outputs/predicted_output.csv with columns: Region, P_Outbreak, Fever_Type, P_Type, Severity_Index
- outputs/metrics_*.txt and plots in outputs/plots/

Notes
-----
- Stage 2 is trained primarily on rows where {cfg.label_outbreak} == 1. If too few rows, it will fallback to all rows where fever type is available.
- Decision threshold for Stage 1 is configurable via --threshold.
- Severity Index = P(Outbreak) * P(Fever_Type)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def make_final_predictions(
    df: pd.DataFrame,
    region_col: str,
    pipe_stage1: Pipeline,
    pipe_stage2: Pipeline,
    threshold: float,
) -> pd.DataFrame:
    # Stage 1 probabilities for all rows
    p_outbreak = pipe_stage1.predict_proba(df)[:, 1]

    # For rows above threshold, run Stage 2 to get class probs + top class
    will_classify = p_outbreak >= threshold
    fever_type = np.array(["None"] * len(df), dtype=object)
    p_type = np.zeros(len(df), dtype=float)

    if will_classify.any():
        proba_mat = pipe_stage2.predict_proba(df[will_classify])
        classes = pipe_stage2.named_steps["clf"].classes_
        top_idx = np.argmax(proba_mat, axis=1)
        fever_type[will_classify] = classes[top_idx]
        p_type[will_classify] = proba_mat[np.arange(proba_mat.shape[0]), top_idx]

    severity = p_outbreak * p_type

    out = pd.DataFrame({
        "Region": df[region_col].astype(str).values,
        "P_Outbreak": p_outbreak,
        "Fever_Type": fever_type,
        "P_Type": p_type,
        "Severity_Index": severity,
    })
    return out


def main():
    parser = argparse.ArgumentParser(description="FeverCast360 ML Pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV (preprocessed)")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--region_col", default="Region")
    parser.add_argument("--label_outbreak", default="outbreak_label")
    parser.add_argument("--label_type", default="fever_type")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_xgboost", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    cfg = Config(
        input_path=args.input,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        region_col=args.region_col,
        label_outbreak=args.label_outbreak,
        label_type=args.label_type,
        threshold=args.threshold,
        use_xgboost=args.use_xgboost,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    ensure_dirs(cfg)

    # Load data
    df = pd.read_csv(cfg.input_path)
    missing_cols = [c for c in [cfg.region_col, cfg.label_outbreak, cfg.label_type] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV missing required columns: {missing_cols}")

    # Feature discovery
    num_cols, cat_cols = split_features(df, cfg)
    if not num_cols and not cat_cols:
        raise ValueError("No feature columns detected. Ensure your CSV has columns besides region/labels.")

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Stage 1 — train on all rows with outbreak label available
    df_stage1 = df.dropna(subset=[cfg.label_outbreak]).copy()
    X1 = df_stage1[num_cols + cat_cols]
    y1 = df_stage1[cfg.label_outbreak].astype(int)

    pipe_stage1, m1 = train_stage1_logreg(X1, y1, preprocessor, cfg)
    joblib.dump(pipe_stage1, os.path.join(cfg.models_dir, "outbreak_model.pkl"))
    save_metrics(m1, os.path.join(cfg.output_dir, "metrics_stage1.txt"))

    # Stage 2 — prefer rows with outbreak_label == 1; fallback to all fever_type rows
    df_ft = df[df[cfg.label_outbreak] == 1] if df[cfg.label_outbreak].notna().any() else df
    df_ft = df_ft.dropna(subset=[cfg.label_type]).copy()

    if len(df_ft) < 20:
        # Fallback if too small
        df_ft = df.dropna(subset=[cfg.label_type]).copy()

    if len(df_ft) < 10:
        raise ValueError("Not enough labeled rows to train Stage 2 classifier.")

    X2 = df_ft[num_cols + cat_cols]
    y2 = df_ft[cfg.label_type].astype(str)

    # Build a separate preprocessor so OHE categories are fit on stage2 data distribution
    preprocessor2 = build_preprocessor(num_cols, cat_cols)
    pipe_stage2, m2, _ = train_stage2_classifier(X2, y2, preprocessor2, cfg)
    joblib.dump(pipe_stage2, os.path.join(cfg.models_dir, "fever_type_model.pkl"))
    save_metrics(m2, os.path.join(cfg.output_dir, "metrics_stage2.txt"))

    # Final predictions on the WHOLE dataset
    # Note: we pass the same columns used for training
    X_all = df[num_cols + cat_cols]
    final_df = make_final_predictions(
        df[[cfg.region_col]].join(X_all),
        region_col=cfg.region_col,
        pipe_stage1=pipe_stage1,
        pipe_stage2=pipe_stage2,
        threshold=cfg.threshold,
    )

    out_csv = os.path.join(cfg.output_dir, "predicted_output.csv")
    final_df.to_csv(out_csv, index=False)

    # Quick readme
    write_quick_readme(cfg, os.path.join(cfg.output_dir, "readme.txt"))

    print("\nDone. Files saved to:")
    print(f"  - {os.path.join(cfg.models_dir, 'outbreak_model.pkl')}")
    print(f"  - {os.path.join(cfg.models_dir, 'fever_type_model.pkl')}")
    print(f"  - {out_csv}")
    print(f"  - {os.path.join(cfg.output_dir, 'metrics_stage1.txt')}")
    print(f"  - {os.path.join(cfg.output_dir, 'metrics_stage2.txt')}")
    print(f"  - {os.path.join(cfg.output_dir, 'plots')}")


def run_pipeline(input_csv: str, models_dir="models", output_dir="outputs"):
    """
    A reusable wrapper so Streamlit can call ML pipeline directly.
    Same behavior as running the script from CLI.
    """
    argv_backup = sys.argv
    sys.argv = [
        "prediction.py",
        "--input", input_csv,
        "--models_dir", models_dir,
        "--output_dir", output_dir
    ]
    try:
        main()
    finally:
        sys.argv = argv_backup

