"""Model utility functions for Credit Card Fraud Detection.

Handles loading of trained Isolation Forest model, threshold calibration,
anomaly score computation (-decision_function), and evaluation metrics calculation.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    auc,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "isolation_forest_improved.pkl"
FALLBACK_MODEL_PATH = BASE_DIR / "models" / "isolation_forest.pkl"


def load_trained_model(custom_path: Path = None) -> Tuple[Any, float, List[str], Path]:
    """Load the trained Isolation Forest model and its calibrated threshold.
    
    Returns:
        model: Trained Isolation Forest estimator
        threshold: Calibrated anomaly score decision threshold
        features: Expected feature names in correct order
        model_file: Path to loaded model file
    """
    model_file = custom_path or (MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_file}. Please check the models directory."
        )

    saved_obj = joblib.load(str(model_file))

    if isinstance(saved_obj, dict):
        model = saved_obj.get("model")
        threshold = float(saved_obj.get("threshold", 0.113000))
        features = list(saved_obj.get("features", []))
    else:
        model = saved_obj
        threshold = 0.0
        features = list(getattr(model, "feature_names_in_", []))

    if not features:
        # Standard features for Kaggle Credit Card Fraud dataset
        features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    return model, threshold, features, model_file


def validate_input_features(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """Validate that required features are present and numeric."""
    if df is None or df.empty:
        raise ValueError("Input transaction DataFrame is empty.")

    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required model features ({len(missing)}): {', '.join(missing[:5])}"
            + ("..." if len(missing) > 5 else "")
        )

    X = df.loc[:, expected_features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"Non-numeric or NaN values found in features: {', '.join(bad_cols[:5])}"
        )

    return X


def calculate_anomaly_scores(model: Any, df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Compute anomaly scores following the exact convention in src/improve_model.py:
    
    anomaly_score = -model.decision_function(X)
    
    Higher score indicates a more anomalous / suspicious transaction.
    """
    X = validate_input_features(df, features)
    decision_vals = model.decision_function(X)
    scores = -np.asarray(decision_vals, dtype=np.float64)
    return scores


def predict_anomalies(
    model: Any,
    df: pd.DataFrame,
    features: List[str],
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score transactions and return anomaly scores, binary flags, and text labels.
    
    Returns:
        scores: Anomaly scores (float)
        binary_pred: 1 for Suspicious, 0 for Normal
        text_pred: "Suspicious" or "Normal"
    """
    scores = calculate_anomaly_scores(model, df, features)
    binary_pred = (scores >= threshold).astype(np.int8)
    text_pred = np.where(binary_pred == 1, "Suspicious", "Normal")
    return scores, binary_pred, text_pred


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray
) -> Dict[str, Any]:
    """Compute comprehensive performance metrics and curves on labeled test data."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)

    if len(y_true) != len(y_pred) or len(y_true) != len(anomaly_scores):
        raise ValueError("Lengths of y_true, y_pred, and anomaly_scores must match.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, roc_thresholds = roc_curve(y_true, anomaly_scores)
    roc_auc = float(roc_auc_score(y_true, anomaly_scores))

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, anomaly_scores)
    pr_auc = float(auc(pr_recall, pr_precision))

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Fraud"],
        output_dict=True,
        zero_division=0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cm": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": fpr,
        "tpr": tpr,
        "pr_precision": pr_precision,
        "pr_recall": pr_recall,
        "y_true": y_true,
        "y_pred": y_pred,
        "scores": anomaly_scores,
        "report_dict": report_dict,
        "total_test": len(y_true),
        "total_fraud": int(np.sum(y_true == 1)),
        "total_normal": int(np.sum(y_true == 0)),
    }
