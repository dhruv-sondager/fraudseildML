"""Data loading and caching layer for Credit Card Fraud Detection.

Automatically discovers existing dataset files, manages precomputed scoring cache,
and handles retrieval of individual transaction details without manual CSV uploads.
"""

import time
from pathlib import Path
from typing import Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.model_utils import (
    load_trained_model,
    predict_anomalies,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Dataset candidate locations in order of preference
DATASET_CANDIDATES = [
    DATA_DIR / "creditcard.csv",
    DATA_DIR / "processed" / "creditcard_cleaned.csv",
    DATA_DIR / "processed" / "test_data.csv",
]

PRECOMPUTED_SCORES_PATH = OUTPUTS_DIR / "full_dataset_scored.joblib"
TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "improved_predictions.csv"


def find_existing_dataset() -> Path:
    """Find the primary transaction dataset in the project directory."""
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No transaction dataset found in data/ directory. "
        "Expected data/creditcard.csv or data/processed/creditcard_cleaned.csv."
    )


def load_raw_dataset(dataset_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the full raw transaction dataset."""
    target_path = dataset_path or find_existing_dataset()
    df = pd.read_csv(target_path)
    return df


def load_or_compute_full_analysis(
    force_recompute: bool = False
) -> Tuple[pd.DataFrame, float, str]:
    """Load precomputed detection results or compute them from the dataset.
    
    Returns:
        df_scored: DataFrame with Transaction_ID, Time, Amount, Anomaly_Score, Prediction, Class
        elapsed_sec: Time taken in seconds
        source_desc: Source description ("Cached Analysis" or "Fresh Model Run")
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not force_recompute and PRECOMPUTED_SCORES_PATH.exists():
        t0 = time.perf_counter()
        df_scored = joblib.load(str(PRECOMPUTED_SCORES_PATH))
        elapsed = time.perf_counter() - t0
        return df_scored, elapsed, "Cached Analysis (Fast Load)"

    # Compute freshly from raw dataset
    dataset_file = find_existing_dataset()
    raw_df = load_raw_dataset(dataset_file)
    model, threshold, features, _ = load_trained_model()

    t0 = time.perf_counter()
    scores, binary_preds, text_preds = predict_anomalies(
        model, raw_df, features, threshold
    )
    elapsed = time.perf_counter() - t0

    tx_ids = [f"TXN-{i + 1:06d}" for i in range(len(raw_df))]
    
    df_scored = pd.DataFrame({
        "Transaction_ID": tx_ids,
        "Time": raw_df["Time"].values if "Time" in raw_df.columns else np.arange(len(raw_df)),
        "Amount": raw_df["Amount"].values if "Amount" in raw_df.columns else 0.0,
        "Anomaly_Score": scores,
        "Prediction": text_preds,
        "Class": raw_df["Class"].values if "Class" in raw_df.columns else 0,
    })

    # Save to disk for fast reload
    joblib.dump(df_scored, str(PRECOMPUTED_SCORES_PATH), compress=3)

    return df_scored, elapsed, f"Computed from {dataset_file.name}"


def load_test_evaluation_data() -> pd.DataFrame:
    """Load the compact held-out test predictions (outputs/improved_predictions.csv).
    
    Contains 56,746 rows with Actual, Predicted, and Anomaly_Score.
    """
    if not TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Test evaluation file not found at {TEST_PREDICTIONS_PATH}."
        )
    return pd.read_csv(TEST_PREDICTIONS_PATH)


def get_single_transaction_features(tx_index: int) -> pd.Series:
    """Retrieve the full 30 feature values for a specific transaction by index."""
    raw_df = load_raw_dataset()
    if 0 <= tx_index < len(raw_df):
        return raw_df.iloc[tx_index]
    raise IndexError(f"Transaction index {tx_index} out of range [0, {len(raw_df)-1}].")
