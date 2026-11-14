"""Live Transaction Simulation Engine for Credit Card Anomaly Detection.

Generates realistic synthetic transaction feature vectors grounded in empirical
dataset distributions, scores them in real-time with the Isolation Forest model,
and maintains a controllable simulation stream.

IMPORTANT: This generates simulated records for demonstration and evaluation.
It does not connect to live banking networks or payment processors.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.model_utils import load_trained_model, calculate_anomaly_scores
from src.data_loader import load_raw_dataset


class TransactionSimulator:
    """Generates and evaluates synthetic transaction records."""

    def __init__(self):
        self.model, self.threshold, self.features, _ = load_trained_model()
        self._seed_pools_ready = False
        self.normal_pool: Optional[np.ndarray] = None
        self.fraud_pool: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self._init_empirical_pools()

    def _init_empirical_pools(self):
        """Extract empirical feature distributions from the authentic dataset."""
        try:
            raw_df = load_raw_dataset()
            feat_df = raw_df[self.features]
            self.feature_stds = feat_df.std().values

            if "Class" in raw_df.columns:
                norm_mask = raw_df["Class"] == 0
                fraud_mask = raw_df["Class"] == 1
                self.normal_pool = feat_df[norm_mask].values
                self.fraud_pool = feat_df[fraud_mask].values
            else:
                self.normal_pool = feat_df.values
                self.fraud_pool = feat_df.sample(min(500, len(feat_df))).values

            self._seed_pools_ready = True
        except Exception as exc:
            # Fallback to standard normal distribution if raw file unreadable
            self.normal_pool = np.random.normal(0, 1, size=(500, len(self.features)))
            self.fraud_pool = np.random.normal(2, 2, size=(100, len(self.features)))
            self.feature_stds = np.ones(len(self.features))
            self._seed_pools_ready = True

    def generate_batch(
        self,
        batch_size: int = 5,
        anomaly_ratio: float = 0.15,
        current_seq: int = 0,
        sim_elapsed_seconds: float = 0.0,
    ) -> pd.DataFrame:
        """Generate a batch of complete synthetic transaction feature vectors.
        
        Args:
            batch_size: Number of transactions to generate
            anomaly_ratio: Target proportion of suspicious/fraud-profile transactions
            current_seq: Current global transaction sequence counter
            sim_elapsed_seconds: Relative simulated time elapsed
            
        Returns:
            DataFrame with simulated metadata, features, anomaly scores, and predictions.
        """
        n_fraud = int(np.random.binomial(batch_size, anomaly_ratio))
        n_norm = batch_size - n_fraud

        synthetic_rows = []

        # Generate normal transactions using empirical normal prototype + Gaussian jitter
        if n_norm > 0:
            idx = np.random.choice(len(self.normal_pool), size=n_norm, replace=True)
            seeds = self.normal_pool[idx]
            noise = np.random.normal(0, 0.04, size=seeds.shape) * self.feature_stds
            norm_synthetic = seeds + noise
            synthetic_rows.append(norm_synthetic)

        # Generate suspicious-profile transactions using empirical fraud prototype + jitter
        if n_fraud > 0 and len(self.fraud_pool) > 0:
            idx = np.random.choice(len(self.fraud_pool), size=n_fraud, replace=True)
            seeds = self.fraud_pool[idx]
            noise = np.random.normal(0, 0.05, size=seeds.shape) * self.feature_stds
            fraud_synthetic = seeds + noise
            synthetic_rows.append(fraud_synthetic)

        if not synthetic_rows:
            return pd.DataFrame()

        vectors = np.vstack(synthetic_rows)
        # Shuffle batch so normal and suspicious are intermixed
        p = np.random.permutation(len(vectors))
        vectors = vectors[p]

        # Construct DataFrame with all 30 model features
        df_batch = pd.DataFrame(vectors, columns=self.features)

        # Format realistic simulated Time and Amount
        time_idx = self.features.index("Time") if "Time" in self.features else -1
        amount_idx = self.features.index("Amount") if "Amount" in self.features else -1

        # Keep Time monotonically progressing
        if time_idx >= 0:
            for i in range(len(df_batch)):
                sim_elapsed_seconds += float(np.random.uniform(0.5, 3.5))
                df_batch.iat[i, time_idx] = round(sim_elapsed_seconds, 2)

        # Ensure Amount is strictly positive and rounded to cents
        if amount_idx >= 0:
            df_batch["Amount"] = np.clip(np.abs(df_batch["Amount"]), 1.0, 5000.0).round(2)

        # Compute legitimate model anomaly scores: -model.decision_function(X)
        scores = calculate_anomaly_scores(self.model, df_batch, self.features)
        preds = np.where(scores >= self.threshold, "Suspicious", "Normal")

        # Assign simulated transaction IDs
        sim_ids = [f"SIM-{current_seq + i + 1:05d}" for i in range(len(df_batch))]

        meta_df = pd.DataFrame({
            "Simulated_ID": sim_ids,
            "Time": df_batch["Time"] if "Time" in df_batch else [round(sim_elapsed_seconds + i, 2) for i in range(len(df_batch))],
            "Amount": df_batch["Amount"] if "Amount" in df_batch else [50.0] * len(df_batch),
            "Anomaly_Score": scores,
            "Prediction": preds,
            "Status": preds,
        })

        # Combine metadata with full 30 feature vector for inspector panel
        result_df = pd.concat([meta_df, df_batch], axis=1)
        # Remove duplicate column names if Time/Amount appear twice
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]

        return result_df
