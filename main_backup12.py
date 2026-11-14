import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# VISUAL THEME (preserved from your original application)
# ============================================================
st.markdown(
    """
    <style>
    :root {
        color-scheme: light !important;
        --navy: #17365d;
        --blue: #245b91;
        --blue-2: #3478b9;
        --ink: #1e293b;
        --muted: #64748b;
        --line: #d8e3ef;
        --surface: #ffffff;
        --page: #f3f7fc;
        --soft-blue: #eaf2fb;
        --green: #16845b;
        --red: #d9574f;
        --amber: #b7791f;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--page) !important;
        color: var(--ink) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(243,247,252,.96) !important;
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4, h5, h6, p, li, label,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stCaptionContainer"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
    }

    h1 {
        color: var(--navy) !important;
        font-weight: 750 !important;
        letter-spacing: -.035em;
    }

    h2, h3 {
        color: var(--navy) !important;
        font-weight: 680 !important;
        letter-spacing: -.02em;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }

    [data-testid="stSidebar"] {
        background: #e6eff9 !important;
        border-right: 1px solid #d2dfed !important;
    }

    [data-testid="stSidebar"] * {
        color: #183957 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-weight: 550;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #35516f !important;
    }

    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 13px !important;
        padding: 17px 18px !important;
        box-shadow: 0 2px 8px rgba(24, 54, 93, .035) !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: var(--navy) !important;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background: var(--blue) !important;
        color: #ffffff !important;
        border: 1px solid var(--blue) !important;
        border-radius: 8px !important;
        min-height: 2.55rem;
        font-weight: 600 !important;
        padding: .45rem 1rem !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #19466f !important;
        border-color: #19466f !important;
        color: #ffffff !important;
    }

    div.stButton > button:focus,
    div.stDownloadButton > button:focus {
        box-shadow: 0 0 0 .18rem rgba(52,120,185,.22) !important;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 1px dashed #9db7d2 !important;
        border-radius: 12px !important;
        padding: 14px !important;
    }

    [data-testid="stFileUploader"] * {
        color: var(--ink) !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #f7faff !important;
    }

    input, textarea, [data-baseweb="select"] > div {
        background: #ffffff !important;
        color: var(--ink) !important;
        border-color: #cbd9e8 !important;
    }

    [data-baseweb="popover"], [data-baseweb="menu"] {
        background: #ffffff !important;
        color: var(--ink) !important;
    }

    [role="option"] {
        color: var(--ink) !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: #ffffff !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    hr {
        border-color: var(--line) !important;
    }

    .eyebrow {
        color: #55718f !important;
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .09em;
        margin-bottom: .35rem;
    }

    .page-subtitle {
        color: #5d7188 !important;
        font-size: 1rem;
        margin-top: -.45rem;
    }

    .section-note {
        color: #6b7d91 !important;
        font-size: .84rem;
        margin-top: -.55rem;
    }

    .panel {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 17px 18px;
        box-shadow: 0 2px 8px rgba(24,54,93,.035);
    }

    .panel-title {
        color: var(--navy) !important;
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: .15rem;
    }

    .panel-subtitle {
        color: var(--muted) !important;
        font-size: .82rem;
    }

    .status-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        background: #e5f5ed;
        color: #14734f !important;
        font-size: .76rem;
        font-weight: 700;
    }

    .demo-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        background: #fff4dc;
        color: #946200 !important;
        font-size: .76rem;
        font-weight: 700;
    }

    .risk-high {
        color: #bd3f37 !important;
        font-weight: 700;
    }

    .risk-low {
        color: #16845b !important;
        font-weight: 700;
    }

    .hero-strip {
        background: #e8f0f9;
        border: 1px solid #d4e1ef;
        border-radius: 13px;
        padding: 15px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PATHS AND MODEL LOADING
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "isolation_forest_improved.pkl"
FALLBACK_MODEL_PATH = BASE_DIR / "models" / "isolation_forest.pkl"
SAVED_RESULTS_PATH = BASE_DIR / "outputs" / "improved_predictions.csv"


@st.cache_resource(show_spinner="Loading trained Isolation Forest model...")
def load_model(model_path: str):
    """Load the saved model once per Streamlit server session."""
    return joblib.load(model_path)


model_file = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
if not model_file.exists():
    st.error(
        "Trained model not found. Keep the models folder beside main.py and "
        "include isolation_forest_improved.pkl."
    )
    st.stop()

try:
    saved_model = load_model(str(model_file))
    if isinstance(saved_model, dict):
        model = saved_model["model"]
        threshold = float(saved_model["threshold"])
        features = list(saved_model["features"])
    else:
        # Legacy model support. A plain IsolationForest does not contain
        # the improved-model threshold, so use its own offset as fallback.
        model = saved_model
        threshold = 0.0
        features = list(getattr(model, "feature_names_in_", []))
        if not features:
            st.error(
                "The fallback model does not contain feature names. "
                "Use isolation_forest_improved.pkl instead."
            )
            st.stop()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()

# This project model was saved with scikit-learn 1.9.1.
# A different version may still load, but can produce compatibility warnings.
try:
    import sklearn
    sklearn_version = sklearn.__version__
except Exception:
    sklearn_version = "Unknown"

# ============================================================
# SESSION STATE
# ============================================================
DEFAULT_STATE = {
    "analysis_results": None,
    "uploaded_filename": None,
    "uploaded_raw_data": None,
    "manual_result": None,
    "manual_history": pd.DataFrame(),
    "detection_mode": "Demo Mode",
    "demo_results": None,
    "performance_results": None,
    "performance_source": None,
    "performance_upload_name": None,
    "performance_raw_data": None,
    "baseline_medians": None,
    "last_analysis_source": None,
}
for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# MODEL PREDICTION
# IMPORTANT: threshold is calibrated on -decision_function().
# Higher anomaly score = more anomalous.
# ============================================================
def validate_features(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("The transaction dataset is empty.")

    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError("Missing required model features: " + ", ".join(missing))

    X = data.loc[:, features].apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Missing or non-numeric values in model features: "
            + ", ".join(bad[:15])
        )

    values = X.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Infinite values found in the model features.")

    return X


def detect_fraud(data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict transactions using the exact score convention used by
    src/improve_model.py:
        anomaly_score = -model.decision_function(X)
        fraud = anomaly_score >= saved_threshold
    """
    X = validate_features(data)
    decision_values = model.decision_function(X)
    anomaly_scores = -np.asarray(decision_values, dtype=float)
    predictions = (anomaly_scores >= threshold).astype(np.int8)

    results = data.copy()
    results["Anomaly Score"] = anomaly_scores
    results["Prediction"] = np.where(predictions == 1, "Suspicious", "Normal")
    return results


# ============================================================
# CACHED INPUT HELPERS
# ============================================================
@st.cache_data(show_spinner=False)
def load_saved_test_predictions(path: str, modified_time: float):
    """Load the compact held-out test predictions once."""
    return pd.read_csv(path)


def get_prediction_counts(data):
    if data is None or data.empty or "Prediction" not in data.columns:
        return 0, 0, 0
    suspicious = int((data["Prediction"] == "Suspicious").sum())
    normal = int((data["Prediction"] == "Normal").sum())
    return len(data), normal, suspicious


def create_demo_data():
    rng = np.random.default_rng(42)
    demo = pd.DataFrame(
        rng.normal(0, 1, size=(500, len(features))),
        columns=features,
    )
    # Give the demo a small, reproducible set of unusual observations.
    if len(demo) >= 15:
        demo.iloc[:15, :] = demo.iloc[:15, :] * 4.0
    return demo


def get_demo_results():
    if st.session_state.demo_results is None:
        st.session_state.demo_results = detect_fraud(create_demo_data())
    return st.session_state.demo_results


def add_manual_history(result):
    if result is None or result.empty:
        return
    current = st.session_state.manual_history
    st.session_state.manual_history = pd.concat(
        [current, result], ignore_index=True
    ).tail(5000)


def get_label_column(data):
    candidates = [
        "Class", "class", "Actual", "actual",
        "Label", "label", "Fraud", "fraud",
    ]
    for column in candidates:
        if column in data.columns:
            return column
    return None


def normalise_labels(series):
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna()
    values = values.loc[valid].astype(int)
    if not set(values.unique()).issubset({0, 1}):
        raise ValueError("Actual labels must be 0 = Normal and 1 = Fraud.")
    if values.nunique() < 2:
        raise ValueError("Evaluation data must contain both Normal and Fraud.")
    return values, valid


def calculate_metrics(y_true, y_pred, anomaly_scores):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)

    if len(np.unique(y_true)) != 2:
        raise ValueError("Evaluation data must contain both classes.")
    if not (
        len(y_true) == len(y_pred) == len(anomaly_scores)
    ):
        raise ValueError("The labels, predictions and scores have different lengths.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, roc_thresholds = roc_curve(y_true, anomaly_scores)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_true, anomaly_scores
    )

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, anomaly_scores),
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "pr_precision": precision_curve,
        "pr_recall": recall_curve,
        "pr_thresholds": pr_thresholds,
        "y_true": y_true,
        "y_pred": y_pred,
        "scores": anomaly_scores,
    }


# ============================================================
# VISUALISATION HELPERS
# ============================================================
def render_detection_charts(results, key_prefix="detection"):
    """Charts use the full result set for counts and a bounded sample for plots."""
    if results is None or results.empty:
        st.info("No detection results are available yet.")
        return

    total, normal, suspicious = get_prediction_counts(results)
    rate = (suspicious / total * 100) if total else 0

    st.markdown("### Detection Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions Analysed", f"{total:,}")
    c2.metric("Normal", f"{normal:,}")
    c3.metric("Suspicious", f"{suspicious:,}")
    c4.metric("Suspicious Rate", f"{rate:.2f}%")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("### Normal vs Suspicious")
        distribution = pd.DataFrame(
            {"Transactions": [normal, suspicious]},
            index=["Normal", "Suspicious"],
        )
        st.bar_chart(distribution, use_container_width=True)

    with right:
        st.markdown("### Anomaly Score Distribution")
        plot_data = results[["Anomaly Score", "Prediction"]].copy()
        if len(plot_data) > 50000:
            plot_data = plot_data.sample(50000, random_state=42)
        fig, ax = plt.subplots(figsize=(8, 4))
        for label in ["Normal", "Suspicious"]:
            values = plot_data.loc[
                plot_data["Prediction"] == label, "Anomaly Score"
            ].to_numpy()
            if len(values):
                ax.hist(
                    values,
                    bins=45,
                    alpha=0.58,
                    label=label,
                    density=False,
                )
        ax.axvline(
            threshold,
            linestyle="--",
            linewidth=1.8,
            label=f"Threshold ({threshold:.4f})",
        )
        ax.set_xlabel("Anomaly Score (higher = more anomalous)")
        ax.set_ylabel("Number of transactions")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("### Transaction Amount Analysis")
        if "Amount" in results.columns:
            amount_data = results[["Amount", "Prediction"]].copy()
            amount_data["Amount"] = pd.to_numeric(
                amount_data["Amount"], errors="coerce"
            )
            amount_data = amount_data.dropna()
            if len(amount_data) > 50000:
                amount_data = amount_data.sample(50000, random_state=42)
            if not amount_data.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                for label in ["Normal", "Suspicious"]:
                    values = amount_data.loc[
                        amount_data["Prediction"] == label, "Amount"
                    ].to_numpy()
                    if len(values):
                        ax.hist(values, bins=40, alpha=0.55, label=label)
                ax.set_xlabel("Transaction Amount")
                ax.set_ylabel("Frequency")
                ax.set_title("Transaction Amount by Prediction")
                ax.legend()
                ax.grid(alpha=0.2)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            st.info("The dataset does not contain an Amount column.")

    with right:
        st.markdown("### Anomaly Score vs Amount")
        if "Amount" in results.columns:
            scatter = results[["Amount", "Anomaly Score", "Prediction"]].copy()
            scatter["Amount"] = pd.to_numeric(scatter["Amount"], errors="coerce")
            scatter = scatter.replace([np.inf, -np.inf], np.nan).dropna()
            if len(scatter) > 5000:
                scatter = scatter.sample(5000, random_state=42)
            fig, ax = plt.subplots(figsize=(8, 4))
            for label in ["Normal", "Suspicious"]:
                subset = scatter[scatter["Prediction"] == label]
                if not subset.empty:
                    ax.scatter(
                        subset["Amount"],
                        subset["Anomaly Score"],
                        s=10,
                        alpha=0.45,
                        label=label,
                    )
            ax.axhline(threshold, linestyle="--", linewidth=1.5, label="Threshold")
            ax.set_xlabel("Transaction Amount")
            ax.set_ylabel("Anomaly Score")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("The dataset does not contain an Amount column.")

    if "Time" in results.columns:
        st.divider()
        st.markdown("### Suspicious Transactions Over Time")
        time_data = results[["Time", "Prediction"]].copy()
        time_data["Time"] = pd.to_numeric(time_data["Time"], errors="coerce")
        time_data = time_data.dropna()
        if not time_data.empty:
            time_data["Time Group"] = (time_data["Time"] // 3600).astype(int)
            trend = (
                time_data.assign(
                    Suspicious=(time_data["Prediction"] == "Suspicious").astype(int)
                )
                .groupby("Time Group")["Suspicious"]
                .sum()
                .rename("Suspicious Transactions")
            )
            st.line_chart(trend, use_container_width=True)
            st.caption(
                "Time is grouped into one-hour intervals. This chart is shown "
                "only when the source dataset has a Time column."
            )

    st.divider()
    st.markdown("### Suspicious Transaction Records")
    suspicious_data = results[results["Prediction"] == "Suspicious"].copy()
    if suspicious_data.empty:
        st.success("No suspicious transactions were detected.")
    else:
        suspicious_total = len(suspicious_data)
        sort_column = (
            "Anomaly Score" if "Anomaly Score" in suspicious_data.columns else None
        )
        if sort_column:
            suspicious_data = suspicious_data.nlargest(100, sort_column)
        else:
            suspicious_data = suspicious_data.head(100)
        st.caption(
            f"Showing up to 100 of {suspicious_total:,} suspicious transactions. "
            "Download contains the complete result set."
        )
        st.dataframe(
            suspicious_data.head(100),
            use_container_width=True,
            hide_index=True,
            height=360,
        )

    # Serialize a result DataFrame only once per session. Streamlit reruns the
    # script on widget interaction; rebuilding a 284k-row CSV on every rerun
    # was a major source of page-switch latency.
    csv_cache = st.session_state.setdefault("csv_payload_cache", {})
    cache_key = id(results)
    if cache_key not in csv_cache:
        csv_cache[cache_key] = results.to_csv(index=False).encode("utf-8")
        # Keep the session cache bounded.
        while len(csv_cache) > 6:
            csv_cache.pop(next(iter(csv_cache)))
    st.download_button(
        "Download Complete Detection Results",
        data=csv_cache[cache_key],
        file_name="fraud_detection_results.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}_download",
    )


def render_performance_charts(metrics):
    cm = metrics["cm"]
    tn, fp, fn, tp = cm.ravel()

    st.markdown("### Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f'{metrics["precision"] * 100:.2f}%')
    c2.metric("Recall", f'{metrics["recall"] * 100:.2f}%')
    c3.metric("F1 Score", f'{metrics["f1"] * 100:.2f}%')
    c4.metric("ROC-AUC", f'{metrics["roc_auc"] * 100:.2f}%')

    st.divider()
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1], ["Normal", "Fraud"])
    ax.set_yticks([0, 1], ["Normal", "Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Negative", f"{tn:,}")
    c2.metric("False Positive", f"{fp:,}")
    c3.metric("False Negative", f"{fn:,}")
    c4.metric("True Positive", f"{tp:,}")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("### ROC Curve")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(
            metrics["fpr"],
            metrics["tpr"],
            linewidth=2,
            label=f'Isolation Forest (AUC = {metrics["roc_auc"]:.4f})',
        )
        ax.plot([0, 1], [0, 1], "--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("### Precision–Recall Curve")
        pr_auc = auc(metrics["pr_recall"], metrics["pr_precision"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(
            metrics["pr_recall"],
            metrics["pr_precision"],
            linewidth=2,
            label=f"PR-AUC = {pr_auc:.4f}",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()
    st.markdown("### Anomaly Score Distribution by Actual Class")
    score_frame = pd.DataFrame(
        {"Actual": metrics["y_true"], "Anomaly Score": metrics["scores"]}
    )
    if len(score_frame) > 50000:
        score_frame = score_frame.sample(50000, random_state=42)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, name in [(0, "Normal"), (1, "Fraud")]:
        vals = score_frame.loc[
            score_frame["Actual"] == label, "Anomaly Score"
        ].to_numpy()
        if len(vals):
            ax.hist(vals, bins=50, alpha=0.55, label=name)
    ax.axvline(threshold, linestyle="--", label=f"Saved threshold = {threshold:.4f}")
    ax.set_xlabel("Anomaly Score (higher = more anomalous)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Metric Interpretation")
    st.write(
        f"""
        - **Precision ({metrics["precision"] * 100:.2f}%):** Of all transactions
          flagged as suspicious, this proportion was actually fraudulent.
        - **Recall ({metrics["recall"] * 100:.2f}%):** Of all actual fraud
          transactions, this proportion was detected.
        - **F1 Score ({metrics["f1"] * 100:.2f}%):** Harmonic mean of precision
          and recall.
        - **ROC-AUC ({metrics["roc_auc"] * 100:.2f}%):** Measures how well the
          anomaly scores rank fraudulent transactions above normal transactions.
        """
    )


# ============================================================
# SIDEBAR - DETECTION MODE AND NAVIGATION
# ============================================================
def reset_navigation():
    st.session_state["main_navigation"] = "Dashboard"


def navigate_to(page_name):
    """Set the navigation target before Streamlit recreates the radio widget."""
    st.session_state["main_navigation"] = page_name


with st.sidebar:
    st.markdown("## 💳 Fraud Detection")
    st.caption("Credit Card Anomaly Detection System")
    st.divider()
    st.markdown("### Detection Mode")

    st.radio(
        "Select detection mode",
        ["Demo Mode", "Dataset Analysis Mode", "Manual Transaction Mode"],
        key="detection_mode",
        on_change=reset_navigation,
    )

    st.divider()
    st.markdown("### NAVIGATION")

    if st.session_state.detection_mode == "Demo Mode":
        page_options = ["Dashboard", "Fraud Detection", "Model Performance"]
    elif st.session_state.detection_mode == "Dataset Analysis Mode":
        page_options = [
            "Dashboard", "Upload Dataset", "Fraud Detection", "Model Performance"
        ]
    else:
        page_options = [
            "Dashboard", "Upload Dataset", "Manual Detection",
            "Fraud Detection", "Model Performance"
        ]

    if st.session_state.get("main_navigation") not in page_options:
        st.session_state["main_navigation"] = "Dashboard"

    page = st.radio("Go to", page_options, key="main_navigation", label_visibility="collapsed")


# ============================================================
# APPLICATION HEADER
# ============================================================
st.markdown(
    '<div class="eyebrow">MACHINE LEARNING • FINANCIAL SECURITY</div>',
    unsafe_allow_html=True,
)
st.title("Credit Card Fraud Detection")
st.markdown(
    '<div class="page-subtitle">Identify unusual transaction patterns using '
    'an unsupervised machine learning model.</div>',
    unsafe_allow_html=True,
)
st.write("")


# ============================================================
# DASHBOARD PAGE
# ============================================================
if page == "Dashboard":
    st.subheader("Dashboard Overview")

    if st.session_state.detection_mode == "Demo Mode":
        st.markdown(
            '<span class="demo-pill">ILLUSTRATIVE DEMO DATA</span>',
            unsafe_allow_html=True,
        )
        st.info(
            "Demo Mode uses generated sample transactions. These results are "
            "illustrative and are not real customer transactions."
        )
        dashboard_results = get_demo_results()

    elif st.session_state.detection_mode == "Dataset Analysis Mode":
        dashboard_results = st.session_state.analysis_results
        if dashboard_results is None:
            st.info("Upload a dataset and run Fraud Detection to populate this dashboard.")
            st.button(
                "Go to Upload Dataset",
                type="primary",
                on_click=navigate_to,
                args=("Upload Dataset",),
            )
            st.stop()

    else:
        dashboard_results = st.session_state.manual_history
        if dashboard_results is None or dashboard_results.empty:
            st.info("No manual transactions have been analysed yet.")
            st.button(
                "Go to Manual Detection",
                type="primary",
                on_click=navigate_to,
                args=("Manual Detection",),
            )
            st.stop()

    render_detection_charts(dashboard_results, key_prefix="dashboard")


# ============================================================
# UPLOAD DATASET PAGE (existing workflow retained)
# ============================================================
elif page == "Upload Dataset":
    st.subheader("Upload Transaction Dataset")
    st.markdown(
        "Upload a CSV file containing transaction records for fraud and anomaly analysis."
    )
    st.info(
        "Your dataset must contain all feature columns required by the trained "
        "Isolation Forest model. If it contains a Class column, it can also be "
        "used for model evaluation."
    )

    uploaded_file = st.file_uploader(
        "Choose your CSV dataset",
        type=["csv"],
        key="transaction_dataset_uploader",
    )

    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            st.markdown("### Dataset Preview")
            st.write(
                f"Rows: **{uploaded_data.shape[0]:,}** | "
                f"Columns: **{uploaded_data.shape[1]:,}**"
            )
            st.dataframe(
                uploaded_data.head(10),
                use_container_width=True,
                hide_index=True,
            )

            missing_features = [
                feature for feature in features if feature not in uploaded_data.columns
            ]
            if missing_features:
                st.error("This dataset is not compatible with the trained model.")
                st.write("Missing required feature columns:")
                st.code(", ".join(missing_features))
            else:
                try:
                    validate_features(uploaded_data)
                    st.success(
                        "Compatible dataset detected! All required model features "
                        "are available and numeric."
                    )
                    if st.button(
                        "Use This Dataset",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state.uploaded_raw_data = uploaded_data.copy()
                        st.session_state.uploaded_filename = uploaded_file.name
                        st.session_state.analysis_results = None
                        st.session_state.last_analysis_source = None
                        st.session_state.baseline_medians = None
                        st.success("Dataset uploaded and saved successfully.")
                        st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        except Exception as exc:
            st.error(f"Unable to read the uploaded CSV file: {exc}")

    if st.session_state.uploaded_raw_data is not None:
        st.divider()
        st.markdown("### Currently Loaded Dataset")
        current_data = st.session_state.uploaded_raw_data
        st.success(
            f"Loaded dataset: {st.session_state.uploaded_filename or 'Uploaded dataset'}"
        )
        st.write(
            f"**Records:** {len(current_data):,} | "
            f"**Columns:** {len(current_data.columns):,}"
        )
        st.dataframe(current_data.head(5), use_container_width=True, hide_index=True)

        if st.button("Remove Uploaded Dataset", key="remove_uploaded_dataset"):
            st.session_state.uploaded_raw_data = None
            st.session_state.uploaded_filename = None
            st.session_state.analysis_results = None
            st.session_state.baseline_medians = None
            st.session_state.last_analysis_source = None
            st.success("Uploaded dataset removed.")
            st.rerun()
    else:
        st.warning("No dataset is currently loaded. Upload a compatible CSV to continue.")


# ============================================================
# FRAUD DETECTION PAGE
# ============================================================
elif page == "Fraud Detection":
    st.subheader("Fraud Detection Analysis")
    st.markdown(
        "Analyse transactions using the trained Isolation Forest model. "
        "Results and charts are retained while you navigate between pages."
    )
    st.info(
        f"The saved threshold is {threshold:.6f}. The application uses the "
        "same score direction as the training script: higher anomaly score "
        "means more suspicious."
    )

    mode = st.session_state.detection_mode
    selected_data = None

    if mode == "Demo Mode":
        st.markdown(
            '<span class="demo-pill">ILLUSTRATIVE DEMO DATA</span>',
            unsafe_allow_html=True,
        )
        st.caption("Demo mode generates a reproducible sample of 500 transactions.")
        selected_data = create_demo_data()
        if st.button(
            "Run Demo Fraud Detection",
            type="primary",
            use_container_width=True,
            key="run_demo_detection",
        ):
            with st.spinner("Analysing demo transactions..."):
                st.session_state.demo_results = detect_fraud(selected_data)
            st.success("Demo detection completed.")
            st.rerun()

        results_to_show = st.session_state.demo_results

    elif mode == "Dataset Analysis Mode":
        source_option = st.radio(
            "Select dataset source",
            ["Use Uploaded Dataset", "Upload New Dataset"],
            horizontal=True,
            key="dataset_fraud_source",
        )

        if source_option == "Use Uploaded Dataset":
            if st.session_state.uploaded_raw_data is None:
                st.warning("No dataset uploaded. Please visit Upload Dataset first.")
            else:
                selected_data = st.session_state.uploaded_raw_data
                st.success(
                    f"Using: {st.session_state.uploaded_filename or 'Uploaded dataset'}"
                )
        else:
            fraud_file = st.file_uploader(
                "Upload CSV for fraud detection",
                type=["csv"],
                key="fraud_detection_uploader",
            )
            if fraud_file is not None:
                try:
                    selected_data = pd.read_csv(fraud_file)
                    st.caption(f"Selected file: {fraud_file.name}")
                except Exception as exc:
                    st.error(f"Unable to read CSV file: {exc}")

        results_to_show = st.session_state.analysis_results

        if selected_data is not None:
            st.write(f"Dataset contains **{len(selected_data):,} transactions**.")
            st.dataframe(
                selected_data.head(5),
                use_container_width=True,
                hide_index=True,
            )
            if st.button(
                "Run Fraud Detection",
                type="primary",
                use_container_width=True,
                key="run_dataset_detection",
            ):
                try:
                    with st.spinner(
                        f"Analysing {len(selected_data):,} transactions. "
                        "This runs once; switching pages will not rerun the model."
                    ):
                        results = detect_fraud(selected_data)
                        st.session_state.analysis_results = results
                        st.session_state.last_analysis_source = (
                            st.session_state.uploaded_filename
                            if source_option == "Use Uploaded Dataset"
                            else "Fraud Detection upload"
                        )
                        st.session_state.performance_results = None
                    st.success("Fraud detection completed successfully.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Fraud detection failed: {exc}")

    else:
        st.info(
            "Manual Transaction Mode displays the transactions entered on the "
            "Manual Detection page."
        )
        results_to_show = st.session_state.manual_history
        if results_to_show is None or results_to_show.empty:
            st.warning("No manual detection history yet. Analyse a transaction first.")
        else:
            st.caption("Showing your manual transaction history and its analytics.")

    if results_to_show is not None and not results_to_show.empty:
        st.divider()
        render_detection_charts(results_to_show, key_prefix="fraud_detection")
    elif mode == "Demo Mode":
        st.info("Select Run Demo Fraud Detection to generate the charts and results.")
    elif mode == "Dataset Analysis Mode":
        st.info("Run Fraud Detection to generate the charts and results.")


# ============================================================
# MANUAL DETECTION PAGE
# ============================================================
elif page == "Manual Detection":
    st.subheader("Manual Transaction Detection")
    st.markdown(
        "Enter transaction details and identify unusual transactions using "
        "the trained Isolation Forest model."
    )

    baseline = st.session_state.uploaded_raw_data
    medians = None
    compatible_baseline = False

    if (
        isinstance(baseline, pd.DataFrame)
        and not baseline.empty
        and all(feature in baseline.columns for feature in features)
    ):
        if st.session_state.baseline_medians is None:
            numeric_baseline = baseline[features].apply(pd.to_numeric, errors="coerce")
            st.session_state.baseline_medians = numeric_baseline.median()
        candidate = st.session_state.baseline_medians
        if candidate.notna().all() and np.isfinite(candidate.to_numpy(dtype=float)).all():
            compatible_baseline = True
            medians = candidate

    entry_type = st.radio(
        "Select transaction entry type",
        ["Single Transaction", "Multiple Transactions"],
        horizontal=True,
        key="manual_entry_type",
    )
    entry_mode = st.radio(
        "Choose input method",
        ["Quick Entry — 5 fields", "Full Entry — all model features"],
        horizontal=True,
        key="manual_entry_method",
    )

    quick_features = [
        feature for feature in ["Time", "Amount", "V1", "V2", "V3"]
        if feature in features
    ]
    if len(quick_features) < 5:
        quick_features = features[:5]

    quick_blocked = entry_mode.startswith("Quick") and not compatible_baseline

    if entry_mode.startswith("Quick"):
        if not compatible_baseline:
            st.warning(
                "Quick Entry requires a compatible uploaded dataset. The model "
                f"requires all {len(features)} features; the remaining features "
                "are filled using medians from your uploaded dataset. Upload a "
                "dataset on the Upload Dataset page, or select Full Entry."
            )
        else:
            st.success(
                f"Baseline loaded from {st.session_state.uploaded_filename}. "
                f"The remaining {len(features) - len(quick_features)} features "
                "will use dataset medians."
            )
    else:
        st.info(
            f"Full Entry allows you to enter all {len(features)} model features. "
            "No baseline dataset is required."
        )

    def build_manual_row(values):
        if entry_mode.startswith("Quick"):
            if not compatible_baseline:
                raise ValueError("Upload a compatible dataset before using Quick Entry.")
            row = {feature: float(medians[feature]) for feature in features}
            for feature in quick_features:
                row[feature] = float(values[feature])
        else:
            row = {feature: float(values[feature]) for feature in features}
        return row

    if entry_type == "Single Transaction":
        with st.form("single_manual_transaction_form"):
            st.markdown("### Enter Transaction Details")
            entered = {}
            visible_features = quick_features if entry_mode.startswith("Quick") else features
            cols = st.columns(3 if not entry_mode.startswith("Quick") else 2)
            for index, feature in enumerate(visible_features):
                default = 100.0 if feature == "Amount" else 0.0
                with cols[index % len(cols)]:
                    entered[feature] = st.number_input(
                        feature,
                        value=float(default),
                        step=0.01,
                        format="%.6f",
                        key=f"manual_single_{entry_mode}_{feature}",
                    )
            submitted = st.form_submit_button(
                "Analyse Transaction",
                type="primary",
                use_container_width=True,
                disabled=quick_blocked,
            )

        if submitted:
            try:
                row = pd.DataFrame([build_manual_row(entered)], columns=features)
                result = detect_fraud(row)
                st.session_state.manual_result = result
                add_manual_history(result)
                st.success("Transaction analysed successfully.")
            except Exception as exc:
                st.error(f"Unable to analyse transaction: {exc}")

    else:
        st.markdown("### Enter Multiple Transactions")
        count = st.number_input(
            "Number of transactions",
            min_value=2,
            max_value=100,
            value=5,
            step=1,
            key="manual_batch_size",
        )
        visible_features = quick_features if entry_mode.startswith("Quick") else features
        batch_data = pd.DataFrame(
            [
                {feature: (100.0 if feature == "Amount" else 0.0)
                 for feature in visible_features}
                for _ in range(int(count))
            ]
        )
        edited = st.data_editor(
            batch_data,
            num_rows="fixed",
            use_container_width=True,
            key="manual_batch_editor",
        )
        if st.button(
            "Analyse All Transactions",
            type="primary",
            use_container_width=True,
            disabled=quick_blocked,
            key="analyse_manual_batch",
        ):
            try:
                rows = []
                for _, item in edited.iterrows():
                    rows.append(
                        build_manual_row(
                            {feature: float(item[feature]) for feature in visible_features}
                        )
                    )
                result = detect_fraud(pd.DataFrame(rows, columns=features))
                st.session_state.manual_result = result
                add_manual_history(result)
                st.success(f"{len(result)} transactions analysed successfully.")
            except Exception as exc:
                st.error(f"Unable to analyse transactions: {exc}")

    if st.session_state.manual_result is not None:
        st.divider()
        st.markdown("### Latest Transaction Results")
        render_detection_charts(
            st.session_state.manual_result,
            key_prefix="manual_latest",
        )

    st.divider()
    st.markdown("### Manual Detection Activity History")
    history = st.session_state.manual_history
    if history is not None and not history.empty:
        render_detection_charts(history, key_prefix="manual_history")
        if st.button("Clear Manual Detection History", key="clear_manual_history"):
            st.session_state.manual_history = pd.DataFrame()
            st.session_state.manual_result = None
            st.rerun()
    else:
        st.info("No manual transaction activity recorded yet.")


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================
elif page == "Model Performance":
    st.subheader("Model Performance")
    st.markdown(
        "Evaluate the trained Isolation Forest model using actual transaction labels."
    )
    st.info(
        "Precision, recall, F1-score and ROC-AUC require actual labels. "
        "The saved test-prediction file is used as the default benchmark, "
        "so opening this page does not rerun the model on all 284,807 rows."
    )

    available_sources = ["Saved Test Predictions"]
    analysed_data = st.session_state.analysis_results
    analysed_label = (
        get_label_column(analysed_data)
        if isinstance(analysed_data, pd.DataFrame)
        else None
    )
    if analysed_data is not None and analysed_label:
        available_sources.insert(0, "Previously Analysed Labelled Dataset")

    source = st.radio(
        "Select evaluation source",
        available_sources + ["Upload Labelled Dataset"],
        horizontal=True,
        key="performance_source_selector",
    )

    metrics = None

    if source == "Saved Test Predictions":
        if SAVED_RESULTS_PATH.exists():
            try:
                saved_predictions = load_saved_test_predictions(
                    str(SAVED_RESULTS_PATH),
                    SAVED_RESULTS_PATH.stat().st_mtime,
                )
                required = {"Actual", "Predicted", "Anomaly_Score"}
                if not required.issubset(saved_predictions.columns):
                    st.error(
                        "The saved prediction CSV must contain Actual, Predicted "
                        "and Anomaly_Score columns."
                    )
                else:
                    y = pd.to_numeric(saved_predictions["Actual"], errors="coerce")
                    pred = pd.to_numeric(saved_predictions["Predicted"], errors="coerce")
                    scores = pd.to_numeric(
                        saved_predictions["Anomaly_Score"], errors="coerce"
                    )
                    valid = y.notna() & pred.notna() & scores.notna()
                    y = y[valid].astype(int)
                    pred = pred[valid].astype(int)
                    scores = scores[valid].astype(float)
                    metrics = calculate_metrics(y, pred, scores)
                    st.success(
                        f"Loaded held-out test predictions: {len(y):,} transactions. "
                        "No model prediction rerun was required."
                    )
            except Exception as exc:
                st.error(f"Unable to load saved test predictions: {exc}")
        else:
            st.warning(
                "outputs/improved_predictions.csv was not found. Upload a labelled "
                "dataset or place the saved test prediction file in outputs/."
            )

    elif source == "Previously Analysed Labelled Dataset":
        try:
            y, valid = normalise_labels(analysed_data[analysed_label])
            aligned = analysed_data.loc[valid].copy()
            if "Anomaly Score" not in aligned.columns:
                raise ValueError("The analysed dataset has no saved anomaly scores.")
            scores = pd.to_numeric(aligned["Anomaly Score"], errors="coerce")
            valid_scores = scores.notna()
            aligned = aligned.loc[valid_scores]
            y = y.loc[valid_scores]
            scores = scores.loc[valid_scores]
            pred = (scores.to_numpy() >= threshold).astype(int)
            metrics = calculate_metrics(y, pred, scores)
            st.success(
                f"Evaluated existing analysis results: {len(y):,} transactions. "
                "Existing anomaly scores were reused; the model was not rerun."
            )
        except Exception as exc:
            st.error(f"Unable to evaluate analysed dataset: {exc}")

    else:
        performance_file = st.file_uploader(
            "Upload labelled CSV (must contain all model features and Class/Actual label)",
            type=["csv"],
            key="performance_dataset_uploader",
        )
        if performance_file is not None:
            try:
                perf_data = pd.read_csv(performance_file)
                label_col = get_label_column(perf_data)
                if label_col is None:
                    st.error("No Class, Actual, Label or Fraud label column was found.")
                else:
                    st.write(
                        f"Uploaded evaluation data: **{len(perf_data):,} rows** "
                        f"and **{len(perf_data.columns):,} columns**."
                    )
                    if st.button(
                        "Evaluate Uploaded Dataset",
                        type="primary",
                        use_container_width=True,
                        key="evaluate_uploaded_performance",
                    ):
                        with st.spinner(
                            "Scoring the labelled evaluation dataset once..."
                        ):
                            y, valid = normalise_labels(perf_data[label_col])
                            clean = perf_data.loc[valid].copy()
                            X = validate_features(clean)
                            anomaly_scores = -model.decision_function(X)
                            y_pred = (anomaly_scores >= threshold).astype(int)
                            st.session_state.performance_results = calculate_metrics(
                                y, y_pred, anomaly_scores
                            )
                            st.session_state.performance_upload_name = performance_file.name
                        st.success("Model evaluation completed.")
                        st.rerun()
            except Exception as exc:
                st.error(f"Unable to prepare evaluation dataset: {exc}")

        if st.session_state.performance_results is not None:
            metrics = st.session_state.performance_results
            st.success(
                "Showing the latest uploaded-dataset evaluation: "
                f"{st.session_state.performance_upload_name or 'Uploaded dataset'}."
            )

    if metrics is not None:
        st.divider()
        render_performance_charts(metrics)
    else:
        st.info("Select an available evaluation source to display performance metrics.")

    st.divider()
    st.markdown("### Model Information")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Algorithm**

            Isolation Forest

            **Learning Type**

            Unsupervised Machine Learning

            **Purpose**

            Detect unusual transaction patterns.
            """
        )
    with c2:
        st.markdown(
            f"""
            **Number of Model Features**

            {len(features)}

            **Decision Threshold**

            {threshold:.6f}

            **Score Direction**

            Higher anomaly score = more suspicious

            **Model File**

            {model_file.name}

            **scikit-learn Runtime**

            {sklearn_version}
            """
        )


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Credit Card Fraud Detection System | Isolation Forest Machine Learning Model"
)
