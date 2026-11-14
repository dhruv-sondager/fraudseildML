import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
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
# VISUAL THEME
# A fixed light-blue palette is deliberately used so the app
# remains readable even when the viewer's OS/browser is in dark mode.
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
    [data-testid="stHeader"] { background: rgba(243,247,252,.96) !important; }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1500px; }

    h1, h2, h3, h4, h5, h6, p, li, label,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stCaptionContainer"],
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: var(--ink) !important;
    }
    h1 { color: var(--navy) !important; font-weight: 750 !important; letter-spacing: -.035em; }
    h2, h3 { color: var(--navy) !important; font-weight: 680 !important; letter-spacing: -.02em; }
    [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

    [data-testid="stSidebar"] {
        background: #e6eff9 !important;
        border-right: 1px solid #d2dfed !important;
    }
    [data-testid="stSidebar"] * { color: #183957 !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { font-weight: 550; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #35516f !important; }

    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 13px !important;
        padding: 17px 18px !important;
        box-shadow: 0 2px 8px rgba(24, 54, 93, .035) !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] { color: var(--navy) !important; }

    div.stButton > button, div.stDownloadButton > button {
        background: var(--blue) !important;
        color: #ffffff !important;
        border: 1px solid var(--blue) !important;
        border-radius: 8px !important;
        min-height: 2.55rem;
        font-weight: 600 !important;
        padding: .45rem 1rem !important;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: #19466f !important; border-color: #19466f !important;
        color: #ffffff !important;
    }
    div.stButton > button:focus, div.stDownloadButton > button:focus {
        box-shadow: 0 0 0 .18rem rgba(52,120,185,.22) !important;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff !important; border: 1px dashed #9db7d2 !important;
        border-radius: 12px !important; padding: 14px !important;
    }
    [data-testid="stFileUploader"] * { color: var(--ink) !important; }
    [data-testid="stFileUploaderDropzone"] { background: #f7faff !important; }

    input, textarea, [data-baseweb="select"] > div {
        background: #ffffff !important; color: var(--ink) !important;
        border-color: #cbd9e8 !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"] {
        background: #ffffff !important; color: var(--ink) !important;
    }
    [role="option"] { color: var(--ink) !important; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background: #ffffff !important; border: 1px solid var(--line) !important;
        border-radius: 10px !important; overflow: hidden;
    }
    [data-testid="stAlert"] { border-radius: 10px !important; }
    hr { border-color: var(--line) !important; }

    .eyebrow { color: #55718f !important; font-size: .78rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: .09em; margin-bottom: .35rem; }
    .page-subtitle { color: #5d7188 !important; font-size: 1rem; margin-top: -.45rem; }
    .section-note { color: #6b7d91 !important; font-size: .84rem; margin-top: -.55rem; }
    .panel { background: #ffffff; border: 1px solid var(--line); border-radius: 13px;
        padding: 17px 18px; box-shadow: 0 2px 8px rgba(24,54,93,.035); }
    .panel-title { color: var(--navy) !important; font-size: 1.08rem; font-weight: 700; margin-bottom: .15rem; }
    .panel-subtitle { color: var(--muted) !important; font-size: .82rem; }
    .status-pill { display: inline-block; padding: 4px 10px; border-radius: 20px;
        background: #e5f5ed; color: #14734f !important; font-size: .76rem; font-weight: 700; }
    .demo-pill { display: inline-block; padding: 4px 10px; border-radius: 20px;
        background: #fff4dc; color: #946200 !important; font-size: .76rem; font-weight: 700; }
    .risk-high { color: #bd3f37 !important; font-weight: 700; }
    .risk-low { color: #16845b !important; font-weight: 700; }
    .hero-strip { background: #e8f0f9; border: 1px solid #d4e1ef; border-radius: 13px;
        padding: 15px 18px; }
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


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


model_file = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
if not model_file.exists():
    st.error(
        "Trained model file was not found. Keep the models folder beside main.py "
        "and include isolation_forest_improved.pkl or isolation_forest.pkl."
    )
    st.stop()

try:
    saved_model = load_model(str(model_file))
    model = saved_model["model"]
    threshold = float(saved_model["threshold"])
    features = list(saved_model["features"])
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "uploaded_raw_data" not in st.session_state:
    st.session_state.uploaded_raw_data = None


# ============================================================
# MODEL PREDICTION FUNCTION - ORIGINAL ISOLATION FOREST LOGIC
# ============================================================
def detect_fraud(data: pd.DataFrame) -> pd.DataFrame:
    """Score transactions with the saved Isolation Forest model."""
    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError("Missing required model features: " + ", ".join(missing))

    X = data.loc[:, features].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        bad_columns = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Missing or non-numeric values found in model features: "
            + ", ".join(bad_columns[:12])
        )
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("The model features contain infinite values.")

    scores = -model.decision_function(X)
    predictions = (scores >= threshold).astype(int)

    results = data.copy()
    results["Anomaly Score"] = scores
    results["Prediction"] = np.where(predictions == 1, "Suspicious", "Normal")
    return results


def find_column(frame: pd.DataFrame, candidates):
    """Return the first matching column, ignoring case and spaces."""
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def get_dashboard_data():
    """Return live predictions when available; otherwise return labelled demo data."""
    live = st.session_state.get("analysis_results")
    if isinstance(live, pd.DataFrame) and not live.empty:
        return live.copy(), False
    return make_demo_data(), True


def make_demo_data() -> pd.DataFrame:
    """Small illustrative sample; never used for model training or evaluation."""
    return pd.DataFrame(
        {
            "Transaction ID": ["DEMO-1048", "DEMO-1047", "DEMO-1046", "DEMO-1045", "DEMO-1044", "DEMO-1043"],
            "Transaction Time": pd.to_datetime([
                "2026-11-12 14:42", "2026-11-12 14:35", "2026-11-12 14:21",
                "2026-11-12 13:58", "2026-11-12 13:41", "2026-11-12 13:20"
            ]),
            "Amount": [18450.00, 1250.00, 24990.00, 3450.00, 799.00, 12600.00],
            "Anomaly Score": [0.91, 0.08, 0.84, 0.13, 0.06, 0.76],
            "Prediction": ["Suspicious", "Normal", "Suspicious", "Normal", "Normal", "Suspicious"],
        }
    )


def transaction_summary(results: pd.DataFrame):
    prediction_col = find_column(results, ["Prediction", "Predicted"])
    if prediction_col is None:
        return len(results), 0, len(results)
    labels = results[prediction_col].astype(str).str.lower()
    suspicious = labels.isin(["suspicious", "fraud", "1", "true", "anomaly"]).sum()
    total = len(results)
    return total, total - int(suspicious), int(suspicious)


def format_amount(value):
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def make_chart_style(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="#52677f", labelsize=9)
    ax.xaxis.label.set_color("#52677f")
    ax.yaxis.label.set_color("#52677f")
    ax.title.set_color("#17365d")
    for spine in ax.spines.values():
        spine.set_color("#d8e3ef")
    ax.grid(axis="y", color="#e8eef5", linewidth=.8)
    ax.set_axisbelow(True)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("<div style='font-size:2rem'>💳</div>", unsafe_allow_html=True)
    st.markdown("### Credit Card Fraud Detection")
    st.caption("Anomaly detection using machine learning")
    st.divider()

    page_options = ["Dashboard", "Upload Dataset", "Fraud Detection", "Model Performance"]
    requested_page = st.session_state.pop("navigation_target", "Dashboard")
    page = st.radio(
        "NAVIGATION",
        page_options,
        index=page_options.index(requested_page) if requested_page in page_options else 0,
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**MODEL INFORMATION**")
    st.write("**Algorithm**  ")
    st.caption("Isolation Forest")
    st.write("**Learning method**  ")
    st.caption("Unsupervised")
    st.write("**Estimators**  ")
    st.caption(str(getattr(model, "n_estimators", "Not available")))
    st.write("**Status**  ")
    st.markdown("<span class='status-pill'>● Model loaded</span>", unsafe_allow_html=True)
    st.divider()
    st.caption("Credit Card Anomaly Detection System")


# ============================================================
# PAGE 1 - DASHBOARD
# ============================================================
if page == "Dashboard":
    results, is_demo = get_dashboard_data()
    total, normal, suspicious = transaction_summary(results)
    suspicious_rate = (suspicious / total * 100) if total else 0

    st.markdown("<div class='eyebrow'>TRANSACTION INTELLIGENCE</div>", unsafe_allow_html=True)
    st.title("Credit Card Fraud Detection System")
    st.markdown(
        "<div class='page-subtitle'>Monitor transaction patterns, review suspicious activity and identify potential fraud.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    if is_demo:
        st.markdown(
            "<div class='hero-strip'>"
            "<span class='demo-pill'>DEMO MODE</span> &nbsp; "
            "You are viewing illustrative sample data. Upload a CSV to run your saved Isolation Forest model on real transactions."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='hero-strip'><span class='status-pill'>LIVE ANALYSIS</span> &nbsp; "
            f"Showing predictions for <b>{st.session_state.get('uploaded_filename') or 'uploaded dataset'}</b>.</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Transaction Summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total transactions", f"{total:,}")
    k2.metric("Normal transactions", f"{normal:,}")
    k3.metric("Suspicious transactions", f"{suspicious:,}")
    k4.metric("Suspicious rate", f"{suspicious_rate:.2f}%")

    st.write("")
    left, right = st.columns([1, 1.55], gap="large")

    with left:
        st.markdown("<div class='panel-title'>Transaction Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-subtitle'>Share of normal and suspicious transactions</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        if total:
            vals = [normal, suspicious]
            wedges, _ = ax.pie(
                vals, startangle=90, counterclock=False,
                colors=["#2f6fa8", "#e36a58"],
                wedgeprops={"width": .34, "edgecolor": "white", "linewidth": 2},
            )
            ax.text(0, .06, f"{total:,}", ha="center", va="center", fontsize=20, fontweight="bold", color="#17365d")
            ax.text(0, -.17, "Transactions", ha="center", va="center", fontsize=9, color="#64748b")
            ax.legend(
                wedges,
                [f"Normal  {normal:,} ({(normal/total*100):.2f}%)", f"Suspicious  {suspicious:,} ({suspicious_rate:.2f}%)"],
                loc="lower center", bbox_to_anchor=(.5, -.13), frameon=False, fontsize=8,
            )
        ax.set_aspect("equal")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("<div class='panel-title'>Transaction Activity Timeline</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-subtitle'>Transaction volume over time</div>", unsafe_allow_html=True)
        time_col = find_column(results, ["Transaction Time", "Timestamp", "Time", "Date", "trans_date_trans_time"])
        if is_demo:
            # Demo records are intentionally recent examples from one day.
            # Use a separate, clearly-labelled 7-day sample trend so the chart
            # demonstrates a meaningful timeline before a real CSV is uploaded.
            demo_days = pd.date_range("2026-11-06", periods=7, freq="D")
            demo_counts = [7200, 7480, 7310, 7690, 7550, 7900, 7616]
            fig, ax = plt.subplots(figsize=(7.4, 3.4))
            ax.plot(demo_days, demo_counts, color="#2f6fa8", linewidth=2.2, marker="o", markersize=4)
            ax.fill_between(demo_days, demo_counts, color="#2f6fa8", alpha=.10)
            ax.set_ylabel("Sample transactions")
            ax.set_xlabel("Sample date")
            make_chart_style(ax)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("Illustrative 7-day trend for demo mode only; it is not calculated from the six sample rows.")
        elif time_col is not None:
            timeline = results.copy()
            timeline["_time"] = pd.to_datetime(timeline[time_col], errors="coerce")
            timeline = timeline.dropna(subset=["_time"])
            if not timeline.empty:
                # Daily buckets are best for multi-day files. For a file covering
                # only one day, use 15-minute buckets so intraday activity is visible.
                if timeline["_time"].dt.date.nunique() > 1:
                    timeline["_bucket"] = timeline["_time"].dt.floor("D")
                    x_label = "Date"
                else:
                    timeline["_bucket"] = timeline["_time"].dt.floor("15min")
                    x_label = "Time"
                total_by_period = timeline.groupby("_bucket").size().sort_index()

                fig, ax = plt.subplots(figsize=(7.4, 3.4))
                ax.plot(
                    total_by_period.index,
                    total_by_period.values,
                    color="#2f6fa8",
                    linewidth=2.2,
                    marker="o",
                    markersize=4,
                )
                ax.fill_between(
                    total_by_period.index,
                    total_by_period.values,
                    color="#2f6fa8",
                    alpha=.10,
                )
                if len(total_by_period) == 1:
                    # A single bucket cannot form a line; show a clear point and note.
                    ax.scatter(total_by_period.index, total_by_period.values, color="#2f6fa8", s=45, zorder=3)
                    ax.annotate(
                        f"{int(total_by_period.iloc[0]):,} transactions",
                        (total_by_period.index[0], total_by_period.iloc[0]),
                        xytext=(8, 10),
                        textcoords="offset points",
                    )
                ax.set_ylabel("Transactions")
                ax.set_xlabel(x_label)
                make_chart_style(ax)
                fig.autofmt_xdate()
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("A readable date/time column was found, but it contains no valid dates.")
        else:
            st.info("Your dataset has no recognised date/time column. Add a timestamp column to view a timeline.")
    st.write("")
    bottom_left, bottom_right = st.columns([1.35, 1], gap="large")

    with bottom_left:
        st.markdown("<div class='panel-title'>Recent Transactions</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-subtitle'>Latest sample or uploaded transaction predictions</div>", unsafe_allow_html=True)
        display = results.copy()
        time_col = find_column(display, ["Transaction Time", "Timestamp", "Time", "Date", "trans_date_trans_time"])
        amount_col = find_column(display, ["Amount", "amt", "Transaction Amount", "Transaction_Amount"])
        id_col = find_column(display, ["Transaction ID", "Transaction_ID", "trans_num", "id"])
        score_col = find_column(display, ["Anomaly Score", "Anomaly_Score", "Score"])
        pred_col = find_column(display, ["Prediction", "Predicted"])
        table = pd.DataFrame()
        table["Transaction ID"] = display[id_col].astype(str) if id_col else [f"TXN-{i+1:05d}" for i in range(len(display))]
        table["Amount"] = display[amount_col].map(format_amount) if amount_col else "Not provided"
        table["Time"] = display[time_col].astype(str) if time_col else "Not provided"
        table["Anomaly score"] = display[score_col].round(4) if score_col else "Not available"
        table["Prediction"] = display[pred_col].astype(str) if pred_col else "Not available"
        if time_col:
            parsed_time = pd.to_datetime(display[time_col], errors="coerce")
            table = table.assign(_sort=parsed_time).sort_values("_sort", ascending=False).drop(columns="_sort")
        st.dataframe(table.head(6), use_container_width=True, hide_index=True)
        if is_demo:
            st.caption("Illustrative sample transactions. Upload your own CSV for live predictions.")

    with bottom_right:
        st.markdown("<div class='panel-title'>Fraud Risk Overview</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-subtitle'>Share of transactions flagged by the model</div>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        r1.metric("Suspicious rate", f"{suspicious_rate:.2f}%")
        risk_label = "High" if suspicious_rate >= 5 else "Moderate" if suspicious_rate >= 1 else "Low"
        r2.metric("Portfolio signal", risk_label)
        st.progress(min(max(suspicious_rate / 10, 0.0), 1.0))
        st.caption("This is a screening indicator based on model flags, not a confirmed fraud or financial loss rate.")
        st.markdown(f"**Model threshold:** `{threshold:.6f}`")
        st.markdown(f"**Model:** Isolation Forest · {getattr(model, 'n_estimators', 'N/A')} estimators")

    st.write("")
    st.markdown("<div class='panel-title'>Top Suspicious Transaction Amounts</div>", unsafe_allow_html=True)
    if amount_col and pred_col:
        suspicious_rows = results[results[pred_col].astype(str).str.lower().isin(["suspicious", "fraud", "1", "true", "anomaly"])].copy()
        if not suspicious_rows.empty:
            suspicious_rows[amount_col] = pd.to_numeric(suspicious_rows[amount_col], errors="coerce")
            top_amounts = suspicious_rows.dropna(subset=[amount_col]).nlargest(5, amount_col)
            if not top_amounts.empty:
                top_table = top_amounts[[c for c in [id_col, amount_col, score_col, pred_col] if c is not None]].copy()
                top_table = top_table.rename(columns={amount_col: "Amount (₹)"})
                st.dataframe(top_table, use_container_width=True, hide_index=True)
            else:
                st.info("No numeric transaction amounts are available for suspicious records.")
        else:
            st.info("No suspicious transactions were flagged in this dataset.")
    elif is_demo:
        st.info("Upload a dataset containing an amount column to rank suspicious transaction values.")
    else:
        st.info("Your dataset needs an amount column and model prediction results to show this analysis.")

    st.write("")
    upload_col, info_col = st.columns([1.15, 1], gap="large")
    with upload_col:
        st.markdown("<div class='panel-title'>Analyse your own transactions</div>", unsafe_allow_html=True)
        st.write("Upload a CSV file and run the trained Isolation Forest model to replace the demo overview with your own results.")
        if st.button("Go to dataset upload", key="dashboard_upload", use_container_width=True):
            st.session_state["navigation_target"] = "Upload Dataset"
            st.rerun()
    with info_col:
        st.markdown("<div class='panel-title'>About this model</div>", unsafe_allow_html=True)
        st.write("Isolation Forest is an unsupervised anomaly-detection algorithm. It identifies observations that are easier to isolate from the broader transaction pattern.")


# ============================================================
# PAGE 2 - UPLOAD DATASET
# ============================================================
elif page == "Upload Dataset":
    st.markdown("<div class='eyebrow'>DATA INPUT</div>", unsafe_allow_html=True)
    st.title("Upload Transaction Dataset")
    st.markdown("<div class='page-subtitle'>Import a CSV file, validate the model features and run fraud detection.</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<div class='panel-title'>Select your CSV file</div>", unsafe_allow_html=True)
    st.caption(f"Your file must include {len(features)} model features used during training.")
    uploaded_file = st.file_uploader("Choose a transaction CSV", type=["csv"], key="transaction_csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.uploaded_raw_data = data
            st.session_state.uploaded_filename = uploaded_file.name
            st.subheader("Dataset Preview")
            p1, p2, p3 = st.columns(3)
            p1.metric("Rows", f"{len(data):,}")
            p2.metric("Columns", f"{len(data.columns):,}")
            p3.metric("File size", f"{uploaded_file.size / (1024 * 1024):.2f} MB")
            st.dataframe(data.head(8), use_container_width=True, hide_index=True)

            missing_features = [column for column in features if column not in data.columns]
            if data.empty:
                st.error("The uploaded dataset is empty.")
            elif missing_features:
                st.error("The following model features are missing: " + ", ".join(missing_features))
                st.info("Use the same feature columns as the dataset used to train the saved Isolation Forest model.")
            else:
                st.success("All required model features are present.")
                st.caption("Optional columns such as transaction ID, timestamp, amount and Class are retained in the output when provided.")
                if st.button("Analyse Transactions", type="primary", use_container_width=True):
                    with st.spinner("Scoring transactions with Isolation Forest..."):
                        analysed = detect_fraud(data)
                        st.session_state.analysis_results = analysed
                        st.session_state.uploaded_filename = uploaded_file.name
                    st.success(f"Analysis completed for {len(analysed):,} transactions.")
                    st.info("Open Dashboard or Fraud Detection to review the predictions.")
                    st.rerun()
        except Exception as exc:
            st.error(f"Unable to read or analyse this dataset: {exc}")
    else:
        st.markdown(
            "<div class='panel' style='text-align:center;padding:34px 20px;'>"
            "<div style='font-size:2.1rem'>⇧</div>"
            "<div class='panel-title'>Drop your transaction CSV here</div>"
            "<div class='panel-subtitle'>CSV format · Your data is analysed using the saved Isolation Forest model</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    if isinstance(st.session_state.get("analysis_results"), pd.DataFrame):
        st.divider()
        st.markdown("<span class='status-pill'>Previous analysis available</span>", unsafe_allow_html=True)
        st.write(f"Current results: **{len(st.session_state.analysis_results):,} transactions** from **{st.session_state.get('uploaded_filename') or 'uploaded file'}**.")
        if st.button("Clear current analysis"):
            st.session_state.analysis_results = None
            st.session_state.uploaded_filename = None
            st.session_state.uploaded_raw_data = None
            st.rerun()


# ============================================================
# PAGE 3 - FRAUD DETECTION
# ============================================================
elif page == "Fraud Detection":
    st.markdown("<div class='eyebrow'>INVESTIGATION</div>", unsafe_allow_html=True)
    st.title("Fraud Detection Results")
    results = st.session_state.get("analysis_results")
    if isinstance(results, pd.DataFrame) and not results.empty:
        total, normal, suspicious = transaction_summary(results)
        rate = suspicious / total * 100 if total else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total transactions", f"{total:,}")
        c2.metric("Normal", f"{normal:,}")
        c3.metric("Suspicious", f"{suspicious:,}")
        c4.metric("Suspicious rate", f"{rate:.2f}%")
        st.divider()

        prediction_col = find_column(results, ["Prediction", "Predicted"])
        if prediction_col:
            filter_option = st.selectbox("Filter transactions", ["All transactions", "Normal", "Suspicious"])
            labels = results[prediction_col].astype(str).str.lower()
            if filter_option == "Normal":
                display_data = results[labels.isin(["normal", "0", "false"])]
            elif filter_option == "Suspicious":
                display_data = results[labels.isin(["suspicious", "fraud", "1", "true", "anomaly"])]
            else:
                display_data = results
        else:
            display_data = results
        st.dataframe(display_data, use_container_width=True, hide_index=True)
        st.caption(f"Displaying {len(display_data):,} of {len(results):,} transactions.")
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction report",
            data=csv_bytes,
            file_name="fraud_detection_results.csv",
            mime="text/csv",
        )
    else:
        st.info("No uploaded transaction analysis is available yet. Visit Upload Dataset to run the model. The Dashboard remains available in demo mode.")


# ============================================================
# PAGE 4 - MODEL PERFORMANCE
# Evaluation metrics are shown only when true labels are available.
# ============================================================
elif page == "Model Performance":
    st.markdown("<div class='eyebrow'>MODEL VALIDATION</div>", unsafe_allow_html=True)
    st.title("Model Performance")
    st.markdown("<div class='page-subtitle'>Evaluation metrics require actual class labels in the dataset.</div>", unsafe_allow_html=True)

    evaluation_source = None
    uploaded_results = st.session_state.get("analysis_results")
    if isinstance(uploaded_results, pd.DataFrame) and not uploaded_results.empty:
        actual_col = find_column(uploaded_results, ["Class", "Actual", "Label", "is_fraud"])
        if actual_col:
            evaluation_source = uploaded_results.copy()

    if evaluation_source is None and SAVED_RESULTS_PATH.exists():
        try:
            saved_results = pd.read_csv(SAVED_RESULTS_PATH)
            if find_column(saved_results, ["Actual", "Class", "Label"]) and find_column(saved_results, ["Predicted", "Prediction"]):
                evaluation_source = saved_results
        except Exception:
            evaluation_source = None

    if evaluation_source is None:
        st.info("Model predictions can run without a Class column, but Precision, Recall, F1-score, ROC-AUC and the confusion matrix need actual labels. Upload a labelled test dataset to view these evaluation results.")
        st.markdown("<div class='panel-title'>Model Configuration</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Algorithm", "Isolation Forest")
        m2.metric("Estimators", str(getattr(model, "n_estimators", "N/A")))
        m3.metric("Anomaly threshold", f"{threshold:.5f}")
    else:
        actual_col = find_column(evaluation_source, ["Class", "Actual", "Label", "is_fraud"])
        predicted_col = find_column(evaluation_source, ["Prediction", "Predicted"])
        score_col = find_column(evaluation_source, ["Anomaly Score", "Anomaly_Score", "Score"])

        actual = pd.to_numeric(evaluation_source[actual_col], errors="coerce")
        if predicted_col == "Prediction":
            predicted = evaluation_source[predicted_col].astype(str).str.lower().isin(["suspicious", "fraud", "1", "true", "anomaly"]).astype(int)
        else:
            predicted = pd.to_numeric(evaluation_source[predicted_col], errors="coerce")
        scores = pd.to_numeric(evaluation_source[score_col], errors="coerce") if score_col else None

        evaluation = pd.DataFrame({"Actual": actual, "Predicted": predicted})
        if scores is not None:
            evaluation["Score"] = scores
        evaluation = evaluation.dropna()
        if evaluation.empty or not set(evaluation["Actual"].astype(int).unique()).issubset({0, 1}):
            st.warning("The actual label column must contain valid 0 (normal) and 1 (fraud) values.")
        else:
            y_true = evaluation["Actual"].astype(int)
            y_pred = evaluation["Predicted"].astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc = None
            if "Score" in evaluation and y_true.nunique() == 2:
                try:
                    auc = roc_auc_score(y_true, evaluation["Score"].astype(float))
                except ValueError:
                    auc = None

            st.markdown("<span class='status-pill'>Evaluation based on labelled data</span>", unsafe_allow_html=True)
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Precision", f"{precision*100:.2f}%")
            e2.metric("Recall", f"{recall*100:.2f}%")
            e3.metric("F1 score", f"{f1*100:.2f}%")
            e4.metric("ROC-AUC", f"{auc*100:.2f}%" if auc is not None else "N/A")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["Actual Normal", "Actual Fraud"], columns=["Predicted Normal", "Predicted Fraud"])
            st.dataframe(cm_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            image = ax.imshow(cm, cmap="Blues")
            ax.set_title("Isolation Forest - Confusion Matrix", color="#17365d", pad=12)
            ax.set_xlabel("Predicted label", color="#52677f")
            ax.set_ylabel("Actual label", color="#52677f")
            ax.set_xticks([0, 1], labels=["Normal", "Fraud"])
            ax.set_yticks([0, 1], labels=["Normal", "Fraud"])
            midpoint = cm.max() / 2 if cm.size else 0
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, f"{cm[row, col]:,}", ha="center", va="center", color="white" if cm[row, col] > midpoint else "#17365d", fontsize=11)
            fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.subheader("ROC Curve")
            if auc is not None and "Score" in evaluation and y_true.nunique() == 2:
                fpr, tpr, _ = roc_curve(y_true, evaluation["Score"].astype(float))
                fig, ax = plt.subplots(figsize=(7, 4.3))
                ax.plot(fpr, tpr, color="#2f6fa8", linewidth=2.2, label=f"Isolation Forest (AUC = {auc:.2f})")
                ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa8b7", linewidth=1)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("Receiver Operating Characteristic", color="#17365d")
                ax.legend(frameon=False)
                make_chart_style(ax)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("ROC-AUC and the ROC curve need a score column and both actual classes.")

    st.divider()
    st.subheader("Model Information")
    info1, info2, info3 = st.columns(3)
    info1.metric("Algorithm", "Isolation Forest")
    info2.metric("Number of estimators", str(getattr(model, "n_estimators", "N/A")))
    info3.metric("Selected threshold", f"{threshold:.6f}")
    st.caption("Training method: Unsupervised anomaly detection. The saved model and threshold are loaded from the project's models folder.")
