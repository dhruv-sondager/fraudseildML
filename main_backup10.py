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

if "manual_result" not in st.session_state:
    st.session_state.manual_result = None

if "manual_history" not in st.session_state:
    st.session_state.manual_history = pd.DataFrame()

if "detection_mode" not in st.session_state:
    st.session_state.detection_mode = "Demo Mode"


# ============================================================
# MODEL PREDICTION FUNCTION
# ORIGINAL ISOLATION FOREST LOGIC
# ============================================================
def detect_fraud(data: pd.DataFrame) -> pd.DataFrame:

    """Score transactions with the saved Isolation Forest model."""

    missing = [
        column for column in features
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required model features: " + ", ".join(missing)
        )

    X = data.loc[:, features].copy()

    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():

        bad_columns = X.columns[
            X.isna().any()
        ].tolist()

        raise ValueError(
            "Missing or non-numeric values found in model features: "
            + ", ".join(bad_columns[:12])
        )

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Infinite values found in model features."
        )

    scores = model.decision_function(X)

    predictions = np.where(
        scores < threshold,
        "Suspicious",
        "Normal"
    )

    results = data.copy()

    results["Anomaly Score"] = scores
    results["Prediction"] = predictions

    return results


# ============================================================
# COMMON HELPER FUNCTIONS
# ============================================================
def format_currency(value):
    try:
        return f"₹{float(value):,.2f}"
    except Exception:
        return "N/A"


def get_prediction_counts(data):

    if data is None or data.empty:
        return 0, 0, 0

    if "Prediction" not in data.columns:
        return 0, 0, 0

    suspicious = int(
        (data["Prediction"] == "Suspicious").sum()
    )

    normal = int(
        (data["Prediction"] == "Normal").sum()
    )

    total = len(data)

    return total, normal, suspicious


def create_demo_data():

    np.random.seed(42)

    demo = pd.DataFrame(
        np.random.normal(
            loc=0,
            scale=1,
            size=(500, len(features))
        ),
        columns=features
    )

    return demo


def add_manual_history(result):

    if st.session_state.manual_history.empty:
        st.session_state.manual_history = result.copy()

    else:
        st.session_state.manual_history = pd.concat(
            [
                st.session_state.manual_history,
                result
            ],
            ignore_index=True
        )


# ============================================================
# SIDEBAR - DETECTION MODE
# ============================================================
with st.sidebar:

    st.markdown("## 💳 Fraud Detection")

    st.caption("Credit Card Anomaly Detection System")

    st.divider()

    st.markdown("### Detection Mode")

    detection_mode = st.radio(
        "Select detection mode",
        [
            "Demo Mode",
            "Dataset Analysis Mode",
            "Manual Transaction Mode"
        ],
        key="detection_mode"
    )

    st.divider()


# ============================================================
# DYNAMIC SIDEBAR NAVIGATION
# ============================================================
shared_pages = [
    "Dashboard",
    "Fraud Detection",
    "Model Performance"
]

if st.session_state.detection_mode == "Manual Transaction Mode":

    page_options = [
        "Dashboard",
        "Upload Dataset",
        "Manual Detection",
        "Fraud Detection",
        "Model Performance"
    ]

elif st.session_state.detection_mode == "Dataset Analysis Mode":

    page_options = [
        "Dashboard",
        "Upload Dataset",
        "Fraud Detection",
        "Model Performance"
    ]

else:

    page_options = shared_pages


requested_page = st.session_state.pop(
    "navigation_target",
    "Dashboard"
)

if requested_page not in page_options:
    requested_page = "Dashboard"


with st.sidebar:

    page = st.radio(
        "NAVIGATION",
        page_options,
        index=page_options.index(requested_page),
        key="main_navigation"
    )


# ============================================================
# APPLICATION HEADER
# ============================================================
st.markdown(
    '<div class="eyebrow">MACHINE LEARNING • FINANCIAL SECURITY</div>',
    unsafe_allow_html=True
)

st.title("Credit Card Fraud Detection")

st.markdown(
    '<div class="page-subtitle">'
    'Identify unusual transaction patterns using an unsupervised '
    'machine learning model.'
    '</div>',
    unsafe_allow_html=True
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
            unsafe_allow_html=True
        )

        st.info(
            "You are viewing illustrative demo data. "
            "Switch to Dataset Analysis Mode or Manual Transaction Mode "
            "to work with your own transactions."
        )

        demo_data = create_demo_data()

        try:
            dashboard_results = detect_fraud(demo_data)

        except Exception as exc:
            st.error(f"Unable to generate demo predictions: {exc}")
            dashboard_results = pd.DataFrame()

    else:

        dashboard_results = st.session_state.analysis_results

        if dashboard_results is None or dashboard_results.empty:

            st.info(
                "No analysis results are available yet. "
                "Upload a dataset or analyse a manual transaction to begin."
            )

            if st.session_state.detection_mode == "Dataset Analysis Mode":

                if st.button("Go to Upload Dataset", type="primary"):
                    st.session_state.navigation_target = "Upload Dataset"
                    st.rerun()

            elif st.session_state.detection_mode == "Manual Transaction Mode":

                if st.button("Go to Manual Detection", type="primary"):
                    st.session_state.navigation_target = "Manual Detection"
                    st.rerun()

            st.stop()

    if dashboard_results is not None and not dashboard_results.empty:

        total, normal, suspicious = get_prediction_counts(
            dashboard_results
        )

        suspicious_rate = (
            suspicious / total * 100
            if total > 0 else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Transactions",
                f"{total:,}"
            )

        with col2:
            st.metric(
                "Normal Transactions",
                f"{normal:,}"
            )

        with col3:
            st.metric(
                "Suspicious Transactions",
                f"{suspicious:,}"
            )

        with col4:
            st.metric(
                "Suspicious Rate",
                f"{suspicious_rate:.2f}%"
            )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.markdown("### Transaction Distribution")

            distribution = pd.DataFrame(
                {
                    "Transaction Type": [
                        "Normal",
                        "Suspicious"
                    ],
                    "Count": [
                        normal,
                        suspicious
                    ]
                }
            )

            st.bar_chart(
                distribution.set_index("Transaction Type")
            )

        with right:

            st.markdown("### Prediction Summary")

            if suspicious > 0:

                st.warning(
                    f"{suspicious} transactions were classified as suspicious."
                )

            else:

                st.success(
                    "No suspicious transactions were detected."
                )

            st.write(
                f"Normal transactions: **{normal:,}**"
            )

            st.write(
                f"Suspicious transactions: **{suspicious:,}**"
            )

        st.divider()

        st.markdown("### Recent Transactions")

        display_columns = [
            column for column in [
                "Time",
                "Amount",
                "Anomaly Score",
                "Prediction"
            ]
            if column in dashboard_results.columns
        ]

        if display_columns:

            st.dataframe(
                dashboard_results[display_columns].tail(10),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.dataframe(
                dashboard_results.tail(10),
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        st.markdown("### Suspicious Transaction Overview")

        suspicious_data = dashboard_results[
            dashboard_results["Prediction"] == "Suspicious"
        ]

        if not suspicious_data.empty:

            if "Amount" in suspicious_data.columns:

                top_suspicious = suspicious_data.sort_values(
                    "Amount",
                    ascending=False
                ).head(10)

                st.dataframe(
                    top_suspicious,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.dataframe(
                    suspicious_data.head(10),
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.info("No suspicious transactions to display.")


# ============================================================
# UPLOAD DATASET PAGE
# ============================================================
elif page == "Upload Dataset":

    st.subheader("Upload Transaction Dataset")

    st.markdown(
        "Upload a CSV file containing transaction records "
        "for fraud and anomaly analysis."
    )

    st.info(
        "Your dataset must contain all the feature columns required "
        "by the trained Isolation Forest model."
    )

    uploaded_file = st.file_uploader(
        "Choose your CSV dataset",
        type=["csv"],
        key="transaction_dataset_uploader"
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
                hide_index=True
            )

            missing_features = [
                feature for feature in features
                if feature not in uploaded_data.columns
            ]

            if missing_features:

                st.error(
                    "This dataset is not compatible with the trained model."
                )

                st.write("Missing required feature columns:")

                st.code(
                    ", ".join(missing_features)
                )

                st.warning(
                    "Upload a dataset containing all the required "
                    "model feature columns."
                )

            else:

                numeric_data = uploaded_data[features].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                invalid_columns = numeric_data.columns[
                    numeric_data.isna().any()
                ].tolist()

                if invalid_columns:

                    st.error(
                        "Some required model features contain missing "
                        "or non-numeric values."
                    )

                    st.write(
                        "Problematic columns: "
                        + ", ".join(invalid_columns)
                    )

                elif not np.isfinite(
                    numeric_data.to_numpy(dtype=float)
                ).all():

                    st.error(
                        "The dataset contains infinite values "
                        "in required model features."
                    )

                else:

                    st.success(
                        "Compatible dataset detected! "
                        "All required model features are available."
                    )

                    if st.button(
                        "Use This Dataset",
                        type="primary",
                        use_container_width=True
                    ):

                        st.session_state.uploaded_raw_data = uploaded_data.copy()

                        st.session_state.uploaded_filename = (
                            uploaded_file.name
                        )

                        st.session_state.analysis_results = None

                        st.success(
                            "Dataset uploaded and saved successfully."
                        )

                        st.rerun()

        except Exception as exc:

            st.error(
                f"Unable to read the uploaded CSV file: {exc}"
            )

    if st.session_state.uploaded_raw_data is not None:

        st.divider()

        st.markdown("### Currently Loaded Dataset")

        current_data = st.session_state.uploaded_raw_data

        st.success(
            f"Loaded dataset: "
            f"{st.session_state.uploaded_filename or 'Uploaded dataset'}"
        )

        st.write(
            f"**Records:** {len(current_data):,}  |  "
            f"**Columns:** {len(current_data.columns):,}"
        )

        if st.button(
            "Remove Uploaded Dataset",
            key="remove_uploaded_dataset"
        ):

            st.session_state.uploaded_raw_data = None

            st.session_state.uploaded_filename = None

            st.session_state.analysis_results = None

            st.success("Uploaded dataset removed.")

            st.rerun()

        st.caption(
            "This dataset remains available when you navigate "
            "between pages during the current Streamlit session."
        )

    else:

        st.warning(
            "No dataset is currently loaded. "
            "Upload a compatible CSV before using Quick Entry "
            "in Manual Transaction Mode."
        )


# ============================================================
# FRAUD DETECTION PAGE
# ============================================================
elif page == "Fraud Detection":

    st.subheader("Fraud Detection Analysis")

    st.markdown(
        "Analyse a transaction dataset using the trained "
        "Isolation Forest model."
    )

    st.info(
        "The model uses all required features to calculate anomaly scores "
        "and classify transactions as Normal or Suspicious."
    )

    source_option = st.radio(
        "Select dataset source",
        [
            "Use Uploaded Dataset",
            "Upload New Dataset"
        ],
        horizontal=True
    )

    selected_data = None

    if source_option == "Use Uploaded Dataset":

        if st.session_state.uploaded_raw_data is None:

            st.warning(
                "No dataset has been uploaded. "
                "Please visit Upload Dataset first."
            )

        else:

            selected_data = st.session_state.uploaded_raw_data.copy()

            st.success(
                f"Using: {st.session_state.uploaded_filename}"
            )

    else:

        fraud_file = st.file_uploader(
            "Upload CSV for fraud detection",
            type=["csv"],
            key="fraud_detection_uploader"
        )

        if fraud_file is not None:

            try:

                selected_data = pd.read_csv(fraud_file)

            except Exception as exc:

                st.error(
                    f"Unable to read CSV file: {exc}"
                )

    if selected_data is not None:

        st.write(
            f"Dataset contains **{len(selected_data):,} transactions**."
        )

        missing_features = [
            feature for feature in features
            if feature not in selected_data.columns
        ]

        if missing_features:

            st.error(
                "Missing required features: "
                + ", ".join(missing_features)
            )

        else:

            st.dataframe(
                selected_data.head(5),
                use_container_width=True,
                hide_index=True
            )

            if st.button(
                "Run Fraud Detection",
                type="primary",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "Analysing transactions..."
                    ):

                        results = detect_fraud(selected_data)

                    st.session_state.analysis_results = results

                    st.success(
                        "Fraud detection completed successfully!"
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Fraud detection failed: {exc}"
                    )

    if st.session_state.analysis_results is not None:

        results = st.session_state.analysis_results

        st.divider()

        st.markdown("### Analysis Results")

        total, normal, suspicious = get_prediction_counts(
            results
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Transactions",
            f"{total:,}"
        )

        col2.metric(
            "Normal",
            f"{normal:,}"
        )

        col3.metric(
            "Suspicious",
            f"{suspicious:,}"
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Detection Results",
            data=csv_data,
            file_name="fraud_detection_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        # ============================================================
# MANUAL DETECTION PAGE
# ============================================================
elif page == "Manual Detection":

    st.subheader("Manual Transaction Detection")

    st.markdown(
        "Enter transaction details manually and use the trained "
        "Isolation Forest model to identify suspicious transactions."
    )

    st.info(
        "The model requires all "
        + str(len(features))
        + " features. You can enter five important features using "
        "Quick Entry or provide all features using Full Entry."
    )

    # --------------------------------------------------------
    # CHECK UPLOADED DATASET COMPATIBILITY
    # --------------------------------------------------------
    uploaded_baseline = st.session_state.uploaded_raw_data

    compatible_baseline = False
    medians = None

    if (
        isinstance(uploaded_baseline, pd.DataFrame)
        and not uploaded_baseline.empty
        and all(feature in uploaded_baseline.columns for feature in features)
    ):

        numeric_baseline = uploaded_baseline[features].apply(
            pd.to_numeric,
            errors="coerce"
        )

        candidate_medians = numeric_baseline.median()

        if all(pd.notna(candidate_medians[feature]) for feature in features):

            if np.isfinite(
                candidate_medians.to_numpy(dtype=float)
            ).all():

                compatible_baseline = True
                medians = candidate_medians

    # --------------------------------------------------------
    # SELECT TRANSACTION ENTRY TYPE
    # --------------------------------------------------------
    entry_type = st.radio(
        "Select transaction entry type",
        [
            "Single Transaction",
            "Multiple Transactions"
        ],
        horizontal=True,
        key="manual_entry_type"
    )

    # --------------------------------------------------------
    # SELECT FEATURE ENTRY METHOD
    # --------------------------------------------------------
    entry_mode = st.radio(
        "Choose input method",
        [
            "Quick Entry — 5 fields",
            "Full Entry — all model features"
        ],
        horizontal=True,
        key="manual_entry_method"
    )

    quick_features = [
        feature for feature in [
            "Time",
            "Amount",
            "V1",
            "V2",
            "V3"
        ]
        if feature in features
    ]

    # If the saved model uses different column names,
    # use the first five model features for Quick Entry.
    if len(quick_features) < 5:

        quick_features = features[:5]

    quick_entry_blocked = (
        entry_mode == "Quick Entry — 5 fields"
        and not compatible_baseline
    )

    # --------------------------------------------------------
    # QUICK ENTRY WARNING
    # --------------------------------------------------------
    if (
        entry_mode == "Quick Entry — 5 fields"
        and not compatible_baseline
    ):

        st.warning(
            """
            **Suitable Dataset Required!**

            You have selected Quick Entry (5 fields).

            The model requires all 30 features to make a prediction.
            The remaining features will be automatically populated
            using median values from your uploaded dataset.

            Please open **Upload Dataset** from the sidebar and upload
            a suitable CSV file containing all required model features.

            Until you upload a compatible dataset, please select
            Full Entry to enter all features manually.
            """
        )

        st.info(
            "Quick Entry is currently disabled because a compatible "
            "dataset has not been uploaded."
        )

    elif entry_mode == "Quick Entry — 5 fields":

        st.success(
            "Compatible dataset loaded: "
            + str(st.session_state.uploaded_filename)
        )

        st.caption(
            "The remaining "
            + str(len(features) - len(quick_features))
            + " features will use median values from your uploaded dataset."
        )

    else:

        st.info(
            "Full Entry allows you to manually provide all "
            + str(len(features))
            + " model features. Dataset upload is not required."
        )

    st.divider()

    # --------------------------------------------------------
    # HELPER: BUILD A TRANSACTION USING THE SELECTED MODE
    # --------------------------------------------------------
    def build_manual_transaction(values, input_features):

        transaction = {}

        if entry_mode == "Quick Entry — 5 fields":

            if not compatible_baseline:

                raise ValueError(
                    "Please upload a compatible dataset before using Quick Entry."
                )

            for feature in features:

                transaction[feature] = float(medians[feature])

            for feature in quick_features:

                transaction[feature] = float(values[feature])

        else:

            for feature in features:

                transaction[feature] = float(values[feature])

        return pd.DataFrame(
            [transaction],
            columns=features
        )

    # --------------------------------------------------------
    # SINGLE TRANSACTION ENTRY
    # --------------------------------------------------------
    if entry_type == "Single Transaction":

        st.markdown("### Enter Transaction Details")

        st.caption(
            "Provide the transaction values below."
        )

        with st.form("single_manual_transaction_form"):

            entered_values = {}

            if entry_mode == "Quick Entry — 5 fields":

                st.markdown("#### Quick Entry")

                st.write(
                    "Enter the five required feature values."
                )

                input_columns = st.columns(2)

                for index, feature in enumerate(quick_features):

                    with input_columns[index % 2]:

                        default_value = (
                            0.0
                            if feature != "Amount"
                            else 100.0
                        )

                        entered_values[feature] = st.number_input(
                            feature,
                            value=float(default_value),
                            step=0.01,
                            format="%.6f",
                            key=f"single_quick_{feature}"
                        )

            else:

                st.markdown("#### Full Entry")

                st.write(
                    "Enter values for all model features."
                )

                input_columns = st.columns(3)

                for index, feature in enumerate(features):

                    with input_columns[index % 3]:

                        default_value = (
                            100.0
                            if feature == "Amount"
                            else 0.0
                        )

                        entered_values[feature] = st.number_input(
                            feature,
                            value=float(default_value),
                            step=0.01,
                            format="%.6f",
                            key=f"single_full_{feature}"
                        )

            submitted = st.form_submit_button(
                "Analyse Transaction",
                type="primary",
                use_container_width=True,
                disabled=quick_entry_blocked
            )

        if submitted:

            try:

                manual_transaction = build_manual_transaction(
                    entered_values,
                    features
                )

                prediction_result = detect_fraud(
                    manual_transaction
                )

                st.session_state.manual_result = prediction_result

                add_manual_history(prediction_result)

                st.success(
                    "Transaction analysed successfully!"
                )

            except Exception as exc:

                st.error(
                    f"Unable to analyse transaction: {exc}"
                )

    # --------------------------------------------------------
    # MULTIPLE TRANSACTION ENTRY
    # --------------------------------------------------------
    else:

        st.markdown("### Enter Multiple Transactions")

        st.write(
            "You can enter between 2 and 100 transactions "
            "and analyse them together."
        )

        number_of_transactions = st.number_input(
            "Number of transactions",
            min_value=2,
            max_value=100,
            value=5,
            step=1,
            key="manual_batch_size"
        )

        st.caption(
            "Edit the transaction values directly in the table."
        )

        if entry_mode == "Quick Entry — 5 fields":

            batch_columns = quick_features

        else:

            batch_columns = features

        default_batch_values = {}

        for feature in batch_columns:

            if feature == "Amount":

                default_batch_values[feature] = 100.0

            else:

                default_batch_values[feature] = 0.0

        batch_data = pd.DataFrame(
            [
                default_batch_values.copy()
                for _ in range(int(number_of_transactions))
            ]
        )

        edited_batch = st.data_editor(
            batch_data,
            num_rows="fixed",
            use_container_width=True,
            key="manual_batch_editor"
        )

        st.caption(
            "Only the features displayed above are entered manually. "
            "In Quick Entry, the other features are filled using "
            "the uploaded dataset's median values."
        )

        if st.button(
            "Analyse All Transactions",
            type="primary",
            use_container_width=True,
            key="analyse_manual_batch",
            disabled=quick_entry_blocked
        ):

            try:

                if edited_batch.empty:

                    st.error(
                        "Please enter at least one transaction."
                    )

                else:

                    transaction_records = []

                    for _, row in edited_batch.iterrows():

                        values = {
                            feature: float(row[feature])
                            for feature in batch_columns
                        }

                        transaction = {}

                        if entry_mode == "Quick Entry — 5 fields":

                            if not compatible_baseline:

                                raise ValueError(
                                    "Upload a compatible dataset before "
                                    "using Quick Entry."
                                )

                            for feature in features:

                                transaction[feature] = float(
                                    medians[feature]
                                )

                            for feature in quick_features:

                                transaction[feature] = values[feature]

                        else:

                            for feature in features:

                                transaction[feature] = values[feature]

                        transaction_records.append(transaction)

                    batch_transactions = pd.DataFrame(
                        transaction_records,
                        columns=features
                    )

                    batch_results = detect_fraud(
                        batch_transactions
                    )

                    st.session_state.manual_result = batch_results

                    add_manual_history(batch_results)

                    st.success(
                        "All transactions analysed successfully!"
                    )

            except Exception as exc:

                st.error(
                    f"Unable to analyse transactions: {exc}"
                )

    # --------------------------------------------------------
    # DISPLAY MANUAL DETECTION RESULTS
    # --------------------------------------------------------
    if st.session_state.manual_result is not None:

        result = st.session_state.manual_result

        st.divider()

        st.markdown("### Transaction Prediction Results")

        total, normal, suspicious = get_prediction_counts(
            result
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Transactions Analysed",
            f"{total:,}"
        )

        col2.metric(
            "Normal Transactions",
            f"{normal:,}"
        )

        col3.metric(
            "Suspicious Transactions",
            f"{suspicious:,}"
        )

        if suspicious > 0:

            st.error(
                f"{suspicious} transaction(s) classified as Suspicious."
            )

        else:

            st.success(
                "All analysed transactions were classified as Normal."
            )

        result_columns = [
            feature for feature in [
                "Time",
                "Amount",
                "Anomaly Score",
                "Prediction"
            ]
            if feature in result.columns
        ]

        if not result_columns:

            result_columns = [
                "Anomaly Score",
                "Prediction"
            ]

        st.dataframe(
            result[result_columns],
            use_container_width=True,
            hide_index=True
        )

        manual_csv = result.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Manual Detection Results",
            data=manual_csv,
            file_name="manual_detection_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    # --------------------------------------------------------
    # MANUAL DETECTION ACTIVITY HISTORY
    # --------------------------------------------------------
    st.divider()

    st.markdown("### Manual Detection Activity")

    history = st.session_state.manual_history

    if history is not None and not history.empty:

        history_total, history_normal, history_suspicious = (
            get_prediction_counts(history)
        )

        history_rate = (
            history_suspicious / history_total * 100
            if history_total > 0 else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Checked",
            f"{history_total:,}"
        )

        col2.metric(
            "Normal",
            f"{history_normal:,}"
        )

        col3.metric(
            "Suspicious",
            f"{history_suspicious:,}"
        )

        col4.metric(
            "Suspicious Rate",
            f"{history_rate:.2f}%"
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        history_csv = history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Complete Manual Activity",
            data=history_csv,
            file_name="manual_detection_activity.csv",
            mime="text/csv"
        )

        if st.button(
            "Clear Manual Detection History",
            key="clear_manual_history"
        ):

            st.session_state.manual_history = pd.DataFrame()

            st.session_state.manual_result = None

            st.rerun()

    else:

        st.info(
            "No manual transaction activity recorded yet. "
            "Analyse a transaction to see your activity here."
        )
        # ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================
elif page == "Model Performance":

    st.subheader("Model Performance")

    st.markdown(
        "Evaluate the trained Isolation Forest model "
        "using labelled transaction data."
    )

    st.info(
        """
        Model performance metrics require actual transaction labels.
        The model's Normal/Suspicious predictions alone cannot calculate
        precision, recall, F1-score or ROC-AUC.
        """
    )

    evaluation_data = None

    # --------------------------------------------------------
    # LOAD LABELLED DATA
    # --------------------------------------------------------
    if st.session_state.analysis_results is not None:

        possible_label_columns = [
            "Class",
            "class",
            "Actual",
            "actual",
            "Label",
            "label",
            "Fraud",
            "fraud"
        ]

        existing_labels = [
            column
            for column in possible_label_columns
            if column in st.session_state.analysis_results.columns
        ]

        if existing_labels:

            selected_label = st.selectbox(
                "Select actual label column",
                existing_labels
            )

            evaluation_data = st.session_state.analysis_results.copy()

    if evaluation_data is None:

        performance_file = st.file_uploader(
            "Upload labelled test dataset for evaluation",
            type=["csv"],
            key="performance_dataset_uploader"
        )

        if performance_file is not None:

            try:

                evaluation_data = pd.read_csv(
                    performance_file
                )

            except Exception as exc:

                st.error(
                    f"Unable to read evaluation dataset: {exc}"
                )

    # --------------------------------------------------------
    # EVALUATE MODEL
    # --------------------------------------------------------
    if evaluation_data is not None:

        possible_label_columns = [
            "Class",
            "class",
            "Actual",
            "actual",
            "Label",
            "label",
            "Fraud",
            "fraud"
        ]

        existing_labels = [
            column
            for column in possible_label_columns
            if column in evaluation_data.columns
        ]

        if not existing_labels:

            st.warning(
                "No actual label column was found. "
                "Please upload a labelled dataset containing a Class "
                "or equivalent actual label column."
            )

        else:

            if "selected_label" not in locals():

                selected_label = st.selectbox(
                    "Select actual label column",
                    existing_labels,
                    key="evaluation_label_column"
                )

            if all(
                feature in evaluation_data.columns
                for feature in features
            ):

                try:

                    y_true = pd.to_numeric(
                        evaluation_data[selected_label],
                        errors="coerce"
                    )

                    valid_rows = y_true.notna()

                    evaluation_data = evaluation_data.loc[
                        valid_rows
                    ].copy()

                    y_true = y_true.loc[valid_rows].astype(int)

                    if not set(y_true.unique()).issubset({0, 1}):

                        st.error(
                            "Actual labels must use 0 for Normal "
                            "and 1 for Fraud."
                        )

                    elif len(y_true.unique()) < 2:

                        st.error(
                            "The evaluation dataset must contain "
                            "both Normal and Fraud examples."
                        )

                    else:

                        X_test = evaluation_data[features].apply(
                            pd.to_numeric,
                            errors="coerce"
                        )

                        if X_test.isna().any().any():

                            st.error(
                                "Evaluation dataset contains missing "
                                "or non-numeric feature values."
                            )

                        else:

                            scores = model.decision_function(
                                X_test
                            )

                            y_pred = np.where(
                                scores < threshold,
                                1,
                                0
                            )

                            # Lower Isolation Forest scores indicate
                            # more anomalous transactions.
                            anomaly_scores = -scores

                            precision = precision_score(
                                y_true,
                                y_pred,
                                zero_division=0
                            )

                            recall = recall_score(
                                y_true,
                                y_pred,
                                zero_division=0
                            )

                            f1 = f1_score(
                                y_true,
                                y_pred,
                                zero_division=0
                            )

                            roc_auc = roc_auc_score(
                                y_true,
                                anomaly_scores
                            )

                            st.success(
                                "Model evaluation completed."
                            )

                            st.divider()

                            st.markdown("### Performance Metrics")

                            col1, col2, col3, col4 = st.columns(4)

                            col1.metric(
                                "Precision",
                                f"{precision * 100:.2f}%"
                            )

                            col2.metric(
                                "Recall",
                                f"{recall * 100:.2f}%"
                            )

                            col3.metric(
                                "F1 Score",
                                f"{f1 * 100:.2f}%"
                            )

                            col4.metric(
                                "ROC-AUC",
                                f"{roc_auc * 100:.2f}%"
                            )

                            st.divider()

                            st.markdown("### Confusion Matrix")

                            cm = confusion_matrix(
                                y_true,
                                y_pred,
                                labels=[0, 1]
                            )

                            cm_df = pd.DataFrame(
                                cm,
                                index=[
                                    "Actual Normal",
                                    "Actual Fraud"
                                ],
                                columns=[
                                    "Predicted Normal",
                                    "Predicted Fraud"
                                ]
                            )

                            st.dataframe(
                                cm_df,
                                use_container_width=True
                            )

                            tn, fp, fn, tp = cm.ravel()

                            col1, col2, col3, col4 = st.columns(4)

                            col1.metric("True Negative", f"{tn:,}")
                            col2.metric("False Positive", f"{fp:,}")
                            col3.metric("False Negative", f"{fn:,}")
                            col4.metric("True Positive", f"{tp:,}")

                            st.divider()

                            st.markdown("### ROC Curve")

                            fpr, tpr, thresholds_roc = roc_curve(
                                y_true,
                                anomaly_scores
                            )

                            fig, ax = plt.subplots(
                                figsize=(8, 5)
                            )

                            ax.plot(
                                fpr,
                                tpr,
                                label=f"ROC-AUC = {roc_auc:.4f}",
                                linewidth=2
                            )

                            ax.plot(
                                [0, 1],
                                [0, 1],
                                linestyle="--",
                                label="Random Classifier"
                            )

                            ax.set_xlabel(
                                "False Positive Rate"
                            )

                            ax.set_ylabel(
                                "True Positive Rate"
                            )

                            ax.set_title(
                                "Receiver Operating Characteristic"
                            )

                            ax.legend()

                            ax.grid(
                                alpha=0.25
                            )

                            st.pyplot(fig)

                            plt.close(fig)

                            st.divider()

                            st.markdown("### Metric Interpretation")

                            st.write(
                                f"""
                                - **Precision ({precision * 100:.2f}%):**
                                  Of all transactions flagged as fraud,
                                  this is the percentage that were actually fraud.
                                - **Recall ({recall * 100:.2f}%):**
                                  Of all actual fraud transactions,
                                  this is the percentage detected by the model.
                                - **F1 Score ({f1 * 100:.2f}%):**
                                  The harmonic mean of precision and recall.
                                - **ROC-AUC ({roc_auc * 100:.2f}%):**
                                  Measures how well the anomaly scores rank
                                  fraudulent transactions above normal transactions.
                                """
                            )

                except Exception as exc:

                    st.error(
                        f"Model evaluation failed: {exc}"
                    )

            else:

                missing = [
                    feature for feature in features
                    if feature not in evaluation_data.columns
                ]

                st.error(
                    "Missing required model features: "
                    + ", ".join(missing)
                )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------
    st.divider()

    st.markdown("### Model Information")

    col1, col2 = st.columns(2)

    with col1:

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

    with col2:

        st.markdown(
            f"""
            **Number of Model Features**

            {len(features)}

            **Decision Threshold**

            {threshold:.6f}

            **Model File**

            {model_file.name}
            """
        )


# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "Credit Card Fraud Detection System | "
    "Isolation Forest Machine Learning Model"
)
