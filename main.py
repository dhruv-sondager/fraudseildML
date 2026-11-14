"""Credit Card Fraud Detection System using Machine Learning.

Enterprise Fintech Anomaly Detection Application.
Powered by Unsupervised Isolation Forest on 30 Transformed Features.
"""

from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.model_utils import (
    load_trained_model,
    compute_evaluation_metrics,
)
from src.data_loader import (
    load_or_compute_full_analysis,
    load_test_evaluation_data,
    load_raw_dataset,
    find_existing_dataset,
)
from src.simulation_engine import TransactionSimulator
from src.ui_components import (
    inject_custom_css,
    render_top_header,
    render_top_navigation,
    render_page_header,
    render_kpi_card,
    render_footer,
    plot_normal_vs_suspicious,
    plot_anomaly_score_distribution,
    plot_amount_distribution,
    plot_anomaly_vs_amount,
    plot_hourly_trend,
    plot_confusion_matrix_heatmap,
    plot_roc_curve_chart,
    plot_precision_recall_chart,
    plot_scores_by_actual_label,
    plot_simulation_timeline,
    plot_score_gauge,
    plot_pca_feature_bars,
    COLOR_NAVY,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_AMBER,
)

# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="FraudShield ML | Credit Card Anomaly Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Theme (Light Mode / Dark Mode)
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
current_theme = st.session_state["theme"]

# Inject theme-aware fintech styling
inject_custom_css(current_theme)

# ============================================================
# MODEL & CACHED DATA INITIALIZATION
# ============================================================
@st.cache_resource(show_spinner="Loading trained Isolation Forest model...")
def get_cached_model():
    """Load model once across user sessions."""
    return load_trained_model()


@st.cache_data(show_spinner="Accessing transaction dataset analysis...")
def get_cached_analysis():
    """Load or retrieve precomputed full dataset analysis."""
    return load_or_compute_full_analysis(force_recompute=False)


@st.cache_data(show_spinner=False)
def get_cached_raw_df():
    """Load raw dataset for inspector lookup."""
    return load_raw_dataset()


@st.cache_data(show_spinner=False)
def get_cached_test_data():
    """Load held-out benchmark test predictions."""
    return load_test_evaluation_data()


# Attempt model load
try:
    model, threshold, features, model_path = get_cached_model()
except Exception as e:
    st.error(f"Failed to load Isolation Forest model: {e}")
    st.stop()

# Initialize session state variables
if "full_analysis" not in st.session_state:
    df_scored, load_time, load_source = get_cached_analysis()
    st.session_state["full_analysis"] = df_scored
    st.session_state["analysis_time"] = load_time
    st.session_state["analysis_source"] = load_source
    st.session_state["analysis_status"] = "Analysis completed"

if "simulator" not in st.session_state:
    st.session_state["simulator"] = TransactionSimulator()

if "sim_history" not in st.session_state:
    st.session_state["sim_history"] = pd.DataFrame()

if "sim_status" not in st.session_state:
    st.session_state["sim_status"] = "IDLE"  # IDLE, RUNNING, PAUSED, STOPPED

if "sim_elapsed" not in st.session_state:
    st.session_state["sim_elapsed"] = 0.0

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Dashboard"

if "manual_history" not in st.session_state:
    st.session_state["manual_history"] = []

# ============================================================
# APPLICATION HEADER & TOP NAVIGATION
# ============================================================
render_top_header(
    total_records=len(st.session_state["full_analysis"]),
    threshold_val=threshold,
    current_theme=current_theme,
)

active_page = render_top_navigation(
    active_page=st.session_state["active_page"],
    theme=current_theme,
)

# ============================================================
# PAGE ROUTING
# ============================================================

# ------------------------------------------------------------
# 1. DASHBOARD OVERVIEW
# ------------------------------------------------------------
if active_page == "Dashboard":
    render_page_header(
        title="Executive Fraud Detection Dashboard",
        subtitle="Operational intelligence and anomaly detection analytics across the stored credit card transaction portfolio.",
        badge_text="SYSTEM OVERVIEW & METRICS",
    )

    df_full = st.session_state["full_analysis"]
    total_tx = len(df_full)
    susp_mask = df_full["Prediction"] == "Suspicious"
    susp_count = int(susp_mask.sum())
    norm_count = total_tx - susp_count
    fraud_rate = (susp_count / total_tx * 100) if total_tx else 0.0

    total_amount = float(df_full["Amount"].sum())
    susp_amount = float(df_full.loc[susp_mask, "Amount"].sum())
    avg_susp_amt = float(df_full.loc[susp_mask, "Amount"].mean()) if susp_count else 0.0
    avg_norm_amt = float(df_full.loc[~susp_mask, "Amount"].mean()) if norm_count else 0.0

    # System Status Bar
    st.markdown(
        f"""
        <div class="status-box">
            <b>Active Monitoring Portfolio:</b> <code>{find_existing_dataset().name}</code> &nbsp;|&nbsp; 
            <b>Total Processed:</b> {total_tx:,} transactions &nbsp;|&nbsp; 
            <b>Model:</b> Isolation Forest (Unsupervised) &nbsp;|&nbsp; 
            <b>Threshold:</b> {threshold:.6f} &nbsp;|&nbsp; 
            <b>Status:</b> {st.session_state.get('analysis_status', 'Operational')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI Metrics Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("Total Transactions", f"{total_tx:,}", "Full portfolio count", "navy", theme=current_theme)
    with k2:
        render_kpi_card("Analysed Records", f"{total_tx:,}", "100% processed", "blue", theme=current_theme)
    with k3:
        render_kpi_card("Normal Transactions", f"{norm_count:,}", f"{(norm_count/total_tx*100):.2f}% portfolio", "green", theme=current_theme)
    with k4:
        render_kpi_card("Suspicious Flagged", f"{susp_count:,}", "Anomaly alerts", "red", theme=current_theme)
    with k5:
        render_kpi_card("Fraud Detection Rate", f"{fraud_rate:.2f}%", "Portfolio exposure", "red" if fraud_rate > 1 else "navy", theme=current_theme)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Visualizations Row 1
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_dist = plot_normal_vs_suspicious(norm_count, susp_count, theme=current_theme)
        st.pyplot(fig_dist, use_container_width=True)
    with col_v2:
        fig_scores = plot_anomaly_score_distribution(
            df_full["Anomaly_Score"].values,
            threshold,
            theme=current_theme,
        )
        st.pyplot(fig_scores, use_container_width=True)

    # Visualizations Row 2
    col_v3, col_v4 = st.columns(2)
    with col_v3:
        fig_amounts = plot_amount_distribution(df_full["Amount"], df_full["Prediction"], theme=current_theme)
        st.pyplot(fig_amounts, use_container_width=True)
    with col_v4:
        fig_scatter = plot_anomaly_vs_amount(
            df_full["Amount"],
            df_full["Anomaly_Score"].values,
            df_full["Prediction"],
            threshold,
            theme=current_theme,
        )
        st.pyplot(fig_scatter, use_container_width=True)

    # Visualizations Row 3: Hourly Monitoring Trend
    if "Time" in df_full.columns:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        fig_trend = plot_hourly_trend(df_full["Time"], df_full["Prediction"], theme=current_theme)
        st.pyplot(fig_trend, use_container_width=True)

    # Financial Exposure Summary Cards
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="fintech-surface">
            <div style="font-weight: 700; color: var(--text-primary); font-size: 1.05rem; margin-bottom: 0.75rem;">
                Financial Exposure & Risk Summary
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 2rem; font-size: 0.9rem;">
                <div>
                    <span style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; font-weight: 600;">Total Portfolio Value</span><br>
                    <b style="font-size: 1.25rem; color: var(--text-primary);">${total_amount:,.2f}</b>
                </div>
                <div>
                    <span style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; font-weight: 600;">Suspicious Volume Exposure</span><br>
                    <b style="font-size: 1.25rem; color: var(--red);">${susp_amount:,.2f}</b> 
                    <span style="font-size: 0.8rem; color: var(--text-muted);">({(susp_amount/total_amount*100):.2f}% of total)</span>
                </div>
                <div>
                    <span style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; font-weight: 600;">Avg Suspicious Amount</span><br>
                    <b style="font-size: 1.25rem; color: var(--text-primary);">${avg_susp_amt:,.2f}</b>
                </div>
                <div>
                    <span style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; font-weight: 600;">Avg Normal Amount</span><br>
                    <b style="font-size: 1.25rem; color: var(--text-primary);">${avg_norm_amt:,.2f}</b>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# 2. TRANSACTION MONITORING
# ------------------------------------------------------------
elif active_page == "Transaction Monitoring":
    render_page_header(
        title="Transaction Monitoring & Triage",
        subtitle="Inspect, filter, search and investigate credit card transactions evaluated by the Isolation Forest detector.",
        badge_text="TRANSACTION SURVEILLANCE",
    )

    df_full = st.session_state["full_analysis"]
    total_tx = len(df_full)
    susp_mask = df_full["Prediction"] == "Suspicious"
    susp_count = int(susp_mask.sum())
    norm_count = total_tx - susp_count
    fraud_rate = (susp_count / total_tx * 100) if total_tx else 0.0

    # Top Control Bar: Status & Re-run Button
    c_status, c_btn = st.columns([3, 1])
    with c_status:
        st.markdown(
            f"""
            <div class="status-box" style="margin-bottom: 0.5rem;">
                <b>Dataset:</b> <code>{find_existing_dataset().name}</code> ({total_tx:,} records) &nbsp;|&nbsp; 
                <b>Status:</b> <span class="badge-normal">{st.session_state.get('analysis_status', 'Analysis completed')}</span> &nbsp;|&nbsp; 
                <b>Execution Time:</b> {st.session_state.get('analysis_time', 0.12):.2f}s
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_btn:
        if st.button("Run Full Detection", type="primary", use_container_width=True):
            with st.spinner(f"Scoring all {total_tx:,} records with Isolation Forest..."):
                t_start = time.perf_counter()
                df_scored_new, elapsed_new, src = load_or_compute_full_analysis(force_recompute=True)
                st.session_state["full_analysis"] = df_scored_new
                st.session_state["analysis_time"] = elapsed_new
                st.session_state["analysis_status"] = f"Analysis completed ({elapsed_new:.2f}s)"
                st.success(f"Successfully analysed {len(df_scored_new):,} transactions in {elapsed_new:.2f} seconds!")
                st.rerun()

    # KPI Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_kpi_card("Total Processed", f"{total_tx:,}", "Full transaction set", "navy", theme=current_theme)
    with m2:
        render_kpi_card("Normal Transactions", f"{norm_count:,}", f"{(norm_count/total_tx*100):.2f}%", "green", theme=current_theme)
    with m3:
        render_kpi_card("Suspicious Flagged", f"{susp_count:,}", "Anomaly alerts", "red", theme=current_theme)
    with m4:
        render_kpi_card("Detection Rate", f"{fraud_rate:.2f}%", "Flagged ratio", "red" if fraud_rate > 1 else "navy", theme=current_theme)
    with m5:
        speed = int(total_tx / max(0.001, st.session_state.get('analysis_time', 1.0)))
        render_kpi_card("Scoring Velocity", f"{speed:,}/s", "Records per second", "blue", theme=current_theme)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Navigation Tabs
    tab_all, tab_suspicious, tab_inspector = st.tabs([
        "All Transactions Portfolio",
        f"Suspicious Queue ({susp_count:,} Alerts)",
        "Transaction Feature Inspector",
    ])

    # ------------------
    # TAB 1: ALL TRANSACTIONS
    # ------------------
    with tab_all:
        st.markdown(
            "<div style='font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.75rem;'>"
            "Search and filter the complete dataset. Table displays paginated records for optimal rendering."
            "</div>",
            unsafe_allow_html=True,
        )

        # Filters Bar
        f_c1, f_c2, f_c3, f_c4 = st.columns([1.5, 1.2, 1.5, 1.8])
        with f_c1:
            search_query = st.text_input(
                "Search Transaction ID",
                placeholder="e.g. TXN-000100",
                key="search_all",
            )
        with f_c2:
            status_filter = st.selectbox(
                "Filter Status",
                ["All", "Suspicious Only", "Normal Only"],
                index=0,
                key="status_filter",
            )
        with f_c3:
            sort_by = st.selectbox(
                "Sort Results",
                [
                    "Anomaly Score (Highest First)",
                    "Anomaly Score (Lowest First)",
                    "Amount (Highest First)",
                    "Amount (Lowest First)",
                    "Transaction ID (Default)",
                ],
                index=0,
                key="sort_all",
            )
        with f_c4:
            amt_min = float(df_full["Amount"].min())
            amt_max = float(df_full["Amount"].max())
            selected_amt = st.slider(
                "Amount Range ($)",
                min_value=round(amt_min, 2),
                max_value=round(amt_max, 2),
                value=(round(amt_min, 2), round(amt_max, 2)),
                key="slider_amt",
            )

        # Apply Filters
        filtered_df = df_full.copy()
        if search_query.strip():
            filtered_df = filtered_df[
                filtered_df["Transaction_ID"].str.contains(search_query.strip(), case=False, na=False)
            ]
        if status_filter == "Suspicious Only":
            filtered_df = filtered_df[filtered_df["Prediction"] == "Suspicious"]
        elif status_filter == "Normal Only":
            filtered_df = filtered_df[filtered_df["Prediction"] == "Normal"]

        filtered_df = filtered_df[
            (filtered_df["Amount"] >= selected_amt[0]) & (filtered_df["Amount"] <= selected_amt[1])
        ]

        # Sorting
        if sort_by == "Anomaly Score (Highest First)":
            filtered_df = filtered_df.sort_values(by="Anomaly_Score", ascending=False)
        elif sort_by == "Anomaly Score (Lowest First)":
            filtered_df = filtered_df.sort_values(by="Anomaly_Score", ascending=True)
        elif sort_by == "Amount (Highest First)":
            filtered_df = filtered_df.sort_values(by="Amount", ascending=False)
        elif sort_by == "Amount (Lowest First)":
            filtered_df = filtered_df.sort_values(by="Amount", ascending=True)

        st.caption(f"Showing {len(filtered_df):,} matching transactions out of {total_tx:,} total.")

        # Pagination Controls
        page_size = 50
        total_pages = max(1, int(np.ceil(len(filtered_df) / page_size)))
        p_col1, p_col2 = st.columns([1, 4])
        with p_col1:
            current_page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key="page_num_all",
            )
        with p_col2:
            st.markdown(
                f"<div style='padding-top: 1.8rem; font-size: 0.85rem; color: var(--text-muted);'>"
                f"Page {current_page} of {total_pages} (Displaying up to {page_size} records per page)"
                f"</div>",
                unsafe_allow_html=True,
            )

        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        page_slice = filtered_df.iloc[start_idx:end_idx].copy()

        # Format display table
        display_table = page_slice[["Transaction_ID", "Time", "Amount", "Anomaly_Score", "Prediction"]].copy()
        display_table["Amount"] = display_table["Amount"].map(lambda x: f"${x:,.2f}")
        display_table["Anomaly_Score"] = display_table["Anomaly_Score"].map(lambda x: f"{x:.6f}")
        display_table["Status"] = display_table["Prediction"].map(
            lambda x: "🚨 Suspicious" if x == "Suspicious" else "✅ Normal"
        )

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
            height=400,
        )

        # Download CSV
        st.download_button(
            label="Download Filtered Results (CSV)",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="credit_card_fraud_detection_results.csv",
            mime="text/csv",
            use_container_width=False,
        )

    # ------------------
    # TAB 2: SUSPICIOUS QUEUE
    # ------------------
    with tab_suspicious:
        st.markdown(
            f"""
            <div class="status-box" style="border-left-color: var(--red);">
                <b>Priority Investigation Queue:</b> Model flagged <b>{susp_count:,} anomalous transactions</b> 
                exceeding the decision threshold (<code>{threshold:.6f}</code>). 
                Transactions are ranked by anomaly score to prioritize manual triage.
            </div>
            """,
            unsafe_allow_html=True,
        )

        suspicious_df = df_full[df_full["Prediction"] == "Suspicious"].sort_values(
            by="Anomaly_Score", ascending=False
        ).copy()

        if suspicious_df.empty:
            st.success("No suspicious transactions detected.")
        else:
            s_col1, s_col2 = st.columns([1, 1])
            with s_col1:
                st.markdown(f"**High-Risk Alerts Count:** `{len(suspicious_df):,}`")
            with s_col2:
                st.markdown(f"**Total Suspicious Exposure:** `${suspicious_df['Amount'].sum():,.2f}`")

            # Table of suspicious transactions
            susp_display = suspicious_df.head(100)[["Transaction_ID", "Time", "Amount", "Anomaly_Score", "Prediction"]].copy()
            susp_display["Amount"] = susp_display["Amount"].map(lambda x: f"${x:,.2f}")
            susp_display["Anomaly_Score"] = susp_display["Anomaly_Score"].map(lambda x: f"{x:.6f}")
            susp_display["Threshold Margin"] = (
                suspicious_df.head(100)["Anomaly_Score"] - threshold
            ).map(lambda x: f"+{x:.6f}")

            st.dataframe(
                susp_display,
                use_container_width=True,
                hide_index=True,
                height=420,
            )
            st.caption(f"Displaying top 100 highest-risk anomalies. Download contains all {len(suspicious_df):,} records.")

            st.download_button(
                label=f"Download All {len(suspicious_df):,} Suspicious Transactions (CSV)",
                data=suspicious_df.to_csv(index=False).encode("utf-8"),
                file_name="suspicious_transactions_queue.csv",
                mime="text/csv",
            )

    # ------------------
    # TAB 3: TRANSACTION FEATURE INSPECTOR
    # ------------------
    with tab_inspector:
        st.markdown(
            "<div style='font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.75rem;'>"
            "Select or search any transaction to inspect its full 30-dimensional feature vector, "
            "anomaly score breakdown, and risk assessment."
            "</div>",
            unsafe_allow_html=True,
        )

        raw_df = get_cached_raw_df()

        # Selector
        top_suspicious_ids = df_full[df_full["Prediction"] == "Suspicious"]["Transaction_ID"].head(25).tolist()
        inspect_id = st.selectbox(
            "Select Transaction to Inspect (or enter ID below)",
            options=top_suspicious_ids,
            index=0,
            key="inspector_select",
        )

        manual_id = st.text_input("Or enter custom Transaction ID (e.g. TXN-000042):", value="", key="custom_inspect_id")
        final_id = manual_id.strip() if manual_id.strip() else inspect_id

        # Lookup transaction index
        try:
            tx_num = int(final_id.replace("TXN-", ""))
            tx_idx = tx_num - 1
            if 0 <= tx_idx < len(df_full):
                row_meta = df_full.iloc[tx_idx]
                row_raw = raw_df.iloc[tx_idx]

                score = float(row_meta["Anomaly_Score"])
                pred = str(row_meta["Prediction"])
                margin = score - threshold
                is_susp = (pred == "Suspicious")

                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                
                # Header card for transaction
                st.markdown(
                    f"""
                    <div class="detail-panel" style="border-left: 4px solid {'var(--red)' if is_susp else 'var(--green)'};">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.25rem; font-weight: 750; color: var(--text-primary);">{row_meta['Transaction_ID']}</span>
                            <span class="{'badge-suspicious' if is_susp else 'badge-normal'}">{pred.upper()}</span>
                        </div>
                        <div style="display: flex; flex-wrap: wrap; gap: 2rem; font-size: 0.9rem;">
                            <div><b>Amount:</b> ${row_meta['Amount']:,.2f}</div>
                            <div><b>Time:</b> {row_meta['Time']}s ({row_meta['Time']/3600:.2f} hrs)</div>
                            <div><b>Anomaly Score:</b> <code>{score:.6f}</code></div>
                            <div><b>Decision Threshold:</b> <code>{threshold:.6f}</code></div>
                            <div><b>Margin vs Threshold:</b> <b style="color: {'var(--red)' if is_susp else 'var(--green)'};">{'+' if margin >= 0 else ''}{margin:.6f}</b></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Surveillance Notice
                st.markdown(
                    f"""
                    <div class="status-box" style="margin-top: 0.75rem;">
                        <b>Surveillance Notice:</b> An anomaly score of <code>{score:.6f}</code> {'exceeds' if is_susp else 'is within'} the calibrated threshold (<code>{threshold:.6f}</code>). 
                        Flagged transactions indicate multivariate feature deviation from normal baseline patterns and are prioritized for investigative triage.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # PCA Feature Values Bar Chart
                pca_features = [f"V{i}" for i in range(1, 29) if f"V{i}" in row_raw]
                pca_values = [float(row_raw[feat]) for feat in pca_features]

                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("#### PCA Transformed Features (V1–V28)")
                
                fig_pca = plot_pca_feature_bars(pca_features, pca_values, row_meta['Transaction_ID'], theme=current_theme)
                st.pyplot(fig_pca, use_container_width=True)

            else:
                st.warning(f"Transaction index {tx_num} out of valid range (1 to {len(df_full):,}).")
        except Exception as e:
            st.error(f"Could not parse transaction ID: {e}")


# ------------------------------------------------------------
# 3. LIVE TRANSACTION SIMULATION
# ------------------------------------------------------------
elif active_page in ("Live Simulation", "Live Transaction Simulation"):
    render_page_header(
        title="Live Financial Stream Simulation",
        subtitle="Simulate incoming credit card transaction batches, score them through the model, and monitor live alerts.",
        badge_text="REAL-TIME STREAM SIMULATOR",
    )

    # Simulation Environment Notice
    st.markdown(
        f"""
        <div class="status-box" style="border-left-color: var(--amber);">
            <b>SIMULATION ENVIRONMENT NOTICE:</b><br>
            This module generates <b>complete synthetic feature vectors [Time, V1–V28, Amount]</b> 
            synthesized from the empirical multidimensional distribution of actual transactions with Gaussian perturbation. 
            All generated records are legitimately scored in real time by the trained <b>Isolation Forest</b> model.<br>
            <i>This is an interactive simulation environment and does NOT represent an actual connection to a bank or live payment gateway.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

    simulator: TransactionSimulator = st.session_state["simulator"]
    sim_history = st.session_state["sim_history"]

    # Simulation Controls Deck
    c_ctrl, c_params = st.columns([2.5, 2])

    with c_params:
        st.markdown(
            f"""
            <div class="fintech-surface" style="padding: 0.9rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-primary); text-transform: uppercase; margin-bottom: 0.4rem;">
                    STREAM CONFIGURATION
                </div>
            """,
            unsafe_allow_html=True,
        )
        batch_size = st.slider("Batch Size (Records per tick)", min_value=1, max_value=25, value=5, step=1, key="sim_batch_size")
        anomaly_ratio = st.slider("Anomaly / Fraud Injection Rate (%)", min_value=0, max_value=40, value=15, step=5, key="sim_anomaly_ratio") / 100.0
        st.markdown("</div>", unsafe_allow_html=True)

    with c_ctrl:
        st.markdown(
            f"""
            <div class="fintech-surface" style="padding: 0.9rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-primary); text-transform: uppercase; margin-bottom: 0.4rem;">
                    ENGINE CONTROLS & STATUS
                </div>
            """,
            unsafe_allow_html=True,
        )
        status_color = {
            "IDLE": "badge-info",
            "RUNNING": "badge-normal",
            "PAUSED": "badge-info",
            "STOPPED": "badge-suspicious",
        }.get(st.session_state["sim_status"], "badge-info")

        st.markdown(
            f"<b>Engine State:</b> <span class='{status_color}'>{st.session_state['sim_status']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("Generate Step", type="primary", use_container_width=True, key="btn_sim_step"):
                st.session_state["sim_status"] = "RUNNING"
                new_batch = simulator.generate_batch(
                    batch_size=batch_size,
                    anomaly_ratio=anomaly_ratio,
                    current_seq=len(sim_history),
                    sim_elapsed_seconds=st.session_state["sim_elapsed"],
                )
                if not new_batch.empty:
                    st.session_state["sim_history"] = pd.concat([sim_history, new_batch], ignore_index=True)
                    st.session_state["sim_elapsed"] += float(new_batch["Time"].max() - new_batch["Time"].min() + 1.0)
                st.rerun()

        with b2:
            if st.button("Generate 5x", use_container_width=True, key="btn_sim_multi"):
                st.session_state["sim_status"] = "RUNNING"
                with st.spinner("Streaming 5 batches..."):
                    for _ in range(5):
                        new_batch = simulator.generate_batch(
                            batch_size=batch_size,
                            anomaly_ratio=anomaly_ratio,
                            current_seq=len(st.session_state["sim_history"]),
                            sim_elapsed_seconds=st.session_state["sim_elapsed"],
                        )
                        st.session_state["sim_history"] = pd.concat([st.session_state["sim_history"], new_batch], ignore_index=True)
                        st.session_state["sim_elapsed"] += float(new_batch["Time"].max() - new_batch["Time"].min() + 1.0)
                st.rerun()

        with b3:
            if st.button("Pause", use_container_width=True, key="btn_sim_pause"):
                st.session_state["sim_status"] = "PAUSED"
                st.rerun()

        with b4:
            if st.button("Stop", use_container_width=True, key="btn_sim_stop"):
                st.session_state["sim_status"] = "STOPPED"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Simulation Telemetry Metrics
    sim_total = len(st.session_state["sim_history"])
    if sim_total > 0:
        sim_susp_mask = st.session_state["sim_history"]["Prediction"] == "Suspicious"
        sim_susp_count = int(sim_susp_mask.sum())
        sim_norm_count = sim_total - sim_susp_count
        sim_rate = (sim_susp_count / sim_total * 100) if sim_total else 0.0
    else:
        sim_susp_count = 0
        sim_norm_count = 0
        sim_rate = 0.0

    sm1, sm2, sm3, sm4 = st.columns(4)
    with sm1:
        render_kpi_card("Generated Records", f"{sim_total:,}", "Synthetic stream total", "navy", theme=current_theme)
    with sm2:
        render_kpi_card("Normal Simulated", f"{sim_norm_count:,}", f"{(sim_norm_count/max(1, sim_total)*100):.1f}%", "green", theme=current_theme)
    with sm3:
        render_kpi_card("Suspicious Flagged", f"{sim_susp_count:,}", "Anomaly triggers", "red", theme=current_theme)
    with sm4:
        render_kpi_card("Simulation Alert Rate", f"{sim_rate:.1f}%", "Trigger frequency", "red" if sim_rate > 10 else "blue", theme=current_theme)

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Live Visualizations & Feed
    if sim_total > 0:
        sim_df = st.session_state["sim_history"]

        # Stream Anomaly Score Chart
        fig_sim = plot_simulation_timeline(sim_df, threshold, theme=current_theme)
        st.pyplot(fig_sim, use_container_width=True)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Live Simulated Transaction Feed (Latest 25 Records)")

        # Display table of latest transactions
        feed_slice = sim_df.tail(25)[["Simulated_ID", "Time", "Amount", "Anomaly_Score", "Prediction"]].iloc[::-1].copy()
        feed_slice["Amount"] = feed_slice["Amount"].map(lambda x: f"${x:,.2f}")
        feed_slice["Anomaly_Score"] = feed_slice["Anomaly_Score"].map(lambda x: f"{x:.6f}")
        feed_slice["Alert Status"] = feed_slice["Prediction"].map(
            lambda x: "🚨 SUSPICIOUS" if x == "Suspicious" else "✅ Normal"
        )

        st.dataframe(
            feed_slice,
            use_container_width=True,
            hide_index=True,
        )

        # Clear Simulation Control with Confirmation
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        c_clear_box, c_clear_btn = st.columns([3, 1])
        with c_clear_box:
            confirm_clear = st.checkbox("Confirm clearing all simulated transaction records", key="confirm_clear_sim")
        with c_clear_btn:
            if st.button("Clear Simulation History", disabled=not confirm_clear, key="btn_clear_sim"):
                st.session_state["sim_history"] = pd.DataFrame()
                st.session_state["sim_status"] = "IDLE"
                st.session_state["sim_elapsed"] = 0.0
                st.success("Simulation records cleared successfully.")
                st.rerun()
    else:
        st.info("Click 'Generate Step' or 'Generate 5x' to initiate the live transaction simulation stream.")


# ------------------------------------------------------------
# 4. MODEL PERFORMANCE
# ------------------------------------------------------------
elif active_page == "Model Performance":
    render_page_header(
        title="Model Performance Evaluation",
        subtitle="Held-out test set evaluation of the trained Isolation Forest model on 56,746 benchmark transactions.",
        badge_text="BENCHMARK EVALUATION",
    )

    # Load authentic held-out test predictions
    try:
        test_df = get_cached_test_data()
        y_true = test_df["Actual"].values
        y_pred = test_df["Predicted"].values
        scores = test_df["Anomaly_Score"].values

        metrics = compute_evaluation_metrics(y_true, y_pred, scores)
    except Exception as e:
        st.error(f"Could not load test benchmark predictions: {e}")
        st.stop()

    # Benchmark Source Telemetry Notice
    st.markdown(
        f"""
        <div class="status-box">
            <b>Evaluation Benchmark Source:</b> <code>outputs/improved_predictions.csv</code> &nbsp;|&nbsp; 
            <b>Test Records:</b> {metrics['total_test']:,} &nbsp;|&nbsp; 
            <b>Ground Truth Fraud:</b> {metrics['total_fraud']} ({metrics['total_fraud']/metrics['total_test']*100:.3f}%) &nbsp;|&nbsp; 
            <b>Ground Truth Normal:</b> {metrics['total_normal']:,}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Primary Evaluation Metrics
    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    with pm1:
        render_kpi_card("Precision", f"{metrics['precision']*100:.2f}%", "TP / (TP + FP)", "navy", theme=current_theme)
    with pm2:
        render_kpi_card("Recall", f"{metrics['recall']*100:.2f}%", "TP / (TP + FN)", "green", theme=current_theme)
    with pm3:
        render_kpi_card("F1-Score", f"{metrics['f1']*100:.2f}%", "Harmonic mean", "blue", theme=current_theme)
    with pm4:
        render_kpi_card("ROC-AUC", f"{metrics['roc_auc']*100:.2f}%", "Ranking discrimination", "green", theme=current_theme)
    with pm5:
        render_kpi_card("PR-AUC", f"{metrics['pr_auc']:.4f}", "Precision-Recall Area", "navy", theme=current_theme)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Row 1: Confusion Matrix & ROC Curve
    c_cm, c_roc = st.columns([1, 1.25])
    with c_cm:
        fig_cm = plot_confusion_matrix_heatmap(metrics["cm"], theme=current_theme)
        st.pyplot(fig_cm, use_container_width=True)
    with c_roc:
        fig_roc = plot_roc_curve_chart(metrics["fpr"], metrics["tpr"], metrics["roc_auc"], theme=current_theme)
        st.pyplot(fig_roc, use_container_width=True)

    # Confusion matrix metrics breakdown cards
    cm_c1, cm_c2, cm_c3, cm_c4 = st.columns(4)
    with cm_c1:
        render_kpi_card("True Negatives (TN)", f"{metrics['tn']:,}", "Correctly identified normal", "green", theme=current_theme)
    with cm_c2:
        render_kpi_card("False Positives (FP)", f"{metrics['fp']:,}", "Normal flagged as suspicious", "red", theme=current_theme)
    with cm_c3:
        render_kpi_card("False Negatives (FN)", f"{metrics['fn']:,}", "Fraud missed by model", "red", theme=current_theme)
    with cm_c4:
        render_kpi_card("True Positives (TP)", f"{metrics['tp']:,}", "Confirmed fraud detected", "green", theme=current_theme)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Row 2: Precision-Recall Curve & Anomaly Score Separation
    c_pr, c_sep = st.columns([1, 1.25])
    with c_pr:
        fig_pr = plot_precision_recall_chart(
            metrics["pr_recall"], metrics["pr_precision"], metrics["pr_auc"], theme=current_theme
        )
        st.pyplot(fig_pr, use_container_width=True)
    with c_sep:
        fig_sep = plot_scores_by_actual_label(scores, y_true, threshold, theme=current_theme)
        st.pyplot(fig_sep, use_container_width=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Full Classification Report Table
    st.markdown("#### Model Classification Report")
    report_dict = metrics["report_dict"]

    report_df = pd.DataFrame({
        "Class": ["Normal (0)", "Fraud (1)", "Macro Average", "Weighted Average"],
        "Precision": [
            f"{report_dict['Normal']['precision']*100:.2f}%",
            f"{report_dict['Fraud']['precision']*100:.2f}%",
            f"{report_dict['macro avg']['precision']*100:.2f}%",
            f"{report_dict['weighted avg']['precision']*100:.2f}%",
        ],
        "Recall": [
            f"{report_dict['Normal']['recall']*100:.2f}%",
            f"{report_dict['Fraud']['recall']*100:.2f}%",
            f"{report_dict['macro avg']['recall']*100:.2f}%",
            f"{report_dict['weighted avg']['recall']*100:.2f}%",
        ],
        "F1-Score": [
            f"{report_dict['Normal']['f1-score']*100:.2f}%",
            f"{report_dict['Fraud']['f1-score']*100:.2f}%",
            f"{report_dict['macro avg']['f1-score']*100:.2f}%",
            f"{report_dict['weighted avg']['f1-score']*100:.2f}%",
        ],
        "Support": [
            f"{int(report_dict['Normal']['support']):,}",
            f"{int(report_dict['Fraud']['support']):,}",
            f"{int(report_dict['macro avg']['support']):,}",
            f"{int(report_dict['weighted avg']['support']):,}",
        ],
    })

    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True,
    )


# ------------------------------------------------------------
# 5. MANUAL TRANSACTION ANALYSIS
# ------------------------------------------------------------
elif active_page in ("Manual Analysis", "Manual Transaction Analysis"):
    render_page_header(
        title="Manual Transaction Analysis",
        subtitle="Enter a transaction amount and time to score it against the trained Isolation Forest model using a sample transaction profile from the dataset.",
        badge_text="SAMPLE-BASED ANALYSIS",
    )

    # Transparency Notice
    st.markdown(
        f"""
        <div class="status-box" style="border-left-color: var(--amber);">
            <b>Sample-Based Analysis Notice:</b><br>
            The Isolation Forest model requires 30 features (Time, V1–V28, Amount) for scoring. Since the PCA-transformed
            features V1–V28 are not directly observable, this tool uses an <b>existing transaction profile</b> from the
            dataset to supply V1–V28 values, combined with your entered Amount and Time.<br>
            <i>Results reflect how the model would score a transaction with the chosen profile's spending pattern at
            your specified amount and time. This is a diagnostic tool, not a production fraud check.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Analysis Form
    with st.form("manual_analysis_form"):
        st.markdown(
            f"<div style='font-weight: 700; color: var(--text-primary); font-size: 1.05rem; margin-bottom: 0.6rem;'>Transaction Parameters</div>",
            unsafe_allow_html=True,
        )

        form_c1, form_c2, form_c3 = st.columns([1.2, 1.2, 1.6])

        with form_c1:
            manual_amount = st.number_input(
                "Transaction Amount (₹)",
                min_value=0.0,
                max_value=1000000.0,
                value=500.0,
                step=10.0,
                format="%.2f",
                help="Enter the transaction amount in INR.",
            )

        with form_c2:
            manual_time_hours = st.number_input(
                "Transaction Time (Hours)",
                min_value=0.0,
                max_value=48.0,
                value=12.0,
                step=0.5,
                format="%.1f",
                help="Time elapsed since the first transaction in the dataset (0–48 hours).",
            )
            st.caption("Internally converted to seconds for model input.")

        with form_c3:
            profile_options = [
                "Sample Profile 1 (Standard Retail)",
                "Sample Profile 2 (E-commerce)",
                "Sample Profile 3 (Atypical Pattern)",
                "Random Sample Profile",
            ]
            selected_profile = st.selectbox(
                "Sample Transaction Profile",
                options=profile_options,
                index=0,
                help="Selects an existing dataset row to supply V1–V28 PCA features.",
            )

        submitted = st.form_submit_button("Analyse Transaction", type="primary", use_container_width=True)

    # Process submission
    if submitted:
        profile_map = {
            "Sample Profile 1 (Standard Retail)": 100,
            "Sample Profile 2 (E-commerce)": 500,
            "Sample Profile 3 (Atypical Pattern)": 164,
        }

        raw_df = get_cached_raw_df()

        if selected_profile == "Random Sample Profile":
            profile_idx = random.randint(0, len(raw_df) - 1)
            profile_label = f"Random (Row {profile_idx})"
        else:
            profile_idx = profile_map[selected_profile]
            profile_label = selected_profile

        sample_row = raw_df.iloc[profile_idx]

        # Build feature vector
        feature_dict = {}
        for feat in features:
            if feat == "Time":
                feature_dict["Time"] = manual_time_hours * 3600.0
            elif feat == "Amount":
                feature_dict["Amount"] = manual_amount
            else:
                feature_dict[feat] = float(sample_row[feat])

        input_df = pd.DataFrame([feature_dict])[features]

        # Score with model
        anomaly_score = float(-model.decision_function(input_df)[0])
        prediction = "Suspicious" if anomaly_score >= threshold else "Normal"
        margin = anomaly_score - threshold

        # Save to session history
        st.session_state["manual_history"].append({
            "Amount (₹)": f"₹{manual_amount:,.2f}",
            "Time (hrs)": f"{manual_time_hours:.1f}h",
            "Profile": profile_label,
            "Anomaly Score": f"{anomaly_score:.6f}",
            "Prediction": prediction,
            "Margin": f"{'+' if margin >= 0 else ''}{margin:.6f}",
        })

        # Display Results
        is_susp = (prediction == "Suspicious")
        result_border_color = "var(--red)" if is_susp else "var(--green)"
        result_badge_class = "badge-suspicious" if is_susp else "badge-normal"

        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="detail-panel" style="border-left: 4px solid {result_border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <span style="font-size: 1.2rem; font-weight: 750; color: var(--text-primary);">Analysis Result</span>
                    <span class="{result_badge_class}" style="font-size: 0.88rem; padding: 5px 16px;">{prediction.upper()}</span>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 2rem; font-size: 0.92rem;">
                    <div><b>Amount:</b> ₹{manual_amount:,.2f}</div>
                    <div><b>Time:</b> {manual_time_hours:.1f} hours ({manual_time_hours * 3600:.0f}s)</div>
                    <div><b>Profile:</b> {profile_label}</div>
                    <div><b>Anomaly Score:</b> <code>{anomaly_score:.6f}</code></div>
                    <div><b>Threshold:</b> <code>{threshold:.6f}</code></div>
                    <div><b>Margin:</b> <b style="color: {result_border_color};">{'+'if margin >= 0 else ''}{margin:.6f}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Score Gauge Visualization
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        fig_gauge = plot_score_gauge(anomaly_score, threshold, theme=current_theme)
        st.pyplot(fig_gauge, use_container_width=True)

        # Explanation
        if is_susp:
            st.markdown(
                f"""
                <div class="status-box" style="border-left-color: var(--red);">
                    <b>Suspicious Result:</b> The constructed transaction profile has an anomaly score of
                    <code>{anomaly_score:.6f}</code> which <b>exceeds</b> the calibrated decision threshold
                    (<code>{threshold:.6f}</code>). This indicates that the combination of the selected PCA
                    feature profile with the entered amount and time produces multivariate characteristics
                    that deviate significantly from normal transaction patterns learned by the model.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-box" style="border-left-color: var(--green);">
                    <b>Normal Result:</b> The constructed transaction profile has an anomaly score of
                    <code>{anomaly_score:.6f}</code> which is <b>below</b> the calibrated decision threshold
                    (<code>{threshold:.6f}</code>). The model considers this transaction's feature pattern
                    consistent with normal behaviour in the training data.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Session History
    if st.session_state.get("manual_history"):
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-weight: 700; color: var(--text-primary); font-size: 1.05rem; margin-bottom: 0.6rem;'>Analysis Session History</div>",
            unsafe_allow_html=True,
        )

        history_df = pd.DataFrame(st.session_state["manual_history"])
        history_df.insert(0, "#", range(1, len(history_df) + 1))
        history_df["Status"] = history_df["Prediction"].map(
            lambda x: "🚨 Suspicious" if x == "Suspicious" else "✅ Normal"
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        col_clear_confirm, col_clear_btn = st.columns([3, 1])
        with col_clear_confirm:
            confirm_clear_manual = st.checkbox("Confirm clearing manual analysis history", key="confirm_clear_manual")
        with col_clear_btn:
            if st.button("Clear History", disabled=not confirm_clear_manual, key="btn_clear_manual"):
                st.session_state["manual_history"] = []
                st.rerun()

# ============================================================
# APPLICATION FOOTER
# ============================================================
render_footer(threshold=threshold, theme=current_theme)
