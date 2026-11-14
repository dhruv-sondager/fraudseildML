"""Professional UI/UX and Data Visualization Components for FraudShield ML.

Comprehensive Light Mode & Dark Mode Fintech Theme System:
- Deep navy/charcoal for Dark Mode, crisp porcelain/slate for Light Mode
- High contrast, legible typography and balanced spacing
- Theme-aware Matplotlib visualizations with zero hardcoded background clashes
- Horizontal navigation bar with five directly clickable buttons
- Polished telemetry indicators and theme switcher
"""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CENTRAL THEME PALETTES
# ============================================================

THEMES = {
    "light": {
        "name": "light",
        "bg_app": "#f4f7fb",
        "bg_card": "#ffffff",
        "bg_card_secondary": "#f8fafc",
        "bg_header": "rgba(244, 247, 251, 0.96)",
        "bg_nav_container": "#edf3fa",
        "bg_nav_item": "#ffffff",
        "border": "#d0dfea",
        "border_subtle": "#e2e8f0",
        "text_primary": "#0f2b48",
        "text_secondary": "#334e68",
        "text_muted": "#64748b",
        "accent_blue": "#1e5b94",
        "accent_blue_light": "#e8f1fa",
        "active_nav_bg": "#0f2b48",
        "active_nav_text": "#ffffff",
        "status_green": "#107e54",
        "status_green_bg": "#e6f6ee",
        "status_green_border": "#b7ebd3",
        "status_red": "#c53030",
        "status_red_bg": "#fde8e8",
        "status_red_border": "#fecaca",
        "status_amber": "#b45309",
        "status_amber_bg": "#fef3c7",
        "status_amber_border": "#fde68a",
        # Matplotlib chart values
        "fig_facecolor": "#ffffff",
        "ax_facecolor": "#ffffff",
        "chart_text": "#0f2b48",
        "chart_text_muted": "#475569",
        "chart_grid": "#e2e8f0",
        "chart_border": "#cbd5e1",
        "chart_card_bg": "#ffffff",
    },
    "dark": {
        "name": "dark",
        "bg_app": "#0b1320",
        "bg_card": "#132238",
        "bg_card_secondary": "#182a44",
        "bg_header": "rgba(11, 19, 32, 0.96)",
        "bg_nav_container": "#101c2e",
        "bg_nav_item": "#1a2d47",
        "border": "#223854",
        "border_subtle": "#1c314b",
        "text_primary": "#f8fafc",
        "text_secondary": "#cbd5e1",
        "text_muted": "#94a3b8",
        "accent_blue": "#38bdf8",
        "accent_blue_light": "rgba(56, 189, 248, 0.14)",
        "active_nav_bg": "#2563eb",
        "active_nav_text": "#ffffff",
        "status_green": "#34d399",
        "status_green_bg": "rgba(16, 185, 129, 0.16)",
        "status_green_border": "rgba(16, 185, 129, 0.4)",
        "status_red": "#f87171",
        "status_red_bg": "rgba(239, 68, 68, 0.16)",
        "status_red_border": "rgba(239, 68, 68, 0.4)",
        "status_amber": "#fbbf24",
        "status_amber_bg": "rgba(245, 158, 11, 0.16)",
        "status_amber_border": "rgba(245, 158, 11, 0.4)",
        # Matplotlib chart values
        "fig_facecolor": "#132238",
        "ax_facecolor": "#132238",
        "chart_text": "#f8fafc",
        "chart_text_muted": "#cbd5e1",
        "chart_grid": "#223854",
        "chart_border": "#334e68",
        "chart_card_bg": "#182a44",
    },
}

# Backwards compatibility constants
COLOR_NAVY = "#0f2b48"
COLOR_BLUE = "#1e5b94"
COLOR_BLUE_LIGHT = "#e8f1fa"
COLOR_SLATE = "#475569"
COLOR_GREEN = "#107e54"
COLOR_GREEN_BG = "#e6f6ee"
COLOR_RED = "#c53030"
COLOR_RED_BG = "#fde8e8"
COLOR_AMBER = "#b45309"
COLOR_AMBER_BG = "#fef3c7"
COLOR_BG_CARD = "#ffffff"
COLOR_BORDER = "#d1dfed"


def get_chart_theme_config(theme: str = "light") -> Dict[str, Any]:
    """Retrieve theme-aware styling dictionary for Matplotlib figures."""
    t = THEMES.get(theme, THEMES["light"])
    return {
        "fig_facecolor": t["fig_facecolor"],
        "ax_facecolor": t["ax_facecolor"],
        "text_color": t["chart_text"],
        "text_muted": t["chart_text_muted"],
        "grid_color": t["chart_grid"],
        "border_color": t["chart_border"],
        "card_bg": t["chart_card_bg"],
        "green": t["status_green"],
        "red": t["status_red"],
        "amber": t["status_amber"],
        "blue": t["accent_blue"],
    }


def inject_custom_css(theme: str = "light"):
    """Inject polished, fintech-grade styling supporting both Light and Dark themes."""
    t = THEMES.get(theme, THEMES["light"])
    is_dark = (theme == "dark")

    st.markdown(
        f"""
        <style>
        /* Base typography & theme variables */
        :root {{
            --bg-app: {t['bg_app']};
            --bg-card: {t['bg_card']};
            --bg-card-sec: {t['bg_card_secondary']};
            --bg-header: {t['bg_header']};
            --bg-nav: {t['bg_nav_container']};
            --bg-nav-item: {t['bg_nav_item']};
            --border: {t['border']};
            --border-subtle: {t['border_subtle']};
            --text-primary: {t['text_primary']};
            --text-secondary: {t['text_secondary']};
            --text-muted: {t['text_muted']};
            --accent-blue: {t['accent_blue']};
            --accent-blue-light: {t['accent_blue_light']};
            --active-nav-bg: {t['active_nav_bg']};
            --active-nav-text: {t['active_nav_text']};
            --green: {t['status_green']};
            --green-bg: {t['status_green_bg']};
            --green-border: {t['status_green_border']};
            --red: {t['status_red']};
            --red-bg: {t['status_red_bg']};
            --red-border: {t['status_red_border']};
            --amber: {t['status_amber']};
            --amber-bg: {t['status_amber_bg']};
            --amber-border: {t['status_amber_border']};
        }}

        /* Application base background and text */
        html, body, [data-testid="stAppViewContainer"], .main, .block-container {{
            background-color: var(--bg-app) !important;
            color: var(--text-primary) !important;
        }}

        /* Streamlit Fixed Header clearance */
        header[data-testid="stHeader"] {{
            background: var(--bg-header) !important;
            backdrop-filter: blur(8px) !important;
            z-index: 99 !important;
        }}

        /* Main app container - generous top clearance so headers never clip */
        .block-container {{
            padding-top: 4.6rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 1400px !important;
            overflow: visible !important;
        }}

        /* Completely remove the left sidebar and collapse control */
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {{
            display: none !important;
        }}

        /* Global text and overflow safeguards */
        [data-testid="stMarkdownContainer"] {{
            overflow: visible !important;
        }}
        [data-testid="stMarkdownContainer"] p {{
            line-height: 1.55;
            margin-bottom: 0.5rem;
            color: inherit;
        }}

        /* Header typography */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            letter-spacing: -0.02em;
            line-height: 1.3 !important;
            color: var(--text-primary) !important;
            overflow: visible !important;
        }}

        /* Page Header Container - Zero negative margins, natural box flow */
        .page-header-box {{
            margin-top: 0.1rem;
            margin-bottom: 1.35rem;
            padding: 0;
            overflow: visible !important;
        }}

        .header-badge-row {{
            margin-bottom: 0.45rem;
            line-height: 1.35;
        }}

        .page-header-title {{
            color: var(--text-primary) !important;
            font-size: 2.1rem !important;
            font-weight: 700 !important;
            line-height: 1.25 !important;
            letter-spacing: -0.025em !important;
            margin: 0 0 0.45rem 0 !important;
            padding: 0 !important;
            overflow: visible !important;
        }}

        .page-header-subtitle {{
            font-size: 0.95rem !important;
            font-weight: 400 !important;
            line-height: 1.55 !important;
            color: var(--text-secondary) !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: visible !important;
        }}

        /* Top Horizontal Navigation Container Box */
        .top-nav-box {{
            background: var(--bg-nav);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 8px 10px;
            margin-bottom: 1.6rem;
            box-shadow: 0 1px 4px {'rgba(0, 0, 0, 0.25)' if is_dark else 'rgba(15, 43, 72, 0.05)'};
            overflow: visible !important;
        }}

        /* Navigation Buttons Styling */
        div[data-testid="stHorizontalBlock"]:has(button[key^="nav_btn_"]) {{
            gap: 8px !important;
        }}

        /* Top Branding Header */
        .top-branding-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1.2rem;
            padding-bottom: 1.1rem;
            margin-bottom: 1.1rem;
            border-bottom: 1px solid var(--border);
            overflow: visible !important;
        }}

        .top-branding-left {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}

        .top-eyebrow {{
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--accent-blue);
            letter-spacing: 0.09em;
            text-transform: uppercase;
            line-height: 1.3;
        }}

        .top-brand-title {{
            font-size: 1.85rem;
            font-weight: 800;
            color: var(--text-primary);
            line-height: 1.2;
            letter-spacing: -0.025em;
        }}

        .top-brand-subtitle {{
            font-size: 0.88rem;
            color: var(--text-secondary);
            line-height: 1.35;
        }}

        .top-telemetry-right {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            align-items: center;
        }}

        .telemetry-pill {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 5px 11px;
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-secondary);
            box-shadow: 0 1px 2px {'rgba(0,0,0,0.2)' if is_dark else 'rgba(15, 43, 72, 0.03)'};
            white-space: nowrap;
        }}

        .telemetry-pill-status {{
            background: var(--green-bg);
            border: 1px solid var(--green-border);
            border-radius: 6px;
            padding: 5px 11px;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--green);
            box-shadow: 0 1px 2px {'rgba(0,0,0,0.2)' if is_dark else 'rgba(16, 126, 84, 0.05)'};
            white-space: nowrap;
        }}

        /* Custom KPI Card */
        .fintech-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 1px 4px {'rgba(0, 0, 0, 0.2)' if is_dark else 'rgba(15, 43, 72, 0.04)'};
            margin-bottom: 0.75rem;
            overflow: visible !important;
        }}

        .fintech-card-label {{
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            line-height: 1.35;
            margin-bottom: 0.4rem;
        }}

        .fintech-card-value {{
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 0.35rem;
            overflow: visible !important;
        }}

        .fintech-card-subtitle {{
            font-size: 0.78rem;
            line-height: 1.35;
            color: var(--text-secondary);
        }}

        /* Status Badges */
        .badge-normal {{
            display: inline-block;
            background-color: var(--green-bg);
            color: var(--green);
            font-weight: 600;
            font-size: 0.75rem;
            line-height: 1.3;
            padding: 3px 10px;
            border-radius: 12px;
            border: 1px solid var(--green-border);
            vertical-align: middle;
            white-space: nowrap;
        }}

        .badge-suspicious {{
            display: inline-block;
            background-color: var(--red-bg);
            color: var(--red);
            font-weight: 600;
            font-size: 0.75rem;
            line-height: 1.3;
            padding: 3px 10px;
            border-radius: 12px;
            border: 1px solid var(--red-border);
            vertical-align: middle;
            white-space: nowrap;
        }}

        .badge-info {{
            display: inline-block;
            background-color: var(--accent-blue-light);
            color: var(--accent-blue);
            font-weight: 600;
            font-size: 0.75rem;
            line-height: 1.3;
            padding: 3px 10px;
            border-radius: 12px;
            border: 1px solid var(--border);
            vertical-align: middle;
            white-space: nowrap;
        }}

        /* Status Callout Box */
        .status-box {{
            background: var(--bg-card-sec);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent-blue);
            border-radius: 6px;
            padding: 0.85rem 1.15rem;
            margin-bottom: 1.2rem;
            font-size: 0.88rem;
            line-height: 1.55;
            color: var(--text-primary);
            overflow: visible !important;
        }}

        /* Details Inspector Panel */
        .detail-panel {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.15rem;
            box-shadow: 0 1px 3px {'rgba(0,0,0,0.2)' if is_dark else 'rgba(0,0,0,0.03)'};
            overflow: visible !important;
            color: var(--text-primary);
        }}

        /* Generic Fintech Surface */
        .fintech-surface {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.1rem 1.4rem;
            box-shadow: 0 1px 4px {'rgba(0,0,0,0.2)' if is_dark else 'rgba(15,43,72,0.04)'};
            color: var(--text-primary);
            margin-bottom: 0.8rem;
        }}

        /* Streamlit Button Overrides */
        div.stButton > button {{
            border-radius: 7px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            padding: 0.45rem 1.1rem !important;
            line-height: 1.4 !important;
            transition: all 0.15s ease-in-out !important;
        }}

        div.stButton > button[kind="secondary"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-secondary) !important;
        }}
        div.stButton > button[kind="secondary"]:hover {{
            background-color: var(--bg-card-sec) !important;
            border-color: var(--accent-blue) !important;
            color: var(--text-primary) !important;
        }}

        div.stButton > button[kind="primary"] {{
            background-color: var(--active-nav-bg) !important;
            border: 1px solid var(--active-nav-bg) !important;
            color: var(--active-nav-text) !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 6px {'rgba(37, 99, 235, 0.4)' if is_dark else 'rgba(15, 43, 72, 0.22)'} !important;
        }}

        /* Streamlit Input & Form Widgets */
        [data-baseweb="input"], [data-baseweb="base-input"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            color: var(--text-primary) !important;
        }}
        [data-baseweb="input"] input, [data-baseweb="base-input"] input {{
            color: var(--text-primary) !important;
            background-color: transparent !important;
        }}
        [data-baseweb="select"] > div {{
            background-color: var(--bg-card) !important;
            border-color: var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: 6px !important;
        }}
        [data-testid="stSelectbox"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stSlider"] label {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }}

        /* Streamlit Tabs */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 4px !important;
            gap: 4px !important;
        }}
        [data-testid="stTabs"] button[role="tab"] {{
            color: var(--text-secondary) !important;
            border-radius: 6px !important;
            padding: 6px 14px !important;
            border: none !important;
            background: transparent !important;
            font-weight: 550 !important;
        }}
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            color: var(--active-nav-text) !important;
            background-color: var(--active-nav-bg) !important;
            font-weight: 700 !important;
        }}

        /* Streamlit DataFrame */
        [data-testid="stDataFrame"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }}

        /* Streamlit Metric Overrides */
        [data-testid="stMetric"] {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px;
            padding: 14px 18px;
            box-shadow: 0 1px 4px {'rgba(0,0,0,0.2)' if is_dark else 'rgba(15, 43, 72, 0.04)'};
            overflow: visible !important;
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            color: var(--text-muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            line-height: 1.3 !important;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
            line-height: 1.25 !important;
            overflow: visible !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# APPLICATION HEADER & TOP NAVIGATION
# ============================================================

def render_top_header(total_records: int, threshold_val: float, current_theme: str = "light"):
    """Render the top fintech branding bar, telemetry pills, and theme switcher."""
    t = THEMES.get(current_theme, THEMES["light"])
    
    col_brand, col_actions = st.columns([1.6, 1.4])
    
    with col_brand:
        st.markdown(
            f"""
            <div class="top-branding-left">
                <div class="top-eyebrow">FINANCIAL AI MONITORING • ANOMALY DETECTION</div>
                <div class="top-brand-title">FraudShield ML</div>
                <div class="top-brand-subtitle">Credit Card Anomaly Detection System</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col_actions:
        # Theme switcher row
        t_col_space, t_col_light, t_col_dark = st.columns([1.4, 1.0, 1.0])
        with t_col_light:
            if st.button("☀️ Light", key="theme_btn_light", type="primary" if current_theme == "light" else "secondary", use_container_width=True):
                if current_theme != "light":
                    st.session_state["theme"] = "light"
                    st.rerun()
        with t_col_dark:
            if st.button("🌙 Dark", key="theme_btn_dark", type="primary" if current_theme == "dark" else "secondary", use_container_width=True):
                if current_theme != "dark":
                    st.session_state["theme"] = "dark"
                    st.rerun()
        
        # Telemetry pills
        st.markdown(
            f"""
            <div class="top-telemetry-right" style="justify-content: flex-end; margin-top: 6px;">
                <span class="telemetry-pill-status">● System Operational</span>
                <span class="telemetry-pill">Model: Isolation Forest (200 Trees)</span>
                <span class="telemetry-pill">Threshold: <code>{threshold_val:.6f}</code></span>
                <span class="telemetry-pill">Portfolio: {total_records:,} Records</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_top_navigation(active_page: str, theme: str = "light") -> str:
    """Render full-width horizontal navigation container with 5 individually clickable buttons.
    
    Strictly uses actual clickable buttons (no st.selectbox, st.radio, dropdowns, or hidden menus).
    """
    pages = [
        ("Dashboard", "📊 Dashboard"),
        ("Transaction Monitoring", "🔍 Transaction Monitoring"),
        ("Live Simulation", "⚡ Live Simulation"),
        ("Model Performance", "📈 Model Performance"),
        ("Manual Analysis", "🛡️ Manual Analysis"),
    ]

    st.markdown('<div class="top-nav-box">', unsafe_allow_html=True)
    cols = st.columns(5)
    for col, (page_key, page_label) in zip(cols, pages):
        with col:
            is_active = (active_page == page_key)
            if st.button(
                page_label,
                key=f"nav_btn_{page_key}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                if active_page != page_key:
                    st.session_state["active_page"] = page_key
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    return st.session_state.get("active_page", "Dashboard")


def render_page_header(title: str, subtitle: str, badge_text: str = "FINANCIAL INTELLIGENCE"):
    """Render a clean, non-clipping fintech page header."""
    badge_html = (
        f'<div class="header-badge-row"><span class="badge-info">{badge_text}</span></div>'
        if badge_text else ''
    )
    st.markdown(
        f"""
        <div class="page-header-box">
            {badge_html}
            <h1 class="page-header-title">{title}</h1>
            <p class="page-header-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    subtitle: str = "",
    status_color: str = "navy",
    theme: str = "light",
):
    """Render a styled fintech KPI card adapted to the active theme."""
    t = THEMES.get(theme, THEMES["light"])
    color_map = {
        "navy": t["text_primary"],
        "green": t["status_green"],
        "red": t["status_red"],
        "blue": t["accent_blue"],
    }
    val_color = color_map.get(status_color, t["text_primary"])

    st.markdown(
        f"""
        <div class="fintech-card">
            <div class="fintech-card-label">{label}</div>
            <div class="fintech-card-value" style="color: {val_color};">{value}</div>
            <div class="fintech-card-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer(threshold: float, theme: str = "light"):
    """Render the application footer."""
    st.markdown("---")
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: var(--text-muted); padding: 0.5rem 0;">
            <div>Credit Card Anomaly Detection System &bull; Isolation Forest Architecture (200 Trees)</div>
            <div>Trained on Scikit-Learn 1.9.1 &bull; Calibrated Threshold: <code>{threshold:.6f}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MATPLOTLIB THEME-AWARE VISUALIZATIONS
# ============================================================

def _style_chart_axes(ax: plt.Axes, cfg: Dict[str, Any]):
    """Apply consistent theme-aware styling to axes."""
    ax.set_facecolor(cfg["ax_facecolor"])
    ax.tick_params(colors=cfg["text_muted"], labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(cfg["border_color"])
        spine.set_linewidth(0.8)
    ax.xaxis.label.set_color(cfg["text_color"])
    ax.yaxis.label.set_color(cfg["text_color"])
    ax.title.set_color(cfg["text_color"])
    ax.grid(color=cfg["grid_color"], alpha=0.6, linestyle="--")
    ax.set_axisbelow(True)


def _style_chart_legend(ax: plt.Axes, cfg: Dict[str, Any], loc: str = "best"):
    """Apply theme-aware styling to legends."""
    legend = ax.legend(loc=loc, frameon=True, fontsize=8.5)
    if legend:
        frame = legend.get_frame()
        frame.set_facecolor(cfg["card_bg"])
        frame.set_edgecolor(cfg["border_color"])
        for text in legend.get_texts():
            text.set_color(cfg["text_color"])


def plot_normal_vs_suspicious(normal_count: int, suspicious_count: int, theme: str = "light") -> plt.Figure:
    """Plot transaction distribution comparison bar chart."""
    cfg = get_chart_theme_config(theme)
    total = normal_count + suspicious_count
    norm_pct = (normal_count / total * 100) if total else 0
    susp_pct = (suspicious_count / total * 100) if total else 0

    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=120, facecolor=cfg["fig_facecolor"])
    _style_chart_axes(ax, cfg)

    bars = ax.bar(
        ["Normal", "Suspicious"],
        [normal_count, suspicious_count],
        color=[cfg["green"], cfg["red"]],
        width=0.48,
        edgecolor=[cfg["border_color"], cfg["border_color"]],
        linewidth=1.0,
    )

    ax.set_ylabel("Transaction Count", fontsize=9.5, fontweight=600)
    ax.set_title("Transaction Classification Breakdown", fontsize=11, fontweight=600, pad=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax.grid(axis="y")

    # Annotate bar values
    for bar, pct in zip(bars, [norm_pct, susp_pct]):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (total * 0.015),
            f"{int(yval):,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight=600,
            color=cfg["text_color"],
        )

    ax.set_ylim(0, max(normal_count, suspicious_count) * 1.18)
    fig.tight_layout()
    return fig


def plot_anomaly_score_distribution(
    scores: np.ndarray,
    threshold: float,
    sample_size: int = 40000,
    theme: str = "light",
) -> plt.Figure:
    """Plot histogram of anomaly scores with calibrated decision threshold."""
    cfg = get_chart_theme_config(theme)
    if len(scores) > sample_size:
        plot_scores = np.random.choice(scores, size=sample_size, replace=False)
    else:
        plot_scores = scores

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=120, facecolor=cfg["fig_facecolor"])
    _style_chart_axes(ax, cfg)

    normal_scores = plot_scores[plot_scores < threshold]
    suspicious_scores = plot_scores[plot_scores >= threshold]

    ax.hist(
        normal_scores,
        bins=50,
        alpha=0.65,
        color=cfg["blue"],
        label=f"Normal (< {threshold:.4f})",
        density=False,
    )
    if len(suspicious_scores) > 0:
        ax.hist(
            suspicious_scores,
            bins=25,
            alpha=0.85,
            color=cfg["red"],
            label=f"Suspicious (>= {threshold:.4f})",
            density=False,
        )

    ax.axvline(
        threshold,
        color=cfg["amber"],
        linestyle="--",
        linewidth=2.0,
        label=f"Decision Threshold: {threshold:.4f}",
    )

    ax.set_xlabel("Anomaly Score: -decision_function(X) [Higher = More Anomalous]", fontsize=9.2)
    ax.set_ylabel("Frequency", fontsize=9.2)
    ax.set_title("Anomaly Score Distribution vs Calibrated Threshold", fontsize=11, fontweight=600, pad=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    _style_chart_legend(ax, cfg)
    fig.tight_layout()
    return fig


def plot_amount_distribution(amounts: pd.Series, predictions: pd.Series, theme: str = "light") -> plt.Figure:
    """Plot transaction amount distribution for normal vs suspicious transactions."""
    cfg = get_chart_theme_config(theme)
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=120, facecolor=cfg["fig_facecolor"])
    _style_chart_axes(ax, cfg)

    norm_amounts = pd.to_numeric(amounts[predictions == "Normal"], errors="coerce").dropna()
    susp_amounts = pd.to_numeric(amounts[predictions == "Suspicious"], errors="coerce").dropna()

    bins = np.logspace(0, 4, 35)

    if not norm_amounts.empty:
        ax.hist(norm_amounts, bins=bins, alpha=0.55, color=cfg["blue"], label="Normal Transactions")
    if not susp_amounts.empty:
        ax.hist(susp_amounts, bins=bins, alpha=0.85, color=cfg["red"], label="Suspicious Transactions")

    ax.set_xscale("log")
    ax.set_xlabel("Transaction Amount in USD (Logarithmic Scale)", fontsize=9.2)
    ax.set_ylabel("Frequency", fontsize=9.2)
    ax.set_title("Transaction Amount Distribution by Prediction", fontsize=11, fontweight=600, pad=12)
    _style_chart_legend(ax, cfg)
    fig.tight_layout()
    return fig


def plot_anomaly_vs_amount(
    amounts: pd.Series,
    scores: np.ndarray,
    predictions: pd.Series,
    threshold: float,
    sample_size: int = 5000,
    theme: str = "light",
) -> plt.Figure:
    """Plot scatter plot of Anomaly Score vs Transaction Amount."""
    cfg = get_chart_theme_config(theme)
    df_plot = pd.DataFrame({
        "Amount": pd.to_numeric(amounts, errors="coerce"),
        "Score": scores,
        "Prediction": predictions,
    }).dropna()

    if len(df_plot) > sample_size:
        susp_df = df_plot[df_plot["Prediction"] == "Suspicious"]
        norm_df = df_plot[df_plot["Prediction"] == "Normal"]
        norm_sample = norm_df.sample(min(len(norm_df), sample_size - len(susp_df)), random_state=42)
        df_plot = pd.concat([norm_sample, susp_df])

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=120, facecolor=cfg["fig_facecolor"])
    _style_chart_axes(ax, cfg)

    norm_sub = df_plot[df_plot["Prediction"] == "Normal"]
    susp_sub = df_plot[df_plot["Prediction"] == "Suspicious"]

    ax.scatter(
        norm_sub["Amount"],
        norm_sub["Score"],
        alpha=0.35,
        s=12,
        color=cfg["blue"],
        label="Normal",
        edgecolors="none",
    )
    ax.scatter(
        susp_sub["Amount"],
        susp_sub["Score"],
        alpha=0.85,
        s=28,
        color=cfg["red"],
        label="Suspicious",
        edgecolors="#7f1d1d" if theme == "light" else "#fca5a5",
        linewidths=0.5,
    )

    ax.axhline(
        threshold,
        color=cfg["amber"],
        linestyle="--",
        linewidth=1.8,
        label=f"Threshold ({threshold:.4f})",
    )

    ax.set_xlabel("Transaction Amount ($)", fontsize=9.2)
    ax.set_ylabel("Anomaly Score", fontsize=9.2)
    ax.set_title("Anomaly Score vs Transaction Amount", fontsize=11, fontweight=600, pad=12)
    _style_chart_legend(ax, cfg)
    fig.tight_layout()
    return fig


def plot_hourly_trend(times: pd.Series, predictions: pd.Series, theme: str = "light") -> plt.Figure:
    """Plot hourly timeline trend of suspicious anomalies."""
    cfg = get_chart_theme_config(theme)
    valid_times = pd.to_numeric(times, errors="coerce").dropna()
    hour_series = (valid_times // 3600).astype(int)

    df_trend = pd.DataFrame({"Hour": hour_series, "Suspicious": (predictions == "Suspicious").astype(int)})
    grouped = df_trend.groupby("Hour")["Suspicious"].sum()

    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=120, facecolor=cfg["fig_facecolor"])
    _style_chart_axes(ax, cfg)

    ax.plot(
        grouped.index,
        grouped.values,
        marker="o",
        markersize=4,
        linewidth=2.0,
        color=cfg["blue"],
        label="Suspicious Anomalies Detected",
    )
    ax.fill_between(grouped.index, 0, grouped.values, color=cfg["blue"], alpha=0.18)

    ax.set_xlabel("Elapsed Time (Hours from Dataset Start)", fontsize=9.2)
    ax.set_ylabel("Suspicious Records Count", fontsize=9.2)
    ax.set_title("Suspicious Transaction Frequency Across Time (48-Hour Dataset)", fontsize=11, fontweight=600, pad=12)
    _style_chart_legend(ax, cfg)
    fig.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(cm: np.ndarray, theme: str = "light") -> plt.Figure:
    """Render a clean confusion matrix heatmap with guaranteed high-contrast text."""
    cfg = get_chart_theme_config(theme)
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=120, facecolor=cfg["fig_facecolor"])
    _style_chart_axes(ax, cfg)

    cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    ax.set_title("Test Confusion Matrix (56,746 Held-Out Records)", fontsize=11, fontweight=600, pad=14)
    ax.set_xlabel("Predicted Class", fontsize=9.5, fontweight=600, labelpad=8)
    ax.set_ylabel("Actual Class", fontsize=9.5, fontweight=600, labelpad=8)
    ax.set_xticks([0, 1], ["Normal (0)", "Fraud (1)"])
    ax.set_yticks([0, 1], ["Normal (0)", "Fraud (1)"])

    labels = [
        [f"TN\n{cm[0, 0]:,}", f"FP\n{cm[0, 1]:,}"],
        [f"FN\n{cm[1, 0]:,}", f"TP\n{cm[1, 1]:,}"],
    ]

    for i in range(2):
        for j in range(2):
            text_color = "#ffffff" if cm[i, j] > (cm.max() / 2) else "#0f2b48"
            ax.text(
                j,
                i,
                labels[i][j],
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight=700,
                color=text_color,
            )

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=cfg["border_color"], labelsize=8)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=cfg["text_muted"])
    fig.tight_layout()
    return fig


def plot_roc_curve_chart(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, theme: str = "light") -> plt.Figure:
    """Render the ROC Curve with AUC score."""
    cfg = get_chart_theme_config(theme)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=120, facecolor=cfg["fig_facecolor"])
    _style_chart_axes(ax, cfg)

    ax.plot(
        fpr,
        tpr,
        color=cfg["blue"],
        linewidth=2.4,
        label=f"Isolation Forest (ROC-AUC = {roc_auc * 100:.2f}%)",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color=cfg["text_muted"], linewidth=1.2, label="Random Guess (AUC = 50.0%)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=9.2)
    ax.set_ylabel("True Positive Rate / Recall (TPR)", fontsize=9.2)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=11, fontweight=600, pad=12)
    _style_chart_legend(ax, cfg, loc="lower right")
    fig.tight_layout()
    return fig


def plot_precision_recall_chart(recall: np.ndarray, precision: np.ndarray, pr_auc: float, theme: str = "light") -> plt.Figure:
    """Render the Precision-Recall curve."""
    cfg = get_chart_theme_config(theme)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=120, facecolor=cfg["fig_facecolor"])
    _style_chart_axes(ax, cfg)

    ax.plot(
        recall,
        precision,
        color=cfg["blue"],
        linewidth=2.4,
        label=f"PR Curve (PR-AUC = {pr_auc:.4f})",
    )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (Fraud Detected)", fontsize=9.2)
    ax.set_ylabel("Precision (Correct Suspicious Calls)", fontsize=9.2)
    ax.set_title("Precision-Recall Curve (Highly Imbalanced Data)", fontsize=11, fontweight=600, pad=12)
    _style_chart_legend(ax, cfg, loc="upper right")
    fig.tight_layout()
    return fig


def plot_scores_by_actual_label(scores: np.ndarray, y_true: np.ndarray, threshold: float, theme: str = "light") -> plt.Figure:
    """Plot anomaly score separation between actual normal and actual fraud records."""
    cfg = get_chart_theme_config(theme)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=120, facecolor=cfg["fig_facecolor"])
    _style_chart_axes(ax, cfg)

    norm_scores = scores[y_true == 0]
    fraud_scores = scores[y_true == 1]

    if len(norm_scores) > 30000:
        norm_scores = np.random.choice(norm_scores, size=30000, replace=False)

    ax.hist(
        norm_scores,
        bins=50,
        alpha=0.55,
        color=cfg["blue"],
        label=f"Actual Normal ({len(norm_scores):,} sample)",
        density=True,
    )
    ax.hist(
        fraud_scores,
        bins=25,
        alpha=0.85,
        color=cfg["red"],
        label=f"Actual Fraud ({len(fraud_scores)} total)",
        density=True,
    )

    ax.axvline(
        threshold,
        color=cfg["amber"],
        linestyle="--",
        linewidth=2.0,
        label=f"Calibrated Threshold ({threshold:.4f})",
    )

    ax.set_xlabel("Anomaly Score (-decision_function)", fontsize=9.2)
    ax.set_ylabel("Probability Density", fontsize=9.2)
    ax.set_title("Anomaly Score Density by Ground Truth Label", fontsize=11, fontweight=600, pad=12)
    _style_chart_legend(ax, cfg)
    fig.tight_layout()
    return fig


def plot_simulation_timeline(sim_df: pd.DataFrame, threshold: float, theme: str = "light") -> plt.Figure:
    """Plot anomaly score stream of generated simulation transactions."""
    cfg = get_chart_theme_config(theme)
    fig, ax = plt.subplots(figsize=(9.5, 3.5), dpi=120, facecolor=cfg["fig_facecolor"])
    _style_chart_axes(ax, cfg)

    if sim_df.empty:
        ax.text(0.5, 0.5, "No simulated transactions yet", ha="center", va="center", color=cfg["text_muted"])
        return fig

    indices = np.arange(len(sim_df))
    scores = sim_df["Anomaly_Score"].values
    preds = sim_df["Prediction"].values

    norm_mask = preds == "Normal"
    susp_mask = preds == "Suspicious"

    if np.any(norm_mask):
        ax.scatter(
            indices[norm_mask],
            scores[norm_mask],
            color=cfg["blue"],
            alpha=0.6,
            s=22,
            label="Normal Simulated",
        )
    if np.any(susp_mask):
        ax.scatter(
            indices[susp_mask],
            scores[susp_mask],
            color=cfg["red"],
            alpha=0.9,
            s=45,
            edgecolors="#7f1d1d" if theme == "light" else "#fca5a5",
            linewidths=1.0,
            label="Suspicious Alert",
        )

    ax.plot(indices, scores, color=cfg["text_muted"], alpha=0.3, linewidth=1.0)
    ax.axhline(
        threshold,
        color=cfg["amber"],
        linestyle="--",
        linewidth=1.8,
        label=f"Threshold ({threshold:.4f})",
    )

    ax.set_xlabel("Simulated Transaction Sequence #", fontsize=9.2)
    ax.set_ylabel("Anomaly Score", fontsize=9.2)
    ax.set_title("Live Simulation Anomaly Score Stream", fontsize=11, fontweight=600, pad=12)
    _style_chart_legend(ax, cfg, loc="upper left")
    fig.tight_layout()
    return fig


def plot_score_gauge(score: float, threshold: float, theme: str = "light") -> plt.Figure:
    """Plot a theme-aware horizontal gauge showing anomaly score relative to the threshold."""
    cfg = get_chart_theme_config(theme)
    is_dark = (theme == "dark")
    fig, ax = plt.subplots(figsize=(8, 1.6), dpi=120, facecolor=cfg["fig_facecolor"])
    _style_chart_axes(ax, cfg)

    bar_y = 0.5
    bar_height = 0.35
    max_score = max(threshold * 2.5, score * 1.3, 0.3)

    # Theme-adapted zone colors
    normal_color = "#064e3b" if is_dark else "#d1fae5"
    normal_edge = "#059669" if is_dark else "#6ee7b7"
    susp_color = "#450a0a" if is_dark else "#fee2e2"
    susp_edge = "#dc2626" if is_dark else "#fca5a5"

    # Normal zone
    ax.barh(bar_y, threshold, height=bar_height, color=normal_color, edgecolor=normal_edge, linewidth=0.8)
    # Suspicious zone
    ax.barh(bar_y, max_score - threshold, left=threshold, height=bar_height, color=susp_color, edgecolor=susp_edge, linewidth=0.8)

    # Threshold line
    ax.axvline(threshold, color=cfg["amber"], linewidth=2.5, linestyle="--", zorder=5)
    ax.text(
        threshold, bar_y + bar_height / 2 + 0.12, f"Threshold\n{threshold:.4f}",
        ha="center", va="bottom", fontsize=7.5, fontweight=600, color=cfg["amber"]
    )

    # Score marker
    marker_color = cfg["red"] if score >= threshold else cfg["green"]
    ax.plot(score, bar_y, marker="v", markersize=14, color=marker_color, zorder=10)
    ax.text(
        score, bar_y - bar_height / 2 - 0.08, f"Score: {score:.4f}",
        ha="center", va="top", fontsize=8.5, fontweight=700, color=marker_color
    )

    # Zone labels
    ax.text(threshold / 2, bar_y, "NORMAL", ha="center", va="center",
            fontsize=8, fontweight=700, color=cfg["green"], alpha=0.9)
    mid_susp = threshold + (max_score - threshold) / 2
    ax.text(mid_susp, bar_y, "SUSPICIOUS", ha="center", va="center",
            fontsize=8, fontweight=700, color=cfg["red"], alpha=0.9)

    ax.set_xlim(-0.02, max_score)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Anomaly Score Scale", fontsize=8.5)
    ax.set_title("Score Position Relative to Decision Threshold", fontsize=10, fontweight=600, pad=10)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_pca_feature_bars(
    pca_features: list,
    pca_values: list,
    txn_id: str,
    theme: str = "light",
) -> plt.Figure:
    """Plot PCA feature breakdown bar chart with full theme adaptation."""
    cfg = get_chart_theme_config(theme)
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=120, facecolor=cfg["fig_facecolor"])
    _style_chart_axes(ax, cfg)

    colors = [cfg["red"] if abs(v) > 3.0 else cfg["blue"] for v in pca_values]
    ax.bar(pca_features, pca_values, color=colors, width=0.6)
    ax.axhline(0, color=cfg["text_muted"], linewidth=0.8)
    ax.set_ylabel("Standardized Value", fontsize=8.5)
    ax.set_title(
        f"Component Breakdown for {txn_id} (Red = |Value| > 3.0)",
        fontsize=10,
        fontweight=600,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=7.5)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    return fig
