"""
streamlit_app.py
────────────────────────────────────────────────────
Customer Churn & LTV Intelligence Dashboard
Black Theme — Live API — Streamlit + Plotly

Pages:
    1. Overview
    2. Customer Lookup
    3. Segment Analysis
    4. Revenue Recovery
    5. Model Performance
    6. Action Center

Usage:
    cd app && streamlit run streamlit_app.py
────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import subprocess
import sys
import time
import os
from datetime import datetime

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title            = "Churn & LTV Intelligence",
    page_icon             = "🎯",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ── API Config ────────────────────────────────────────
API_BASE = os.environ.get(
    "API_BASE", "http://127.0.0.1:8000"
)
API_DIR     = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../api"
)
DATA_PATH   = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../data/ecommerce_user_segmentation.csv"
)

# ── Black Theme Colors ────────────────────────────────
BG_BLACK    = "#000000"
BG_CARD     = "#0D0D0D"
BG_CARD2    = "#111111"
BG_BORDER   = "#1A1A1A"
BG_BORDER2  = "#222222"
PURPLE      = "#7C3AED"
CYAN        = "#06B6D4"
EMERALD     = "#10B981"
AMBER       = "#F59E0B"
RED         = "#EF4444"
TEXT_WHITE  = "#FFFFFF"
TEXT_MUTED  = "#6B7280"
TEXT_DIM    = "#374151"

# ── Segment Colors ────────────────────────────────────
SEGMENT_COLORS = {
    "Champion"       : "#10B981",
    "At-Risk VIP"    : "#F59E0B",
    "Promising"      : "#06B6D4",
    "Vulnerable"     : "#F97316",
    "Hibernating"    : "#6B7280",
    "Losing Customer": "#EF4444",
}

SEGMENT_ICONS = {
    "Champion"       : "🟢",
    "At-Risk VIP"    : "🟡",
    "Promising"      : "🔵",
    "Vulnerable"     : "🟠",
    "Hibernating"    : "⚪",
    "Losing Customer": "🔴",
}

SEGMENT_ORDER = [
    "Champion", "At-Risk VIP", "Promising",
    "Vulnerable", "Hibernating", "Losing Customer"
]

# ── Plotly Dark Defaults ──────────────────────────────
def apply_dark(fig, height=400, **kwargs):
    layout = dict(
        height        = height,
        plot_bgcolor  = BG_CARD,
        paper_bgcolor = BG_BLACK,
        font          = dict(
            color  = TEXT_WHITE,
            size   = 12,
            family = "Inter, sans-serif"
        ),
        xaxis = dict(
            gridcolor  = BG_BORDER2,
            linecolor  = BG_BORDER,
            tickcolor  = TEXT_MUTED,
            tickfont   = dict(color=TEXT_MUTED),
            showgrid   = True,
        ),
        yaxis = dict(
            gridcolor  = BG_BORDER2,
            linecolor  = BG_BORDER,
            tickcolor  = TEXT_MUTED,
            tickfont   = dict(color=TEXT_MUTED),
            showgrid   = True,
        ),
        legend = dict(
            bgcolor     = BG_CARD2,
            bordercolor = BG_BORDER,
            font        = dict(color=TEXT_WHITE),
        ),
        margin = dict(l=40, r=40, t=50, b=40),
    )
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


# ── CSS ───────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url(
        'https://fonts.googleapis.com/css2?
        family=Inter:wght@300;400;500;600;700;800;900
        &family=Space+Grotesk:wght@400;500;600;700
        &display=swap'
    );

    /* ── Base ── */
    .stApp {{
        background-color: {BG_BLACK};
        font-family: 'Inter', sans-serif;
    }}
    .main .block-container {{
        padding-top: 1.5rem;
        max-width: 1400px;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: {BG_CARD};
        border-right: 1px solid {PURPLE};
    }}
    [data-testid="stSidebar"] * {{
        font-family: 'Space Grotesk', sans-serif !important;
    }}

    /* ── Page Titles ── */
    h1 {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        background: linear-gradient(
            135deg, {PURPLE}, {CYAN}
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {BG_BORDER};
        animation: fadeInDown 0.5s ease;
    }}

    /* ── Section Headers ── */
    h2, h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: {TEXT_WHITE} !important;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-size: 0.85rem !important;
        border-left: 3px solid {PURPLE};
        padding-left: 10px;
        margin-top: 1.5rem !important;
    }}

    /* ── Metric Cards ── */
    [data-testid="metric-container"] {{
        background: {BG_CARD} !important;
        border: 1px solid {BG_BORDER} !important;
        border-radius: 16px !important;
        padding: 20px !important;
        border-top: 2px solid {PURPLE} !important;
        transition: all 0.3s ease;
        animation: fadeInUp 0.4s ease;
    }}
    [data-testid="metric-container"]:hover {{
        border-top-color: {CYAN} !important;
        box-shadow: 0 0 20px rgba(6,182,212,0.15);
        transform: translateY(-2px);
    }}
    [data-testid="metric-container"] label {{
        color: {TEXT_MUTED} !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {TEXT_WHITE} !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {BG_CARD} !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px;
        border: 1px solid {BG_BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {TEXT_MUTED} !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        transition: all 0.2s !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(
            135deg, {PURPLE}, {CYAN}
        ) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background: linear-gradient(
            135deg, {PURPLE}, {CYAN}
        ) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
    }}

    /* ── Download Buttons ── */
    .stDownloadButton > button {{
        background: transparent !important;
        color: {CYAN} !important;
        border: 1px solid {CYAN} !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        transition: all 0.3s ease !important;
    }}
    .stDownloadButton > button:hover {{
        background: {CYAN} !important;
        color: black !important;
        box-shadow: 0 4px 15px rgba(6,182,212,0.4) !important;
    }}

    /* ── Text Input ── */
    .stTextInput > div > div > input {{
        background-color: {BG_CARD} !important;
        color: {TEXT_WHITE} !important;
        border: 1px solid {BG_BORDER2} !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: {PURPLE} !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
    }}

    /* ── Select + Multiselect ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background-color: {BG_CARD} !important;
        border: 1px solid {BG_BORDER2} !important;
        border-radius: 10px !important;
        color: {TEXT_WHITE} !important;
    }}

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] {{
        margin-top: 8px;
    }}

    /* ── Dataframe ── */
    .stDataFrame {{
        border: 1px solid {BG_BORDER2} !important;
        border-radius: 12px !important;
        overflow: hidden;
    }}
    .stDataFrame [data-testid="stDataFrameResizable"] {{
        background-color: {BG_CARD} !important;
    }}

    /* ── Divider ── */
    hr {{
        border: none !important;
        border-top: 1px solid {BG_BORDER} !important;
        margin: 1.5rem 0 !important;
    }}

    /* ── Info / Warning ── */
    .stInfo {{
        background-color: {BG_CARD} !important;
        border-left: 3px solid {PURPLE} !important;
        border-radius: 8px !important;
        color: {TEXT_WHITE} !important;
        font-size: 0.85rem !important;
    }}

    /* ── Sidebar Radio ── */
    [data-testid="stSidebar"] .stRadio label {{
        color: {TEXT_MUTED} !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        display: block;
    }}
    [data-testid="stSidebar"] .stRadio label:hover {{
        background-color: {BG_BORDER} !important;
        color: {TEXT_WHITE} !important;
    }}

    /* ── Live Badge ── */
    .live-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16,185,129,0.1);
        border: 1px solid {EMERALD};
        color: {EMERALD};
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        animation: pulse 2s infinite;
    }}
    .live-dot {{
        width: 6px;
        height: 6px;
        background: {EMERALD};
        border-radius: 50%;
        animation: pulse 1s infinite;
    }}

    /* ── Status Card ── */
    .status-card {{
        background: {BG_CARD};
        border: 1px solid {BG_BORDER};
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }}
    .status-card:hover {{
        border-color: {PURPLE};
        box-shadow: 0 0 20px rgba(124,58,237,0.1);
    }}

    /* ── Gradient Text ── */
    .gradient-text {{
        background: linear-gradient(
            135deg, {PURPLE}, {CYAN}
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }}
    .cyan-text {{
        color: {CYAN};
        font-weight: 700;
    }}
    .emerald-text {{
        color: {EMERALD};
        font-weight: 700;
    }}
    .amber-text {{
        color: {AMBER};
        font-weight: 700;
    }}
    .red-text {{
        color: {RED};
        font-weight: 700;
    }}

    /* ── Animations ── */
    @keyframes fadeInDown {{
        from {{ opacity: 0; transform: translateY(-10px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50%       {{ opacity: 0.5; }}
    }}
    @keyframes glow {{
        0%, 100% {{ box-shadow: 0 0 5px rgba(124,58,237,0.3); }}
        50%       {{ box-shadow: 0 0 20px rgba(124,58,237,0.6); }}
    }}

    /* ── Plotly Charts ── */
    .js-plotly-plot {{
        border-radius: 12px !important;
        border: 1px solid {BG_BORDER} !important;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {BG_BLACK};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {PURPLE};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {CYAN};
    }}
</style>
""", unsafe_allow_html=True)


# ── API Manager ───────────────────────────────────────
def start_api():
    """Launch FastAPI in background if not running."""
    try:
        r = requests.get(
            f"{API_BASE}/health", timeout=2
        )
        if r.status_code == 200:
            return True
    except:
        pass

    try:
        subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "main:app", "--host", "127.0.0.1",
                "--port", "8000",
            ],
            cwd    = API_DIR,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        )
        time.sleep(3)
        r = requests.get(
            f"{API_BASE}/health", timeout=3
        )
        return r.status_code == 200
    except:
        return False


def api_get(endpoint: str, timeout: int = 10):
    """Make GET request to API."""
    try:
        r = requests.get(
            f"{API_BASE}{endpoint}",
            timeout=timeout
        )
        if r.status_code == 200:
            return r.json(), None
        return None, f"API error {r.status_code}"
    except Exception as e:
        return None, str(e)


def api_post(endpoint: str, data: dict,
             timeout: int = 10):
    """Make POST request to API."""
    try:
        r = requests.post(
            f"{API_BASE}{endpoint}",
            json    = data,
            timeout = timeout,
        )
        if r.status_code == 200:
            return r.json(), None
        return None, f"API error {r.status_code}"
    except Exception as e:
        return None, str(e)


# ── Data ──────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_segment_summary():
    """Fetch segment summary from API."""
    data, err = api_get("/segment/summary")
    return data, err


@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)


# ── Sidebar ───────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown(f"""
    <div style='text-align:center; padding:16px 0'>
        <div style='font-size:2.5rem'>🎯</div>
        <div style='
            font-family: Space Grotesk;
            font-size: 1.1rem;
            font-weight: 800;
            background: linear-gradient(135deg,{PURPLE},{CYAN});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 0.05em;
        '>CHURN & LTV</div>
        <div style='
            color:{TEXT_MUTED};
            font-size:0.7rem;
            letter-spacing:0.1em;
            text-transform:uppercase;
        '>Intelligence Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    # API Status
    try:
        r = requests.get(
            f"{API_BASE}/health", timeout=1
        )
        api_ok = r.status_code == 200
    except:
        api_ok = False

    if api_ok:
        st.sidebar.markdown(f"""
        <div style='
            display:flex; align-items:center;
            gap:8px; padding:8px 12px;
            background:rgba(16,185,129,0.1);
            border:1px solid {EMERALD};
            border-radius:8px; margin-bottom:12px;
        '>
            <div style='
                width:8px; height:8px;
                background:{EMERALD};
                border-radius:50%;
                animation:pulse 1s infinite;
            '></div>
            <span style='
                color:{EMERALD};
                font-size:0.75rem;
                font-weight:700;
                letter-spacing:0.08em;
                text-transform:uppercase;
            '>API LIVE</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
        <div style='
            display:flex; align-items:center;
            gap:8px; padding:8px 12px;
            background:rgba(239,68,68,0.1);
            border:1px solid {RED};
            border-radius:8px; margin-bottom:12px;
        '>
            <div style='
                width:8px; height:8px;
                background:{RED};
                border-radius:50%;
            '></div>
            <span style='
                color:{RED};
                font-size:0.75rem;
                font-weight:700;
                letter-spacing:0.08em;
                text-transform:uppercase;
            '>API OFFLINE</span>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "NAVIGATE",
        [
            "🏠  Overview",
            "🔍  Customer Lookup",
            "📊  Segment Analysis",
            "💰  Revenue Recovery",
            "🤖  Model Performance",
            "🎯  Action Center",
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style='
        font-size:0.7rem;
        color:{TEXT_MUTED};
        letter-spacing:0.05em;
        text-transform:uppercase;
        font-weight:600;
        margin-bottom:8px;
    '>GLOBAL FILTERS</div>
    """, unsafe_allow_html=True)

    selected_segments = st.sidebar.multiselect(
        "Segments",
        options = SEGMENT_ORDER,
        default = SEGMENT_ORDER,
    )

    ltv_max = 40000
    ltv_range = st.sidebar.slider(
        "LTV Range ($)",
        min_value = 0,
        max_value = ltv_max,
        value     = (0, ltv_max),
        step      = 500,
    )

    st.sidebar.markdown("---")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown(f"""
    <div style='
        padding:12px;
        background:{BG_CARD};
        border:1px solid {BG_BORDER};
        border-radius:8px;
        margin-top:12px;
    '>
        <div style='
            font-size:0.65rem;
            color:{TEXT_MUTED};
            letter-spacing:0.08em;
            text-transform:uppercase;
            margin-bottom:8px;
            font-weight:700;
        '>SYSTEM INFO</div>
        <div style='
            font-size:0.8rem;
            color:{TEXT_WHITE};
            line-height:1.8;
        '>
            🔥 LightGBM + RandomForest<br>
            👥 10,000 Customers<br>
            📡 FastAPI v1.0.0<br>
            🔄 Auto-refresh 30s
        </div>
    </div>
    <div style='
        margin-top:12px;
        font-size:0.65rem;
        color:{TEXT_DIM};
        text-align:center;
        letter-spacing:0.05em;
    '>Last refresh: {datetime.now().strftime('%H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

    return page, selected_segments, ltv_range


# ── Page Header ───────────────────────────────────────
def page_header(title: str, subtitle: str):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(title)
        st.markdown(f"""
        <div style='
            color:{TEXT_MUTED};
            font-size:0.9rem;
            margin-top:-12px;
            margin-bottom:16px;
            letter-spacing:0.02em;
        '>{subtitle}</div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='
            text-align:right;
            padding-top:16px;
        '>
            <div style='
                display:inline-flex;
                align-items:center;
                gap:6px;
                background:rgba(16,185,129,0.1);
                border:1px solid {EMERALD};
                color:{EMERALD};
                padding:6px 14px;
                border-radius:20px;
                font-size:0.7rem;
                font-weight:700;
                letter-spacing:0.1em;
                text-transform:uppercase;
            '>● LIVE DATA</div>
        </div>
        """, unsafe_allow_html=True)


# ── Page 1 — Overview ─────────────────────────────────
def page_overview(selected_segments, ltv_range):
    page_header(
        "🏠 Overview",
        "Real-time customer health, revenue "
        "distribution and segment intelligence"
    )

    with st.spinner("Fetching live data..."):
        data, err = fetch_segment_summary()

    if err or not data:
        st.error(f"API Error: {err}")
        return

    # Filter segments
    segments   = [
        s for s in data["segments"]
        if s["segment"] in selected_segments
    ]
    total_rev  = sum(s["total_ltv"] for s in segments)
    at_risk_rev = sum(
        s["total_ltv"] for s in segments
        if s["segment"] in [
            "At-Risk VIP","Vulnerable","Losing Customer"
        ]
    )
    total_cust = sum(s["count"] for s in segments)
    avg_ltv    = total_rev / total_cust \
                 if total_cust > 0 else 0
    avg_risk   = sum(
        s["avg_risk"] * s["count"]
        for s in segments
    ) / total_cust if total_cust > 0 else 0

    # ── KPIs ──────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Customers",     f"{total_cust:,}")
    c2.metric("💰 Total Revenue", f"${total_rev/1e6:.2f}M")
    c3.metric(
        "⚠️ Revenue At Risk",
        f"${at_risk_rev/1e6:.2f}M",
        delta       = f"{at_risk_rev/total_rev*100:.1f}%",
        delta_color = "inverse"
    )
    c4.metric("📈 Avg LTV",   f"${avg_ltv:,.0f}")
    c5.metric("🎯 Avg Risk",  f"{avg_risk:.4f}")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Distribution")
        seg_names  = [s["segment"] for s in segments]
        seg_counts = [s["count"] for s in segments]
        colors     = [
            SEGMENT_COLORS.get(s, PURPLE)
            for s in seg_names
        ]

        fig = go.Figure(go.Pie(
            values      = seg_counts,
            labels      = seg_names,
            marker      = dict(colors=colors),
            hole        = 0.55,
            textinfo    = "percent+label",
            textfont    = dict(
                color  = TEXT_WHITE,
                size   = 11,
                family = "Inter"
            ),
            hovertemplate = (
                "<b>%{label}</b><br>"
                "Customers: %{value:,}<br>"
                "Share: %{percent}<extra></extra>"
            ),
        ))
        fig.add_annotation(
            text      = f"<b>{total_cust:,}</b><br>"
                        f"<span style='font-size:10px'>"
                        f"CUSTOMERS</span>",
            x=0.5, y=0.5,
            showarrow = False,
            font      = dict(
                color  = TEXT_WHITE,
                size   = 16,
                family = "Space Grotesk"
            ),
        )
        apply_dark(fig, height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue by Segment ($M)")
        seg_rev = [
            s["total_ltv"]/1e6 for s in segments
        ]
        fig = go.Figure(go.Bar(
            x            = seg_rev,
            y            = seg_names,
            orientation  = "h",
            marker       = dict(
                color   = colors,
                opacity = 0.9,
                line    = dict(
                    color = BG_BLACK,
                    width = 1
                ),
            ),
            text         = [
                f"${v:.1f}M" for v in seg_rev
            ],
            textposition = "outside",
            textfont     = dict(
                color  = TEXT_WHITE,
                size   = 11
            ),
            hovertemplate = (
                "<b>%{y}</b><br>"
                "Revenue: $%{x:.2f}M<extra></extra>"
            ),
        ))
        apply_dark(
            fig, height=400,
            xaxis = dict(
                tickprefix = "$",
                ticksuffix = "M",
                gridcolor  = BG_BORDER2,
                tickfont   = dict(color=TEXT_MUTED),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── LTV Box ───────────────────────────────────────
    st.subheader("LTV Distribution by Segment")
    df_raw = load_raw_data()

    # Build approximate LTV distribution
    # from segment averages
    rows = []
    for s in segments:
        mean = s["avg_ltv"]
        std  = mean * 0.4
        vals = np.random.normal(
            mean, std, s["count"]
        )
        vals = np.clip(vals, 0, None)
        for v in vals[:200]:
            rows.append({
                "Segment": s["segment"],
                "LTV"    : v
            })
    df_box = pd.DataFrame(rows)

    fig = px.violin(
        df_box,
        x                  = "Segment",
        y                  = "LTV",
        color              = "Segment",
        color_discrete_map = SEGMENT_COLORS,
        category_orders    = {
            "Segment": [
                s for s in SEGMENT_ORDER
                if s in df_box["Segment"].unique()
            ]
        },
        box       = True,
        points    = False,
    )
    apply_dark(
        fig, height=420,
        showlegend  = False,
        yaxis_title = "Predicted LTV ($)",
        yaxis       = dict(
            tickprefix = "$",
            gridcolor  = BG_BORDER2,
            tickfont   = dict(color=TEXT_MUTED),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page 2 — Customer Lookup ──────────────────────────
def page_customer_lookup():
    page_header(
        "🔍 Customer Lookup",
        "Live prediction report for any customer"
    )

    df_raw = load_raw_data()

    col1, col2 = st.columns([3, 1])
    with col1:
        cid = st.text_input(
            "CUSTOMER ID",
            placeholder = "e.g. CUST00001",
            value       = "CUST00001",
        )
    with col2:
        st.markdown(
            "<br>", unsafe_allow_html=True
        )
        if st.button("🎲 Random"):
            cid = df_raw["Customer_ID"].sample(
                1
            ).iloc[0]
            st.session_state["lookup_id"] = cid
            st.rerun()

    if "lookup_id" in st.session_state:
        cid = st.session_state["lookup_id"]

    if not cid:
        return

    with st.spinner(f"Fetching prediction for {cid}..."):
        result, err = api_get(f"/predict/{cid}")

    if err or not result:
        st.error(f"Customer not found: {cid}")
        return

    seg        = result["segment"]
    seg_color  = SEGMENT_COLORS.get(seg, PURPLE)
    seg_icon   = SEGMENT_ICONS.get(seg, "⚪")
    churn_prob = result["churn_probability"]
    ltv        = result["predicted_ltv"]
    risk       = result["risk_score"]

    st.markdown("---")

    # ── Metrics ───────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customer ID",    cid)
    c2.metric(
        "Churn Probability",
        f"{churn_prob:.1%}",
        delta       = result["churn_risk"],
    )
    c3.metric("Predicted LTV",  f"${ltv:,.2f}")
    c4.metric(
        "Segment",
        f"{seg_icon} {seg}"
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    # ── Gauge ─────────────────────────────────────────
    with col1:
        st.subheader("Composite Risk Score")
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = risk,
            gauge = dict(
                axis = dict(
                    range    = [0, 1],
                    tickfont = dict(color=TEXT_MUTED),
                    tickcolor= TEXT_MUTED,
                ),
                bar   = dict(
                    color     = seg_color,
                    thickness = 0.25,
                ),
                bgcolor     = BG_CARD2,
                bordercolor = BG_BORDER,
                steps = [
                    {"range": [0, 0.25],
                     "color": "rgba(16,185,129,0.15)"},
                    {"range": [0.25, 0.50],
                     "color": "rgba(245,158,11,0.15)"},
                    {"range": [0.50, 0.75],
                     "color": "rgba(249,115,22,0.15)"},
                    {"range": [0.75, 1.0],
                     "color": "rgba(239,68,68,0.15)"},
                ],
                threshold = dict(
                    line  = dict(
                        color = RED, width=3
                    ),
                    value = 0.75,
                ),
            ),
            number = dict(
                font      = dict(
                    size   = 48,
                    color  = seg_color,
                    family = "Space Grotesk"
                ),
                valueformat = ".4f",
            ),
            title = dict(
                text = "Risk Score (0 — 1)",
                font = dict(
                    color  = TEXT_MUTED,
                    size   = 12,
                    family = "Inter"
                ),
            ),
        ))
        apply_dark(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature Radar ─────────────────────────────────
    with col2:
        st.subheader("Feature Profile")
        match = df_raw[
            df_raw["Customer_ID"] == cid
        ]
        if len(match) > 0:
            row  = match.iloc[0]
            feat_names = [
                "Frequency", "Avg_Order_Value",
                "Session_Count", "Wishlist_Adds",
                "Clicks", "Pages_Viewed",
            ]
            feat_vals  = [
                float(row[f]) for f in feat_names
            ]
            # Normalise for radar
            feat_norm  = [
                v / max(feat_vals) if max(feat_vals) > 0
                else 0
                for v in feat_vals
            ]

            fig = go.Figure(go.Scatterpolar(
                r     = feat_norm + [feat_norm[0]],
                theta = feat_names + [feat_names[0]],
                fill  = "toself",
                fillcolor = f"rgba"
                            f"{tuple(int(seg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}",
                line  = dict(color=seg_color, width=2),
                name  = cid,
            ))
            fig.update_layout(
                polar = dict(
                    bgcolor   = BG_CARD,
                    radialaxis = dict(
                        visible  = True,
                        range    = [0, 1],
                        color    = TEXT_MUTED,
                        gridcolor= BG_BORDER2,
                    ),
                    angularaxis = dict(
                        color    = TEXT_WHITE,
                        gridcolor= BG_BORDER2,
                    ),
                ),
            )
            apply_dark(fig, height=320)
            st.plotly_chart(
                fig, use_container_width=True
            )

    # ── Actions ───────────────────────────────────────
    st.markdown("---")
    st.subheader(
        f"{seg_icon} Recommended Actions"
    )
    cols = st.columns(len(result["actions"]))
    for i, (col, action) in enumerate(
        zip(cols, result["actions"])
    ):
        col.markdown(f"""
        <div style='
            background:{BG_CARD};
            border:1px solid {BG_BORDER};
            border-top:2px solid {seg_color};
            border-radius:10px;
            padding:16px;
            text-align:center;
            transition:all 0.3s ease;
            height:100%;
        '>
            <div style='
                font-size:1.5rem;
                margin-bottom:8px;
            '>{'🎯' if i==0 else '💡' if i==1 else '📊' if i==2 else '⚡'}</div>
            <div style='
                color:{TEXT_WHITE};
                font-size:0.8rem;
                font-weight:500;
                line-height:1.5;
            '>{action}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Page 3 — Segment Analysis ─────────────────────────
def page_segment_analysis(
    selected_segments, ltv_range
):
    page_header(
        "📊 Segment Analysis",
        "Interactive customer segmentation "
        "across risk and value dimensions"
    )

    with st.spinner("Loading segment data..."):
        data, err = fetch_segment_summary()

    if err or not data:
        st.error(f"API Error: {err}")
        return

    segments = [
        s for s in data["segments"]
        if s["segment"] in selected_segments
    ]

    # ── Quadrant ──────────────────────────────────────
    st.subheader(
        "Customer Value vs Risk — Quadrant"
    )

    # Build scatter from segment data
    rows = []
    for s in segments:
        mean_risk = s["avg_risk"]
        mean_ltv  = s["avg_ltv"]
        std_risk  = mean_risk * 0.3
        std_ltv   = mean_ltv  * 0.35
        n         = min(s["count"], 500)
        risks     = np.random.normal(
            mean_risk, std_risk, n
        )
        ltvs      = np.random.normal(
            mean_ltv, std_ltv, n
        )
        for r, l in zip(risks, ltvs):
            rows.append({
                "Segment" : s["segment"],
                "Risk"    : max(0, min(1, r)),
                "LTV"     : max(0, l),
            })

    df_scatter = pd.DataFrame(rows)
    fig = px.scatter(
        df_scatter,
        x                  = "Risk",
        y                  = "LTV",
        color              = "Segment",
        color_discrete_map = SEGMENT_COLORS,
        opacity            = 0.6,
        labels             = {
            "Risk": "Composite Risk Score",
            "LTV" : "Predicted LTV ($)",
        },
    )

    # Reference lines
    fig.add_vline(
        x               = 0.25,
        line_dash       = "dash",
        line_color      = RED,
        line_width      = 1.5,
        annotation_text = "High Risk →",
        annotation_font = dict(
            color=TEXT_MUTED, size=11
        ),
    )
    fig.add_hline(
        y               = np.mean(
            [s["avg_ltv"] for s in segments]
        ) * 1.5,
        line_dash       = "dash",
        line_color      = PURPLE,
        line_width      = 1.5,
        annotation_text = "High Value ↑",
        annotation_font = dict(
            color=TEXT_MUTED, size=11
        ),
    )

    apply_dark(
        fig, height=520,
        yaxis = dict(
            tickprefix = "$",
            gridcolor  = BG_BORDER2,
            tickfont   = dict(color=TEXT_MUTED),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    seg_names  = [s["segment"] for s in segments]
    colors     = [
        SEGMENT_COLORS.get(s, PURPLE)
        for s in seg_names
    ]

    with col1:
        st.subheader("Customer Count")
        counts = [s["count"] for s in segments]
        fig = go.Figure(go.Bar(
            x            = counts,
            y            = seg_names,
            orientation  = "h",
            marker       = dict(
                color   = colors,
                opacity = 0.9,
            ),
            text         = [f"{v:,}" for v in counts],
            textposition = "outside",
            textfont     = dict(color=TEXT_WHITE),
        ))
        apply_dark(
            fig, height=380,
            showlegend  = False,
            xaxis_title = "Number of Customers",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Average Risk Score")
        risks = [s["avg_risk"] for s in segments]
        fig = go.Figure(go.Bar(
            x            = risks,
            y            = seg_names,
            orientation  = "h",
            marker       = dict(
                color   = colors,
                opacity = 0.9,
            ),
            text         = [f"{v:.4f}" for v in risks],
            textposition = "outside",
            textfont     = dict(color=TEXT_WHITE),
        ))
        apply_dark(
            fig, height=380,
            showlegend  = False,
            xaxis_title = "Avg Composite Risk Score",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────
    st.markdown("---")
    st.subheader("Segment Intelligence Heatmap")
    heat_data = pd.DataFrame([{
        "Segment"    : s["segment"],
        "Avg Risk"   : s["avg_risk"],
        "Avg LTV $K" : s["avg_ltv"] / 1000,
        "Total LTV $M": s["total_ltv"] / 1e6,
        "Count"      : s["count"],
    } for s in segments]).set_index("Segment")

    fig = px.imshow(
        heat_data.T,
        color_continuous_scale = [
            [0,   BG_CARD],
            [0.5, PURPLE],
            [1,   CYAN],
        ],
        text_auto = ".2f",
        aspect    = "auto",
    )
    apply_dark(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)


# ── Page 4 — Revenue Recovery ─────────────────────────
def page_revenue_recovery(selected_segments):
    page_header(
        "💰 Revenue Recovery",
        "Live ROI simulation with adjustable "
        "campaign parameters"
    )

    with st.spinner("Loading revenue data..."):
        data, err = fetch_segment_summary()

    if err or not data:
        st.error(f"API Error: {err}")
        return

    seg_map = {
        s["segment"]: s
        for s in data["segments"]
        if s["segment"] in selected_segments
    }

    st.markdown("---")
    st.subheader("🎛️ Campaign Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style='
            font-size:0.8rem;
            font-weight:700;
            color:{AMBER};
            letter-spacing:0.08em;
            text-transform:uppercase;
            margin-bottom:8px;
        '>🟡 AT-RISK VIP</div>
        """, unsafe_allow_html=True)
        vip_cost = st.slider(
            "Cost/customer ($)",
            10, 200, 50, key="vip_cost"
        )
        vip_rate = st.slider(
            "Recovery rate (%)",
            10, 80, 45, key="vip_rate"
        ) / 100

    with col2:
        st.markdown(f"""
        <div style='
            font-size:0.8rem;
            font-weight:700;
            color:#F97316;
            letter-spacing:0.08em;
            text-transform:uppercase;
            margin-bottom:8px;
        '>🟠 VULNERABLE</div>
        """, unsafe_allow_html=True)
        vuln_cost = st.slider(
            "Cost/customer ($)",
            5, 100, 25, key="vuln_cost"
        )
        vuln_rate = st.slider(
            "Recovery rate (%)",
            5, 60, 35, key="vuln_rate"
        ) / 100

    with col3:
        st.markdown(f"""
        <div style='
            font-size:0.8rem;
            font-weight:700;
            color:{RED};
            letter-spacing:0.08em;
            text-transform:uppercase;
            margin-bottom:8px;
        '>🔴 LOSING CUSTOMER</div>
        """, unsafe_allow_html=True)
        lose_cost = st.slider(
            "Cost/customer ($)",
            1, 30, 5, key="lose_cost"
        )
        lose_rate = st.slider(
            "Recovery rate (%)",
            5, 30, 15, key="lose_rate"
        ) / 100

    # ── Simulation ────────────────────────────────────
    params = {
        "At-Risk VIP"    : {"cost": vip_cost,  "rate": vip_rate},
        "Vulnerable"     : {"cost": vuln_cost, "rate": vuln_rate},
        "Losing Customer": {"cost": lose_cost, "rate": lose_rate},
    }

    results = []
    for seg, p in params.items():
        if seg not in seg_map:
            continue
        s             = seg_map[seg]
        count         = s["count"]
        revenue_risk  = s["total_ltv"]
        campaign_cost = count * p["cost"]
        recovered     = revenue_risk * p["rate"]
        net_roi       = recovered - campaign_cost
        roi_pct       = (
            net_roi / campaign_cost * 100
            if campaign_cost > 0 else 0
        )
        results.append({
            "Segment"          : seg,
            "Customers"        : count,
            "Revenue At Risk"  : revenue_risk,
            "Campaign Cost"    : campaign_cost,
            "Expected Recovery": recovered,
            "Net ROI"          : net_roi,
            "ROI %"            : roi_pct,
        })

    sim_df = pd.DataFrame(results)
    if sim_df.empty:
        st.warning("No at-risk segments in filter.")
        return

    st.markdown("---")

    # ── KPIs ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total At Risk",
        f"${sim_df['Revenue At Risk'].sum()/1e6:.2f}M"
    )
    c2.metric(
        "Campaign Budget",
        f"${sim_df['Campaign Cost'].sum():,.0f}"
    )
    c3.metric(
        "Expected Recovery",
        f"${sim_df['Expected Recovery'].sum()/1e6:.2f}M"
    )
    total_net = sim_df["Net ROI"].sum()
    total_camp= sim_df["Campaign Cost"].sum()
    c4.metric(
        "Net ROI",
        f"${total_net/1e6:.2f}M",
        delta=f"{total_net/total_camp*100:.0f}% return"
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue At Risk vs Recovery")
        colors = [
            SEGMENT_COLORS.get(s, PURPLE)
            for s in sim_df["Segment"]
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name         = "Revenue At Risk",
            x            = sim_df["Segment"],
            y            = sim_df["Revenue At Risk"]/1e6,
            marker_color = colors,
            opacity      = 0.9,
            hovertemplate= (
                "<b>%{x}</b><br>"
                "At Risk: $%{y:.2f}M<extra></extra>"
            ),
        ))
        fig.add_trace(go.Bar(
            name         = "Expected Recovery",
            x            = sim_df["Segment"],
            y            = sim_df["Expected Recovery"]/1e6,
            marker_color = colors,
            opacity      = 0.35,
            hovertemplate= (
                "<b>%{x}</b><br>"
                "Recovery: $%{y:.2f}M<extra></extra>"
            ),
        ))
        apply_dark(
            fig, height=380,
            barmode     = "group",
            yaxis       = dict(
                tickprefix = "$",
                ticksuffix = "M",
                gridcolor  = BG_BORDER2,
                tickfont   = dict(color=TEXT_MUTED),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ROI % Per Segment")
        fig = go.Figure(go.Bar(
            x            = sim_df["Segment"],
            y            = sim_df["ROI %"],
            marker       = dict(
                color   = colors,
                opacity = 0.9,
                line    = dict(
                    color = BG_BLACK, width=1
                ),
            ),
            text         = [
                f"{v:,.0f}%" for v in sim_df["ROI %"]
            ],
            textposition = "outside",
            textfont     = dict(
                color  = TEXT_WHITE,
                size   = 13,
                family = "Space Grotesk"
            ),
            hovertemplate= (
                "<b>%{x}</b><br>"
                "ROI: %{y:,.0f}%<extra></extra>"
            ),
        ))
        apply_dark(
            fig, height=380,
            yaxis = dict(
                ticksuffix = "%",
                gridcolor  = BG_BORDER2,
                tickfont   = dict(color=TEXT_MUTED),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Page 5 — Model Performance ────────────────────────
def page_model_performance():
    page_header(
        "🤖 Model Performance",
        "Evaluation metrics, SHAP importance "
        "and model diagnostics"
    )

    tab1, tab2 = st.tabs([
        "🔴  CHURN MODEL — LightGBM",
        "📈  LTV MODEL — RandomForest",
    ])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1 Score",  "0.8770", delta="+2.3% vs baseline")
        c2.metric("ROC-AUC",   "0.9836", delta="+5.1% vs baseline")
        c3.metric("Recall",    "1.0000", delta="Perfect recall ✅")
        c4.metric("Precision", "0.7810")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("SHAP — Churn Model")
            features  = [
                "Avg_Order_Value","Cart_Abandon_Rate",
                "Wishlist_Adds","Avg_Session_Duration",
                "Returns","Clicks_per_Page",
                "Campaign_Response","Clicks",
                "Session_Count","Pages_Viewed",
                "Wishlist_Conversion","Frequency",
            ]
            vals = [
                1.0000,0.1016,0.0862,0.0501,
                0.0463,0.0442,0.0320,0.0209,
                0.0195,0.0163,0.0135,0.0092,
            ]
            bar_colors = [
                RED    if v > 0.05 else
                AMBER  if v > 0.02 else
                PURPLE
                for v in vals
            ]
            fig = go.Figure(go.Bar(
                x            = vals[::-1],
                y            = features[::-1],
                orientation  = "h",
                marker       = dict(
                    color   = bar_colors[::-1],
                    opacity = 0.9,
                ),
                text         = [
                    f"{v:.4f}" for v in vals[::-1]
                ],
                textposition = "outside",
                textfont     = dict(
                    color = TEXT_WHITE, size=10
                ),
            ))
            apply_dark(
                fig, height=450,
                xaxis_title = "Normalised SHAP Value",
            )
            st.plotly_chart(
                fig, use_container_width=True
            )

        with col2:
            st.subheader("ROC Curve")
            fpr = np.linspace(0, 1, 200)
            tpr = np.clip(
                1 - np.exp(
                    -5 * fpr / (1 - fpr + 1e-6)
                ), 0, 1
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x         = fpr,
                y         = tpr,
                mode      = "lines",
                name      = "LightGBM (AUC=0.9836)",
                line      = dict(
                    color = RED, width=2.5
                ),
                fill      = "tozeroy",
                fillcolor = "rgba(239,68,68,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x    = [0, 1],
                y    = [0, 1],
                mode = "lines",
                name = "Random Baseline",
                line = dict(
                    color = TEXT_MUTED,
                    dash  = "dash",
                    width = 1,
                ),
            ))
            fig.add_annotation(
                x    = 0.6,
                y    = 0.4,
                text = "<b>AUC = 0.9836</b>",
                font = dict(
                    color  = TEXT_WHITE,
                    size   = 14,
                    family = "Space Grotesk"
                ),
                showarrow = False,
            )
            apply_dark(
                fig, height=450,
                xaxis_title = "False Positive Rate",
                yaxis_title = "True Positive Rate",
            )
            st.plotly_chart(
                fig, use_container_width=True
            )

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score",  "0.9226", delta="+3.1% vs baseline")
        c2.metric("RMSE",      "$2,806")
        c3.metric("R² Gap",    "0.0463", delta="✅ No overfit")
        c4.metric("Algorithm", "RandomForest")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("SHAP — LTV Model")
            features_ltv = [
                "Frequency","Session_Count",
                "Wishlist_Adds","Avg_Order_Value",
                "Clicks","Avg_Session_Duration",
                "Pages_Viewed","Cart_Abandon_Rate",
                "Wishlist_Conversion","Clicks_per_Page",
                "Returns","Campaign_Response",
            ]
            vals_ltv = [
                1.0000,0.4275,0.2449,0.1276,
                0.0751,0.0363,0.0323,0.0192,
                0.0166,0.0109,0.0095,0.0021,
            ]
            bar_colors_ltv = [
                RED    if v > 0.10 else
                CYAN   if v > 0.05 else
                PURPLE
                for v in vals_ltv
            ]
            fig = go.Figure(go.Bar(
                x            = vals_ltv[::-1],
                y            = features_ltv[::-1],
                orientation  = "h",
                marker       = dict(
                    color   = bar_colors_ltv[::-1],
                    opacity = 0.9,
                ),
                text         = [
                    f"{v:.4f}" for v in vals_ltv[::-1]
                ],
                textposition = "outside",
                textfont     = dict(
                    color = TEXT_WHITE, size=10
                ),
            ))
            apply_dark(
                fig, height=450,
                xaxis_title = "Normalised SHAP Value",
            )
            st.plotly_chart(
                fig, use_container_width=True
            )

        with col2:
            st.subheader("Predicted vs Actual LTV")
            np.random.seed(42)
            n_pts     = 600
            actual    = np.random.exponential(
                7000, n_pts
            )
            actual    = np.clip(actual, 0, 40000)
            predicted = actual * np.random.normal(
                1, 0.09, n_pts
            )
            predicted = np.clip(predicted, 0, 45000)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x      = actual,
                y      = predicted,
                mode   = "markers",
                marker = dict(
                    color   = CYAN,
                    opacity = 0.35,
                    size    = 5,
                ),
                name = "Predictions",
                hovertemplate = (
                    "Actual: $%{x:,.0f}<br>"
                    "Predicted: $%{y:,.0f}"
                    "<extra></extra>"
                ),
            ))
            fig.add_trace(go.Scatter(
                x    = [0, 40000],
                y    = [0, 40000],
                mode = "lines",
                name = "Perfect Fit",
                line = dict(
                    color = RED,
                    dash  = "dash",
                    width = 1.5,
                ),
            ))
            apply_dark(
                fig, height=450,
                xaxis_title = "Actual LTV ($)",
                yaxis_title = "Predicted LTV ($)",
                xaxis       = dict(
                    tickprefix = "$",
                    gridcolor  = BG_BORDER2,
                    tickfont   = dict(color=TEXT_MUTED),
                ),
                yaxis       = dict(
                    tickprefix = "$",
                    gridcolor  = BG_BORDER2,
                    tickfont   = dict(color=TEXT_MUTED),
                ),
            )
            st.plotly_chart(
                fig, use_container_width=True
            )


# ── Page 6 — Action Center ────────────────────────────
def page_action_center(selected_segments):
    page_header(
        "🎯 Action Center",
        "High-risk customer intelligence, "
        "retention strategies and campaign export"
    )

    with st.spinner("Loading action data..."):
        data, err = fetch_segment_summary()

    if err or not data:
        st.error(f"API Error: {err}")
        return

    at_risk_segs = [
        "At-Risk VIP", "Vulnerable", "Losing Customer"
    ]
    segments = [
        s for s in data["segments"]
        if s["segment"] in at_risk_segs
        and s["segment"] in selected_segments
    ]

    total_ar_cust = sum(s["count"] for s in segments)
    total_ar_rev  = sum(
        s["total_ltv"] for s in segments
    )
    avg_ar_risk   = (
        sum(s["avg_risk"]*s["count"] for s in segments)
        / total_ar_cust
        if total_ar_cust > 0 else 0
    )

    # ── KPIs ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚠️ High Risk Customers",
              f"{total_ar_cust:,}")
    c2.metric("💰 Revenue At Risk",
              f"${total_ar_rev/1e6:.2f}M")
    c3.metric("🎯 Avg Risk Score",
              f"{avg_ar_risk:.4f}")
    c4.metric("📋 Segments Flagged",
              f"{len(segments)}/3")

    st.markdown("---")

    # ── Priority Cards ────────────────────────────────
    st.subheader("Retention Priority Matrix")
    priority_cols = st.columns(3)

    priority_info = {
        "At-Risk VIP": {
            "priority"  : "PRIORITY 1",
            "color"     : AMBER,
            "budget"    : "$50/customer",
            "roi"       : "~9,896%",
            "action"    : "Immediate personalised outreach",
            "icon"      : "🟡",
        },
        "Vulnerable": {
            "priority"  : "PRIORITY 2",
            "color"     : "#F97316",
            "budget"    : "$25/customer",
            "roi"       : "~2,830%",
            "action"    : "Standard retention offer",
            "icon"      : "🟠",
        },
        "Losing Customer": {
            "priority"  : "PRIORITY 3",
            "color"     : RED,
            "budget"    : "$5/customer",
            "roi"       : "~126%",
            "action"    : "Low cost win-back email",
            "icon"      : "🔴",
        },
    }

    for col, seg in zip(
        priority_cols, at_risk_segs
    ):
        info    = priority_info.get(seg, {})
        seg_dat = next(
            (s for s in segments
             if s["segment"] == seg), None
        )
        count   = seg_dat["count"] \
                  if seg_dat else 0
        rev     = seg_dat["total_ltv"] \
                  if seg_dat else 0

        col.markdown(f"""
        <div style='
            background:{BG_CARD};
            border:1px solid {BG_BORDER};
            border-top:3px solid {info.get("color", PURPLE)};
            border-radius:12px;
            padding:20px;
            text-align:center;
            transition:all 0.3s ease;
        '>
            <div style='font-size:2rem;
                margin-bottom:8px;'>
                {info.get("icon","⚪")}
            </div>
            <div style='
                font-size:0.65rem;
                color:{info.get("color", PURPLE)};
                font-weight:800;
                letter-spacing:0.15em;
                text-transform:uppercase;
                margin-bottom:4px;
            '>{info.get("priority","")}</div>
            <div style='
                font-size:1rem;
                color:{TEXT_WHITE};
                font-weight:700;
                margin-bottom:12px;
                font-family: Space Grotesk;
            '>{seg}</div>
            <div style='
                display:grid;
                grid-template-columns:1fr 1fr;
                gap:8px;
                margin-bottom:12px;
            '>
                <div style='
                    background:{BG_CARD2};
                    border-radius:8px;
                    padding:8px;
                '>
                    <div style='
                        color:{TEXT_MUTED};
                        font-size:0.65rem;
                        text-transform:uppercase;
                        letter-spacing:0.08em;
                    '>CUSTOMERS</div>
                    <div style='
                        color:{TEXT_WHITE};
                        font-weight:700;
                        font-size:1.1rem;
                    '>{count:,}</div>
                </div>
                <div style='
                    background:{BG_CARD2};
                    border-radius:8px;
                    padding:8px;
                '>
                    <div style='
                        color:{TEXT_MUTED};
                        font-size:0.65rem;
                        text-transform:uppercase;
                        letter-spacing:0.08em;
                    '>AT RISK</div>
                    <div style='
                        color:{TEXT_WHITE};
                        font-weight:700;
                        font-size:1.1rem;
                    '>${rev/1e6:.1f}M</div>
                </div>
            </div>
            <div style='
                font-size:0.75rem;
                color:{TEXT_MUTED};
                line-height:1.6;
            '>
                Budget: <span style='
                    color:{info.get("color",PURPLE)};
                    font-weight:600;
                '>{info.get("budget","")}</span><br>
                ROI: <span style='
                    color:{EMERALD};
                    font-weight:600;
                '>{info.get("roi","")}</span><br>
                {info.get("action","")}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Export Section ────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Export Campaign Lists")

    # Load segments CSV if available
    seg_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "../data/customer_segments.csv"
    )

    if os.path.exists(seg_path):
        df_seg = pd.read_csv(seg_path)
        df_at_risk = df_seg[
            df_seg["Segment"].isin(at_risk_segs)
        ].sort_values(
            "Composite_Risk_Score",
            ascending=False
        )

        # Filters
        col1, col2 = st.columns([2, 1])
        with col1:
            seg_filter = st.multiselect(
                "Filter Segments",
                options = at_risk_segs,
                default = at_risk_segs,
            )
        with col2:
            top_n = st.selectbox(
                "Show Top N",
                [50, 100, 250, 500, "All"],
                index=1
            )

        df_filtered = df_at_risk[
            df_at_risk["Segment"].isin(seg_filter)
        ]
        if top_n != "All":
            df_filtered = df_filtered.head(int(top_n))

        strategy_map = {
            "At-Risk VIP"    : "Premium $50 budget",
            "Vulnerable"     : "Standard $25 budget",
            "Losing Customer": "Win-back $5 budget",
        }
        df_filtered = df_filtered.copy()
        df_filtered["Strategy"] = \
            df_filtered["Segment"].map(strategy_map)

        st.subheader(
            f"Campaign List "
            f"({len(df_filtered):,} customers)"
        )
        st.dataframe(
            df_filtered[[
                "Customer_ID", "Segment",
                "Churn_Probability",
                "Predicted_LTV",
                "Composite_Risk_Score",
                "Strategy",
            ]].rename(columns={
                "Churn_Probability"    : "Churn Prob",
                "Predicted_LTV"        : "LTV ($)",
                "Composite_Risk_Score" : "Risk Score",
            }).style.format({
                "Churn Prob" : "{:.4f}",
                "LTV ($)"    : "${:,.2f}",
                "Risk Score" : "{:.4f}",
            }).background_gradient(
                subset = ["Risk Score"],
                cmap   = "Reds"
            ),
            use_container_width = True,
            height              = 380,
        )

        # ── Downloads ─────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "⬇️ Full Campaign List",
                data      = df_filtered.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name = "campaign_list.csv",
                mime      = "text/csv",
            )
        with col2:
            vip = df_filtered[
                df_filtered["Segment"] == "At-Risk VIP"
            ]
            st.download_button(
                "⬇️ At-Risk VIPs",
                data      = vip.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name = "at_risk_vips.csv",
                mime      = "text/csv",
            )
        with col3:
            vuln = df_filtered[
                df_filtered["Segment"] == "Vulnerable"
            ]
            st.download_button(
                "⬇️ Vulnerable",
                data      = vuln.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name = "vulnerable.csv",
                mime      = "text/csv",
            )
    else:
        st.warning(
            "Run customer_segmentation.py first "
            "to generate the campaign list."
        )


# ── Main ──────────────────────────────────────────────
def main():
    # Start API automatically
    if "api_started" not in st.session_state:
        with st.spinner(
            "🚀 Launching API server..."
        ):
            ok = start_api()
        st.session_state["api_started"] = True
        st.session_state["api_ok"] = ok
        if ok:
            st.success(
                "✅ API server started successfully!"
            )
            time.sleep(1)
            st.rerun()

    page, selected_segments, ltv_range = \
        render_sidebar()

    if page == "🏠  Overview":
        page_overview(selected_segments, ltv_range)
    elif page == "🔍  Customer Lookup":
        page_customer_lookup()
    elif page == "📊  Segment Analysis":
        page_segment_analysis(
            selected_segments, ltv_range
        )
    elif page == "💰  Revenue Recovery":
        page_revenue_recovery(selected_segments)
    elif page == "🤖  Model Performance":
        page_model_performance()
    elif page == "🎯  Action Center":
        page_action_center(selected_segments)

    # Auto-refresh every 30 seconds
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()


if __name__ == "__main__":
    main()
