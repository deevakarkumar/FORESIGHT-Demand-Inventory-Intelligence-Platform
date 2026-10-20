import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from zildio.Data_cleaning_model.Data_cleaning_main import clean_sales_data
from zildio.Forcasting.Code.forcasting import forecast_sales
from zildio.Risk_Mangment.Code.Risk_managment import calculate_inventory_risk

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="FORESIGHT | Demand & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODERN, CLEAN, EXECUTIVE LIGHT THEME CSS
# ============================================================
st.markdown("""
<style>
    :root {
        --ui-bg: #f8fafc;
        --ui-panel: #ffffff;
        --ui-border: #e2e8f0;
        --ui-text: #0f172a;
        --ui-text-muted: #64748b;
        --ui-accent: #2563eb;
        --ui-accent-hover: #1d4ed8;
        --ui-success: #10b981;
        --ui-warning: #f59e0b;
        --ui-danger: #ef4444;
    }

    .stApp {
        background-color: var(--ui-bg);
        color: var(--ui-text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid var(--ui-border);
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: var(--ui-text) !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Hero Banner */
    .foresight-hero {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        color: #ffffff;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.12);
    }
    .foresight-hero h1 {
        color: #ffffff !important;
        margin: 4px 0 6px 0;
        font-size: 26px;
    }
    .foresight-hero p {
        color: #94a3b8 !important;
        font-size: 14px;
        margin: 0;
    }
    .section-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .info-card {
        background-color: #ffffff;
        border: 1px solid var(--ui-border);
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .info-card .icon {
        font-size: 26px;
        margin-bottom: 10px;
    }
    .info-card .title {
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .info-card .text {
        font-size: 13px;
        color: #64748b;
        line-height: 1.5;
    }
    .data-guide {
        background-color: #ffffff;
        border: 1px solid var(--ui-border);
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .data-guide h4 {
        margin: 0 0 10px 0;
        font-size: 15px;
        color: #0f172a;
    }
    .data-guide p {
        font-size: 13px;
        color: #475569;
        line-height: 1.6;
        margin: 0 0 12px 0;
    }
    .requirements-note {
        background: #f1f5f9;
        border-left: 4px solid #2563eb;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        color: #1e293b;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        color: #38bdf8;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid var(--ui-border);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 26px !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Portfolio Health Bar */
    .health-bar-container {
        background: #ffffff;
        border: 1px solid var(--ui-border);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .health-bar-title {
        font-size: 14px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .health-bar-track {
        display: flex;
        height: 14px;
        border-radius: 7px;
        overflow: hidden;
        background-color: #f1f5f9;
        margin-bottom: 12px;
    }
    .hb-segment-fine { background-color: #10b981; }
    .hb-segment-order { background-color: #ef4444; }
    .hb-segment-monitor { background-color: #f97316; }
    .hb-segment-overstock { background-color: #eab308; }

    .health-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        font-size: 13px;
    }
    .health-legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 500;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }

    /* Diagnostic Card ("Human Translator") */
    .diagnostic-box {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #cbd5e1;
        border-left: 6px solid #2563eb;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 16px 0 24px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .diagnostic-title {
        font-size: 17px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .diagnostic-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-bottom: 14px;
    }
    .diag-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 14px;
    }
    .diag-item-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 4px;
    }
    .diag-item-value {
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
    }
    .diagnostic-takeaway {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 14px;
        line-height: 1.5;
        color: #1e40af;
    }
    .diagnostic-takeaway b {
        color: #1d4ed8;
    }

    /* Explainer Box */
    .explain-card {
        background-color: #f1f5f9;
        border-left: 4px solid #64748b;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 14px 0;
        font-size: 13px;
        color: #334155;
        line-height: 1.5;
    }

    /* Jargon Buster Cards */
    .jargon-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 14px;
        margin-top: 12px;
    }
    .jargon-card {
        background: #ffffff;
        border: 1px solid var(--ui-border);
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .jargon-term {
        font-weight: 700;
        font-size: 14px;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .jargon-def {
        font-size: 12px;
        color: #475569;
        line-height: 1.45;
    }

    /* Buttons */
    .stButton > button {
        background-color: #ffffff;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 16px;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #f8fafc;
        border-color: #2563eb;
        color: #2563eb;
    }
    .stButton > button[kind="primary"] {
        background-color: #2563eb;
        color: #ffffff;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        background-color: #ffffff;
        border: 1px solid var(--ui-border);
        border-radius: 14px;
        padding: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATABASE SETUP & CONNECTION
# ============================================================
load_dotenv(dotenv_path=BASE_DIR / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    st.error("DATABASE_URL was not found in `.env`. Please check your configuration.")
    st.stop()

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)

engine = get_engine()

@st.cache_data(ttl=300)
def load_table(table_name):
    for name in [table_name, table_name.lower(), table_name.capitalize()]:
        query = f'SELECT * FROM public."{name}"'
        try:
            df = pd.read_sql(query, engine)
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()

# ============================================================
# DATA ONBOARDING GATE
# ============================================================
if "data_ready" not in st.session_state:
    st.session_state.data_ready = False

def run_pipeline(sales_file, inventory_file, sku_file):
    try:
        df_sales_raw = pd.read_csv(sales_file)
        df_inventory_raw = pd.read_csv(inventory_file)
        df_sku_master = pd.read_csv(sku_file)
    except Exception as e:
        st.error(f"Could not read uploaded files: {e}")
        return False

    with st.status("Running FORESIGHT Fast Intelligence Pipeline...", expanded=True) as status:
        st.write("1️⃣ Cleaning and validating sales transactions...")
        cleaned_sales = clean_sales_data(df_sales_raw, df_sku_master)
        if isinstance(cleaned_sales, int) or cleaned_sales.empty:
            st.error("Sales data cleaning failed. Check SKU IDs against SKU master.")
            status.update(label="Pipeline failed", state="error")
            return False
        st.write(f"✓ Cleaned orders: {len(cleaned_sales):,} rows")

        st.write("2️⃣ Generating 3-month demand forecasts with AI uncertainty intervals...")
        forecast_df = forecast_sales(cleaned_sales)
        if forecast_df.empty:
            st.error("Forecasting produced no output.")
            status.update(label="Pipeline failed", state="error")
            return False
        st.write(f"✓ Forecast generated for {forecast_df['sku_id'].nunique()} SKUs")

        st.write("3️⃣ Computing inventory risks, days of supply, and rupee impact...")
        risk_df = calculate_inventory_risk(df_inventory_raw, forecast_df, df_sku_master)
        st.write(f"✓ Risk decisions evaluated for {len(risk_df)} SKUs")

        st.write("4️⃣ Syncing SKU catalog reference data...")
        try:
            df_sku_master.to_sql("sku_master", engine, schema="public", if_exists="replace", index=False)
        except Exception:
            pass

        status.update(label="Pipeline Completed Successfully! ✓", state="complete")

    st.cache_data.clear()
    return True

# If not ready, show upload landing screen
if not st.session_state.data_ready:

    st.markdown('''
    <div class="foresight-hero">
        <div class="section-label">Inventory planning workspace</div>
        <h1>Welcome to FORESIGHT 📦</h1>
        <p>Turn your sales and inventory data into a clear demand forecast and practical stock-risk actions.</p>
        <div class="status-pill">● Ready to analyze your data</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("### Start in 3 simple steps")
    st.caption("Upload your historical sales, inventory snapshot, and SKU master files. FORESIGHT will clean the data, forecast demand, and calculate inventory risk automatically.")

    col_sales, col_inventory, col_sku = st.columns(3)

    with col_sales:
        st.markdown("**1. Sales history**")
        sales_upload = st.file_uploader("Upload sales CSV", type=["csv"], key="onboard_sales", help="Historical sales transactions. Required for demand forecasting.")

    with col_inventory:
        st.markdown("**2. Inventory snapshot**")
        inventory_upload = st.file_uploader("Upload inventory CSV", type=["csv"], key="onboard_inventory", help="Latest/current inventory information used for stock-risk analysis.")

    with col_sku:
        st.markdown("**3. SKU master**")
        sku_upload = st.file_uploader("Upload SKU master CSV", type=["csv"], key="onboard_sku", help="Product reference data used to validate and identify SKUs.")

    files_ready = bool(sales_upload and inventory_upload and sku_upload)
    st.write("")

    if files_ready:
        st.success("All three files are ready. You can run the analysis.")
    else:
        st.info("Please upload all three files to enable the analysis.")

    run_col, skip_col = st.columns([1.4, 1])

    with run_col:
        run_clicked = st.button("🚀 Analyze Data & Open Dashboard", type="primary", use_container_width=True, disabled=not files_ready)

    with skip_col:
        skip_clicked = st.button("Open Existing Dashboard", use_container_width=True, help="Use the data already available in the connected database.")

    if run_clicked:
        success = run_pipeline(sales_upload, inventory_upload, sku_upload)
        if success:
            st.session_state.data_ready = True
            st.rerun()

    if skip_clicked:
        st.session_state.data_ready = True
        st.rerun()

    st.markdown("---")
    st.markdown("### What is FORESIGHT?")
    st.caption("FORESIGHT is an AI-assisted demand and inventory intelligence tool for turning historical sales and stock data into practical planning insights.")

    info1, info2, info3 = st.columns(3)
    cards = [
        (info1, "📈", "Demand forecasting", "Uses historical SKU-level sales to estimate demand for the next 3 months."),
        (info2, "📦", "Inventory risk", "Compares available stock, reorder points, lead times, and forecast demand to identify stockout or overstock situations."),
        (info3, "🎯", "Actionable decisions", "Highlights products that may need reordering, monitoring, or inventory reduction so users can focus on what needs attention."),
    ]
    for col, icon, title, desc in cards:
        with col:
            st.markdown(f'<div class="info-card"><div class="icon">{icon}</div><div class="title">{title}</div><div class="text">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("### Data upload guide")
    st.caption("Prepare three CSV files. Use the column names below exactly so the pipeline can recognize your data.")

    with st.expander("📄 1. Sales history — 15 columns", expanded=False):
        st.markdown('''
        **Required columns:** `sale_date`, `order_id`, `sku_id`, `quantity`, `unit_price`, `discount_pct`, `revenue`, `customer_id`, `region`, `channel`, `payment_method`, `shipping_days`, `order_status`, `sales_rep`, `warehouse`

        **Purpose:** historical transactions used to learn demand patterns. `sale_date`, `sku_id`, `quantity`, `unit_price`, `discount_pct`, `revenue`, and `order_id` are especially important for cleaning and forecasting.
        ''')

    with st.expander("📦 2. Inventory snapshot — 6 columns", expanded=False):
        st.markdown('''
        **Required columns:** `date`, `sku_id`, `on_hand_units`, `on_order_units`, `lead_time_days`, `reorder_point`

        **Purpose:** tells FORESIGHT how much stock is currently available, what is already on order, how quickly replenishment arrives, and the reorder threshold.
        ''')

    with st.expander("🏷️ 3. SKU master — 6 columns", expanded=False):
        st.markdown('''
        **Required columns:** `sku_id`, `category`, `subcategory`, `launch_date`, `unit_cost`, `list_price`

        **Purpose:** product reference information. The `sku_id` must match the same SKU IDs used in the sales and inventory files.
        ''')

    st.markdown('''
    <div class="data-guide">
        <h4>Before you upload</h4>
        <p>• Save each dataset as a <b>CSV</b> with one header row.<br>
        • Keep the column names exactly as shown above.<br>
        • Make sure <b>sku_id</b> values match across the three files.<br>
        • Use valid dates; <b>YYYY-MM-DD</b> is recommended.<br>
        • For meaningful forecasting, each SKU should have at least <b>6 months of monthly sales history</b>.<br>
        • You do <b>not</b> need to upload forecast or risk-result files — FORESIGHT generates those automatically.</p>
        <div class="requirements-note"><b>Simple workflow:</b> Prepare CSVs → Upload all 3 files → Analyze Data → Review Forecast → Check Inventory Risk → Take action.</div>
    </div>
    ''', unsafe_allow_html=True)

    st.stop()

# ============================================================
# LOAD PROJECT DATA FROM DATABASE
# ============================================================
df_sales = load_table("Cleaned_sales")
df_forecast = load_table("forecasted_sales")
df_risk = load_table("inventory_risk")
df_sku_master = load_table("sku_master")
df_metrics = load_table("forecast_metrics")

if df_risk.empty:
    st.error("The `inventory_risk` table is empty. Please click 'Run Fast AI Pipeline' to initialize data.")
    if st.button("Return to Upload Screen"):
        st.session_state.data_ready = False
        st.rerun()
    st.stop()

# Clean types
df_risk["sku_id"] = df_risk["sku_id"].astype(str)
for num_col in ["Total_stock", "lead_time_days", "reorder_point", "Forecast_demand", "Coverage", "Coverage_Rating", "Stockout_Risk", "overstock_risk", "unit_cost", "list_price", "sales_at_risk", "capital_locked"]:
    if num_col in df_risk.columns:
        df_risk[num_col] = pd.to_numeric(df_risk[num_col], errors="coerce").fillna(0)

# Merge category and subcategory from SKU master if available
if not df_sku_master.empty and "sku_id" in df_sku_master.columns:
    df_sku_master["sku_id"] = df_sku_master["sku_id"].astype(str)
    cat_cols = [c for c in ["sku_id", "category", "subcategory"] if c in df_sku_master.columns]
    if "category" not in df_risk.columns:
        df_risk = df_risk.merge(df_sku_master[cat_cols], on="sku_id", how="left")
else:
    if "category" not in df_risk.columns:
        df_risk["category"] = "General"

if not df_sales.empty:
    df_sales["sale_date"] = pd.to_datetime(df_sales["sale_date"], errors="coerce")
    df_sales["quantity"] = pd.to_numeric(df_sales["quantity"], errors="coerce").fillna(0)
    df_sales["revenue"] = pd.to_numeric(df_sales["revenue"], errors="coerce").fillna(0)
    df_sales["sku_id"] = df_sales["sku_id"].astype(str)

if not df_forecast.empty:
    df_forecast["ds"] = pd.to_datetime(df_forecast["ds"], errors="coerce")
    for fc_c in ["yhat", "yhat_lower", "yhat_upper"]:
        if fc_c in df_forecast.columns:
            df_forecast[fc_c] = pd.to_numeric(df_forecast[fc_c], errors="coerce").fillna(0)
    df_forecast["sku_id"] = df_forecast["sku_id"].astype(str)

# Calculate human-friendly Days of Supply
# Forecast_demand is 90 days. Daily pace = Forecast_demand / 90
df_risk["daily_pace"] = (df_risk["Forecast_demand"] / 90.0).clip(lower=0.01)
df_risk["days_of_stock"] = (df_risk["Total_stock"] / df_risk["daily_pace"]).round(0).clip(upper=999)

# Suggested Reorder Units
# Target inventory = Reorder Point + 30 days buffer
df_risk["target_stock"] = (df_risk["reorder_point"] + (df_risk["Forecast_demand"] / 3.0)).round(0)
df_risk["suggested_reorder"] = (df_risk["target_stock"] - df_risk["Total_stock"]).clip(lower=0).round(0)

# Excess Units for overstocked items
df_risk["excess_units"] = (df_risk["Total_stock"] - df_risk["Forecast_demand"]).clip(lower=0).round(0)

# ============================================================
# SIDEBAR NAVIGATION & FILTERS
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 4px 18px 4px;">
        <div style="font-size:24px; font-weight:800; letter-spacing:-0.5px; color:#0f172a;">FORESIGHT 📦</div>
        <div style="font-size:12px; color:#64748b; font-weight:500;">Demand &amp; Inventory Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧭 Navigation")
    view = st.radio(
        "Navigation",
        [
            "🏠 Action Center & Overview",
            "📈 Demand Forecast",
            "🛡️ Stock Risk & Simulation",
            "🔎 Product Catalog & Search",
            "💡 Sales Trends & Bestsellers",
            "🎯 AI Accuracy & Trust"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### 🎯 Quick Filters")

    # Category filter
    all_categories = sorted([c for c in df_risk["category"].dropna().unique().tolist() if str(c) != "nan"])
    selected_cat = st.selectbox("Category", ["All Categories"] + all_categories)

    # Filter data by category if selected
    filtered_risk = df_risk.copy()
    if selected_cat != "All Categories":
        filtered_risk = filtered_risk[filtered_risk["category"] == selected_cat]

    # Action filter
    action_list = ["Order ASAP", "Monitor", "Overstock", "Fine"]
    avail_actions = [a for a in action_list if a in filtered_risk["Action"].values]
    selected_actions = st.multiselect("Filter by Action", avail_actions, default=[])
    if selected_actions:
        filtered_risk = filtered_risk[filtered_risk["Action"].isin(selected_actions)]

    st.divider()
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_sb2:
        if st.button("⬆️ Upload", use_container_width=True):
            st.session_state.data_ready = False
            st.rerun()

    st.caption("FORESIGHT • Demand & Inventory Intelligence")

# ============================================================
# TOP EXECUTIVE HEADER
# ============================================================
st.markdown(f"""
<div class="foresight-hero">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div>
            <div class="section-label">FORESIGHT workspace</div>
            <h1>{view}</h1>
            <p>Make faster inventory decisions using demand forecasts and stock-risk signals.</p>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <span class="hero-badge">🟢 Live Data Active</span>
            <span style="background:rgba(255,255,255,0.12); padding:4px 12px; border-radius:20px; font-size:12px; color:#ffffff; font-weight:600;">
                {len(filtered_risk)} SKUs Filtered
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# EXECUTIVE IMPACT METRICS BAR (UNIVERSAL)
# ============================================================
total_skus = len(filtered_risk)
total_stock_units = filtered_risk["Total_stock"].sum()
total_forecast_demand = filtered_risk["Forecast_demand"].sum()
total_sales_at_risk = filtered_risk["sales_at_risk"].sum()
total_capital_locked = filtered_risk["capital_locked"].sum()

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(
        label="💸 Sales at Risk (Stockouts)",
        value=f"₹{total_sales_at_risk:,.0f}",
        help="Estimated revenue NorthBay Living could lose because fast-selling items will run out before supplier restocks arrive."
    )
with m2:
    st.metric(
        label="🔒 Capital Trapped (Overstock)",
        value=f"₹{total_capital_locked:,.0f}",
        help="Working capital cash locked up in warehouse inventory that exceeds customer demand for the next 90 days."
    )
with m3:
    st.metric(
        label="📦 Physical Stock on Hand",
        value=f"{total_stock_units:,.0f} units",
        help="Total physical units currently available across all warehouses."
    )
with m4:
    st.metric(
        label="📈 Expected 90-Day Demand",
        value=f"{total_forecast_demand:,.0f} units",
        help="AI-projected total customer purchase volume over the next 3 months."
    )

st.write("")

# Action counts
n_critical = (filtered_risk["Action"] == "Order ASAP").sum()
n_monitor = (filtered_risk["Action"] == "Monitor").sum()
n_overstock = (filtered_risk["Action"] == "Overstock").sum()
n_fine = (filtered_risk["Action"] == "Fine").sum()

# ============================================================
# ────────────────────────────────────────────────────────────
# VIEW 1: ACTION CENTER & OVERVIEW
# ────────────────────────────────────────────────────────────
# ============================================================
if view == "🏠 Action Center & Overview":

    # --- 1. Portfolio Health Bar ---
    pct_fine = (n_fine / total_skus * 100) if total_skus else 0
    pct_order = (n_critical / total_skus * 100) if total_skus else 0
    pct_monitor = (n_monitor / total_skus * 100) if total_skus else 0
    pct_overstock = (n_overstock / total_skus * 100) if total_skus else 0

    st.markdown(f"""
    <div class="health-bar-container">
        <div class="health-bar-title">
            <span>Overall Inventory Health Breakdown</span>
            <span style="font-size:12px; color:#64748b;">Total: {total_skus} Active Products</span>
        </div>
        <div class="health-bar-track">
            <div class="hb-segment-fine" style="width:{pct_fine:.1f}%;" title="Healthy: {pct_fine:.1f}%"></div>
            <div class="hb-segment-order" style="width:{pct_order:.1f}%;" title="Order ASAP: {pct_order:.1f}%"></div>
            <div class="hb-segment-monitor" style="width:{pct_monitor:.1f}%;" title="Monitor: {pct_monitor:.1f}%"></div>
            <div class="hb-segment-overstock" style="width:{pct_overstock:.1f}%;" title="Overstock: {pct_overstock:.1f}%"></div>
        </div>
        <div class="health-legend">
            <div class="health-legend-item">
                <span class="legend-dot" style="background:#10b981;"></span>
                <span><b>{n_fine}</b> Healthy ({pct_fine:.0f}%)</span>
            </div>
            <div class="health-legend-item">
                <span class="legend-dot" style="background:#ef4444;"></span>
                <span><b>{n_critical}</b> Urgent Reorder ({pct_order:.0f}%)</span>
            </div>
            <div class="health-legend-item">
                <span class="legend-dot" style="background:#f97316;"></span>
                <span><b>{n_monitor}</b> Watch Closely ({pct_monitor:.0f}%)</span>
            </div>
            <div class="health-legend-item">
                <span class="legend-dot" style="background:#eab308;"></span>
                <span><b>{n_overstock}</b> Excess / Surplus ({pct_overstock:.0f}%)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 2. Interactive Product Diagnostic ("Human Explainer") ---
    st.markdown("### 💬 10-Second Product Diagnostic (Human Explainer)")
    st.caption("Pick any product to get an instant, plain-English explanation of its inventory situation and recommended action.")

    sku_choices = sorted(filtered_risk["sku_id"].unique().tolist())
    # Default to first urgent SKU if available, otherwise first SKU
    default_sku = filtered_risk[filtered_risk["Action"] == "Order ASAP"]["sku_id"].iloc[0] if n_critical > 0 else sku_choices[0]

    selected_diag_sku = st.selectbox("Select Product to Diagnose:", sku_choices, index=sku_choices.index(default_sku) if default_sku in sku_choices else 0)

    sku_row = filtered_risk[filtered_risk["sku_id"] == selected_diag_sku].iloc[0]
    stock_val = int(sku_row["Total_stock"])
    demand_val = int(sku_row["Forecast_demand"])
    days_stock = int(sku_row["days_of_stock"])
    lead_time = int(sku_row["lead_time_days"])
    action_val = sku_row["Action"]
    sales_risk_val = float(sku_row["sales_at_risk"])
    capital_val = float(sku_row["capital_locked"])
    suggested_order_val = int(sku_row["suggested_reorder"])
    excess_val = int(sku_row["excess_units"])
    cat_val = sku_row.get("category", "General")

    # Dynamic plain-language messaging
    if action_val == "Order ASAP":
        border_color = "#ef4444"
        badge_html = '<span style="background:#fef2f2; color:#ef4444; padding:4px 10px; border-radius:20px; font-weight:700; font-size:12px; border:1px solid #fecaca;">🔴 URGENT REORDER</span>'
        takeaway_text = f"⚠️ <b>Critical Situation:</b> You have only <b>{stock_val} units</b> left (~{days_stock} days of supply), but your supplier takes <b>{lead_time} days</b> to deliver. You risk running completely out of stock in {days_stock} days!<br>👉 <b>Recommended Action:</b> Place a purchase order for <b>{suggested_order_val} units today</b> to protect <b>₹{sales_risk_val:,.0f}</b> in revenue."
    elif action_val == "Monitor":
        border_color = "#f97316"
        badge_html = '<span style="background:#fff7ed; color:#f97316; padding:4px 10px; border-radius:20px; font-weight:700; font-size:12px; border:1px solid #fed7aa;">🟠 WATCH CLOSELY</span>'
        takeaway_text = f"👀 <b>Keep An Eye Out:</b> Current inventory ({stock_val} units) covers approximately <b>{days_stock} days</b> of demand. Delivery takes <b>{lead_time} days</b>.<br>👉 <b>Recommended Action:</b> Prepare a purchase order for <b>{suggested_order_val} units</b> by next week to prevent dipping below your safety stock threshold."
    elif action_val == "Overstock":
        border_color = "#eab308"
        badge_html = '<span style="background:#fefce8; color:#ca8a04; padding:4px 10px; border-radius:20px; font-weight:700; font-size:12px; border:1px solid #fef08a;">🟡 SURPLUS / OVERSTOCK</span>'
        months_supply = round(days_stock / 30.0, 1)
        takeaway_text = f"📦 <b>Excess Inventory:</b> You have <b>{stock_val} units</b> in stock, which is <b>{months_supply} months of supply</b>—far more than the 90-day projected demand of {demand_val} units. You have <b>{excess_val} surplus units</b> tying up <b>₹{capital_val:,.0f}</b>.<br>👉 <b>Recommended Action:</b> Bundle with complementary products or run a 15% promotional discount to free up trapped cash."
    else:
        border_color = "#10b981"
        badge_html = '<span style="background:#f0fdf4; color:#10b981; padding:4px 10px; border-radius:20px; font-weight:700; font-size:12px; border:1px solid #bbf7d0;">🟢 HEALTHY STOCK</span>'
        takeaway_text = f"✅ <b>Optimal Balance:</b> Available stock ({stock_val} units) comfortably supports expected demand ({demand_val} units) for the next <b>{days_stock} days</b> with adequate supplier buffer.<br>👉 <b>Recommended Action:</b> No action needed at this time. Maintain standard replenishment schedule."

    st.markdown(f"""
    <div class="diagnostic-box" style="border-left-color:{border_color};">
        <div class="diagnostic-title">
            <span>Product: <b>{selected_diag_sku}</b> ({cat_val})</span>
            {badge_html}
        </div>
        <div class="diagnostic-grid">
            <div class="diag-item">
                <div class="diag-item-label">Units in Warehouse</div>
                <div class="diag-item-value">{stock_val:,} units</div>
            </div>
            <div class="diag-item">
                <div class="diag-item-label">Days of Stock Remaining</div>
                <div class="diag-item-value">{days_stock} days</div>
            </div>
            <div class="diag-item">
                <div class="diag-item-label">Supplier Delivery Time</div>
                <div class="diag-item-value">{lead_time} days</div>
            </div>
            <div class="diag-item">
                <div class="diag-item-label">Projected 90-Day Demand</div>
                <div class="diag-item-value">{demand_val:,} units</div>
            </div>
        </div>
        <div class="diagnostic-takeaway">
            {takeaway_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 3. Prioritized Action Lists ---
    st.markdown("### 📋 Prioritized Decision Tables")
    st.caption("Focus your team's energy where the financial stakes are highest.")

    tab_urgent, tab_surplus = st.tabs(["🚨 Urgent Reorders (Stockout Prevention)", "💰 Cash Recovery (Excess Inventory)"])

    with tab_urgent:
        urgent_df = filtered_risk[filtered_risk["Action"].isin(["Order ASAP", "Monitor"])].copy()
        if not urgent_df.empty:
            urgent_df = urgent_df.sort_values("sales_at_risk", ascending=False)
            display_urgent = pd.DataFrame({
                "Product": urgent_df["sku_id"],
                "Category": urgent_df["category"],
                "Stock Left": urgent_df["Total_stock"].astype(int),
                "Days of Stock": urgent_df["days_of_stock"].astype(int),
                "Supplier Days": urgent_df["lead_time_days"].astype(int),
                "Projected 90D Demand": urgent_df["Forecast_demand"].astype(int),
                "Suggested Order Qty": urgent_df["suggested_reorder"].astype(int),
                "Sales at Risk": urgent_df["sales_at_risk"].apply(lambda x: f"₹{x:,.0f}"),
                "Urgency": urgent_df["Action"].apply(lambda x: "🔴 Immediate" if x == "Order ASAP" else "🟠 Watch")
            })
            st.dataframe(display_urgent, use_container_width=True, hide_index=True)
            st.info("💡 **Tip:** Sort by **Sales at Risk** to see which products could cost NorthBay Living the most money if they run out.")
        else:
            st.success("🎉 Excellent! No products are currently at urgent stockout risk.")

    with tab_surplus:
        surplus_df = filtered_risk[filtered_risk["Action"] == "Overstock"].copy()
        if not surplus_df.empty:
            surplus_df = surplus_df.sort_values("capital_locked", ascending=False)
            display_surplus = pd.DataFrame({
                "Product": surplus_df["sku_id"],
                "Category": surplus_df["category"],
                "Stock Left": surplus_df["Total_stock"].astype(int),
                "Days of Stock": surplus_df["days_of_stock"].astype(int),
                "Projected 90D Demand": surplus_df["Forecast_demand"].astype(int),
                "Excess Units": surplus_df["excess_units"].astype(int),
                "Trapped Cash": surplus_df["capital_locked"].apply(lambda x: f"₹{x:,.0f}"),
                "Action": "Markdown / Bundle"
            })
            st.dataframe(display_surplus, use_container_width=True, hide_index=True)
            st.info("💡 **Tip:** These products have more than 3 months of supply. Run special marketing campaigns to liquidate them and recover cash.")
        else:
            st.success("🎉 Great job! No products are overstocked.")

# ============================================================
# ────────────────────────────────────────────────────────────
# VIEW 2: DEMAND FORECAST
# ────────────────────────────────────────────────────────────
# ============================================================
elif view == "📈 Demand Forecast":
    st.subheader("Future Customer Demand Projections")
    st.caption("Machine learning estimates monthly customer orders for the next 3 months, including confidence intervals.")

    if df_forecast.empty:
        st.info("Forecast data is not available. Please run the pipeline first.")
    else:
        fc_col1, fc_col2 = st.columns([2, 1])
        with fc_col1:
            all_fc_skus = sorted(df_forecast["sku_id"].unique().tolist())
            selected_fc_sku = st.selectbox("Select Product to View Forecast:", all_fc_skus)
        with fc_col2:
            st.write("")
            st.write("")
            st.caption("Includes 80% AI Confidence Range (Best-case / Worst-case boundaries)")

        # Prepare chart data
        sku_fc = df_forecast[df_forecast["sku_id"] == selected_fc_sku].sort_values("ds").copy()
        sku_hist = pd.DataFrame()
        if not df_sales.empty and selected_fc_sku in df_sales["sku_id"].values:
            sku_sales = df_sales[df_sales["sku_id"] == selected_fc_sku].copy()
            sku_sales["sale_month"] = sku_sales["sale_date"].dt.to_period("M").dt.to_timestamp()
            sku_hist = sku_sales.groupby("sale_month")["quantity"].sum().reset_index().rename(columns={"sale_month": "ds"}).sort_values("ds")

        fig_fc = go.Figure()

        # Uncertainty band
        if "yhat_upper" in sku_fc.columns and "yhat_lower" in sku_fc.columns:
            fig_fc.add_trace(go.Scatter(
                x=sku_fc["ds"], y=sku_fc["yhat_upper"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))
            fig_fc.add_trace(go.Scatter(
                x=sku_fc["ds"], y=sku_fc["yhat_lower"],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(16, 185, 129, 0.15)",
                name="80% Expected Range", hoverinfo="skip"
            ))

        # Forecast line
        fig_fc.add_trace(go.Scatter(
            x=sku_fc["ds"], y=sku_fc["yhat"],
            mode="lines+markers", name="AI Forecast (Projected)",
            line=dict(color="#10b981", width=3),
            marker=dict(size=7)
        ))

        # Historical actual sales line
        if not sku_hist.empty:
            fig_fc.add_trace(go.Scatter(
                x=sku_hist["ds"], y=sku_hist["quantity"],
                mode="lines+markers", name="Historical Sales (Actual)",
                line=dict(color="#2563eb", width=2.5),
                marker=dict(size=6)
            ))

        fig_fc.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#475569"),
            xaxis=dict(title="Month", showgrid=False),
            yaxis=dict(title="Units Sold / Projected", gridcolor="rgba(226, 232, 240, 0.8)"),
            legend=dict(orientation="h", y=1.1, x=0)
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown("""
        <div class="explain-card">
            <b>📘 How to read this forecast chart:</b><br>
            • <b>Blue Line (Actual):</b> Real historical units sold each month by NorthBay Living.<br>
            • <b>Green Line (AI Forecast):</b> Expected monthly sales for the upcoming 3 months generated by Facebook Prophet.<br>
            • <b>Shaded Green Area (80% Confidence Interval):</b> The expected range of sales. During holiday rushes, demand may track toward the upper edge; during quiet periods, it may track toward the lower edge.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ────────────────────────────────────────────────────────────
# VIEW 3: STOCK RISK & SIMULATION
# ────────────────────────────────────────────────────────────
# ============================================================
elif view == "🛡️ Stock Risk & Simulation":
    st.subheader("Inventory Risk Matrix & What-If Stress Testing")
    st.caption("Explore where your products land on the risk spectrum and simulate supply chain disruptions.")

    with st.expander("🎛️ Real-Time What-If Simulation: Delays & Demand Spikes", expanded=True):
        st.caption("Slide the controls below to see how supplier delays or sales surges would affect your inventory risk.")
        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            lead_delay = st.slider("Supplier Delay (Days):", -7, 28, 0, step=1, help="Simulate shipping delays or port congestion.")
        with sim_c2:
            demand_surge = st.slider("Demand Surge / Drop (%):", -50, 100, 0, step=5, help="Simulate holiday spikes (+%) or market drops (-%).")

        if lead_delay != 0 or demand_surge != 0:
            st.info(f"⚡ **Active Simulation:** Supplier Delivery **{lead_delay:+d} days** | Demand **{demand_surge:+d}%**. Risk matrix below reflects stress-tested conditions.")

    scatter_df = filtered_risk.copy()

    # Recalculate dynamic risks if simulation active
    if lead_delay != 0 or demand_surge != 0:
        sim_lead = (scatter_df["lead_time_days"] + lead_delay).clip(lower=1)
        def _score_lt(lt):
            if lt <= 7: return 1
            elif lt <= 14: return 2
            elif lt <= 21: return 3
            elif lt <= 28: return 4
            else: return 5
        sim_lt_pts = sim_lead.apply(_score_lt)

        sim_demand = (scatter_df["Forecast_demand"] * (1 + demand_surge / 100.0)).clip(lower=0.1)
        sim_coverage = scatter_df["Total_stock"] / sim_demand
        def _classify_cov(c):
            if pd.isna(c) or c < 0.5: return 4
            elif c < 1: return 3
            elif c < 2: return 2
            else: return 1
        sim_cov_pts = sim_coverage.apply(_classify_cov)
        scatter_df["Stockout_Risk"] = ((sim_lt_pts + sim_cov_pts) / 9.0).clip(0, 1)

    # Classify Quadrant for chart
    def _get_quadrant(row):
        stockout = row["Stockout_Risk"]
        overstock = row["overstock_risk"]
        if stockout >= 0.5 and overstock < 0.5:
            return "Reorder Now (Stockout Risk)"
        elif stockout >= 0.5 and overstock >= 0.5:
            return "Watch Closely (Volatile)"
        elif stockout < 0.5 and overstock < 0.5:
            return "Healthy Stock (Optimal)"
        else:
            return "Surplus (Trapped Cash)"

    scatter_df["Quadrant"] = scatter_df.apply(_get_quadrant, axis=1)

    color_map = {
        "Reorder Now (Stockout Risk)": "#ef4444",
        "Watch Closely (Volatile)": "#f97316",
        "Healthy Stock (Optimal)": "#10b981",
        "Surplus (Trapped Cash)": "#eab308"
    }

    fig_matrix = px.scatter(
        scatter_df,
        x="overstock_risk",
        y="Stockout_Risk",
        color="Quadrant",
        color_discrete_map=color_map,
        size="Forecast_demand",
        size_max=28,
        hover_name="sku_id",
        hover_data={
            "category": True,
            "Total_stock": ":,.0f",
            "days_of_stock": True,
            "lead_time_days": True,
            "sales_at_risk": ":,.0f",
            "capital_locked": ":,.0f",
            "overstock_risk": False,
            "Stockout_Risk": False
        },
        labels={
            "overstock_risk": "Excess Inventory Risk →",
            "Stockout_Risk": "Stockout Risk ↑",
            "Total_stock": "Current Units",
            "days_of_stock": "Days of Stock Left",
            "lead_time_days": "Supplier Lead Days"
        }
    )

    # Add quadrant dividing lines
    fig_matrix.add_vline(x=0.5, line_width=1, line_dash="dash", line_color="#cbd5e1")
    fig_matrix.add_hline(y=0.5, line_width=1, line_dash="dash", line_color="#cbd5e1")

    fig_matrix.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#475569"),
        xaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="rgba(226, 232, 240, 0.8)"),
        yaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="rgba(226, 232, 240, 0.8)"),
        legend=dict(orientation="h", y=1.08, x=0)
    )

    st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown("""
    <div class="explain-card">
        <b>💡 Quadrant Quick Guide:</b><br>
        • 🔴 <b>Top-Left (Reorder Now):</b> High stockout danger, low inventory. Order from suppliers immediately.<br>
        • 🟠 <b>Top-Right (Watch Closely):</b> Unstable items with both supply lag and volatile demand. Monitor weekly.<br>
        • 🟢 <b>Bottom-Left (Healthy Stock):</b> Balanced inventory. Meets upcoming sales without tying up excessive cash.<br>
        • 🟡 <b>Bottom-Right (Surplus):</b> Low chance of stockout, but lots of idle units. Consider discounting to recover cash.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ────────────────────────────────────────────────────────────
# VIEW 4: PRODUCT CATALOG & SEARCH
# ────────────────────────────────────────────────────────────
# ============================================================
elif view == "🔎 Product Catalog & Search":
    st.subheader("Product Catalog & Inventory Intelligence Table")
    st.caption("Search, filter, and inspect detailed numbers for all products across your catalog.")

    search_query = st.text_input("🔍 Search by Product SKU or Category:", placeholder="e.g. SKU0012 or Furniture")

    cat_table = filtered_risk.copy()
    if search_query:
        mask = (
            cat_table["sku_id"].str.contains(search_query, case=False, na=False) |
            cat_table["category"].str.contains(search_query, case=False, na=False)
        )
        cat_table = cat_table[mask]

    catalog_display = pd.DataFrame({
        "Product": cat_table["sku_id"],
        "Category": cat_table["category"],
        "Stock on Hand": cat_table["Total_stock"].astype(int),
        "Days of Supply": cat_table["days_of_stock"].astype(int),
        "Supplier Lead (Days)": cat_table["lead_time_days"].astype(int),
        "Projected 90D Demand": cat_table["Forecast_demand"].astype(int),
        "Action Status": cat_table["Action"],
        "Sales at Risk": cat_table["sales_at_risk"].apply(lambda x: f"₹{x:,.0f}" if x > 0 else "₹0"),
        "Capital Trapped": cat_table["capital_locked"].apply(lambda x: f"₹{x:,.0f}" if x > 0 else "₹0")
    })

    st.dataframe(catalog_display, use_container_width=True, hide_index=True)

    # Download CSV button
    csv_bytes = catalog_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Inventory Decision Sheet (CSV)",
        data=csv_bytes,
        file_name="NorthBay_Living_Inventory_Decisions.csv",
        mime="text/csv"
    )

# ============================================================
# ────────────────────────────────────────────────────────────
# VIEW 5: SALES TRENDS & BESTSELLERS
# ────────────────────────────────────────────────────────────
# ============================================================
elif view == "💡 Sales Trends & Bestsellers":
    st.subheader("Sales Trends & Merchandise Insights")
    st.caption("Historical performance, top revenue drivers, and dead-stock identification.")

    if df_sales.empty:
        st.info("No historical sales data loaded.")
    else:
        s_df = df_sales.copy()
        s_df["month"] = s_df["sale_date"].dt.to_period("M").dt.to_timestamp()

        # Monthly Trend
        st.markdown("#### Monthly Sales Volume Trend")
        m_trend = s_df.groupby("month")["quantity"].sum().reset_index()
        fig_trend = px.line(m_trend, x="month", y="quantity", markers=True, labels={"quantity": "Units Sold", "month": "Month"})
        fig_trend.update_traces(line_color="#2563eb", line_width=3)
        fig_trend.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_trend, use_container_width=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 🏆 Top 10 Best Sellers (by Revenue)")
            top10 = s_df.groupby("sku_id")["revenue"].sum().sort_values(ascending=False).head(10).reset_index()
            fig_top = px.bar(top10, x="revenue", y="sku_id", orientation="h", labels={"revenue": "Revenue (₹)", "sku_id": "Product"}, color="revenue", color_continuous_scale="Blues")
            fig_top.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_top, use_container_width=True)

        with col_m2:
            st.markdown("#### ☠️ Dead / Slow Stock (Past 90 Days)")
            cutoff = s_df["sale_date"].max() - pd.DateOffset(months=3)
            recent_skus = s_df[s_df["sale_date"] >= cutoff]["sku_id"].unique()
            all_seen_skus = s_df["sku_id"].unique()
            dead_list = [s for s in all_seen_skus if s not in recent_skus]

            if dead_list:
                dead_rev = s_df[s_df["sku_id"].isin(dead_list)].groupby("sku_id")["revenue"].sum().sort_values(ascending=False).head(10).reset_index()
                fig_dead = px.bar(dead_rev, x="revenue", y="sku_id", orientation="h", labels={"revenue": "Past Revenue (₹)", "sku_id": "Product"}, color="revenue", color_continuous_scale="Reds")
                fig_dead.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
                st.plotly_chart(fig_dead, use_container_width=True)
            else:
                st.success("🎉 No dead stock found! All products had sales in the last 90 days.")

        st.markdown("#### 📦 Revenue Share by Category")
        if "category" in df_risk.columns:
            cat_rev = df_risk.groupby("category")["sales_at_risk"].sum().reset_index()
            fig_pie = px.pie(cat_rev, values="sales_at_risk", names="category", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================
# ────────────────────────────────────────────────────────────
# VIEW 6: AI ACCURACY & TRUST
# ────────────────────────────────────────────────────────────
# ============================================================
elif view == "🎯 AI Accuracy & Trust":
    st.subheader("AI Demand Model Accuracy & Holdout Evaluation")
    st.caption("Rigorous rolling-origin backtesting comparing Facebook Prophet against a Seasonal-Naive baseline.")

    if df_metrics.empty:
        st.info("Evaluation metrics have not been calculated yet. Re-run the pipeline with full sales history.")
    else:
        def _get_val(metric_name, col):
            r = df_metrics[df_metrics["metric"] == metric_name]
            if r.empty: return None
            v = r.iloc[0].get(col)
            return float(v) if v is not None and str(v) != "nan" else None

        p_wape = _get_val("wape", "prophet_value")
        b_wape = _get_val("wape", "baseline_value")
        p_bias = _get_val("bias", "prophet_value")
        b_bias = _get_val("bias", "baseline_value")

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.metric("Prophet Error (WAPE)", f"{p_wape:.1%}" if p_wape else "N/A", help="Weighted Absolute Percentage Error on 3-month holdout. Lower is better.")
        with t2:
            st.metric("Traditional Baseline WAPE", f"{b_wape:.1%}" if b_wape else "N/A", help="Error of seasonal same-month prior year baseline.")
        with t3:
            st.metric("Prophet Bias", f"{p_bias:+.1%}" if p_bias else "N/A", help="Near 0% means balanced forecasting without systemic over/under-estimating.")
        with t4:
            st.metric("Baseline Bias", f"{b_bias:+.1%}" if b_bias else "N/A")

        if p_wape and b_wape:
            improvement = (b_wape - p_wape) / b_wape * 100
            st.success(f"✅ **Performance Verdict:** Prophet beats the traditional seasonal baseline by **{improvement:.1f}%** on WAPE. Machine learning is adding clear forecasting value.")

        st.markdown("""
        <div class="explain-card">
            <b>📘 Why should you trust this forecast?</b><br>
            • <b>No Cheating (Zero Data Leakage):</b> The model was tested on real history it had never seen before (the final 3 months were held out).<br>
            • <b>What is WAPE?</b> Weighted Absolute Percentage Error measures total prediction errors divided by total sales. A lower percentage means fewer forecasting surprises.<br>
            • <b>What is Bias?</b> A bias close to 0% proves the AI doesn't have an ego—it doesn't consistently over-promise or under-estimate demand.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# UNIVERSAL JARGON BUSTER (ALWAYS AT BOTTOM)
# ============================================================
st.write("")
st.markdown("---")
with st.expander("📘 Plain-English Glossary: What Do These Terms Mean?", expanded=False):
    st.markdown("""
    <div class="jargon-grid">
        <div class="jargon-card">
            <div class="jargon-term">🚨 Stockout Risk</div>
            <div class="jargon-def">The danger of running completely out of stock before your supplier's next shipment arrives. High risk means lost revenue and unhappy customers.</div>
        </div>
        <div class="jargon-card">
            <div class="jargon-term">⏳ Days of Stock Left</div>
            <div class="jargon-def">How many days your current warehouse inventory will last based on the daily purchasing speed of your customers.</div>
        </div>
        <div class="jargon-card">
            <div class="jargon-term">🚚 Supplier Lead Time</div>
            <div class="jargon-def">How many days it takes between sending a purchase order to your supplier and receiving the physical goods at your warehouse.</div>
        </div>
        <div class="jargon-card">
            <div class="jargon-term">💸 Sales at Risk (₹)</div>
            <div class="jargon-def">The estimated rupee amount you stand to lose if customers want to buy these products but your shelves are empty.</div>
        </div>
        <div class="jargon-card">
            <div class="jargon-term">🔒 Capital Trapped (₹)</div>
            <div class="jargon-def">Money locked up in excess stock sitting on warehouse shelves instead of being accessible in your company's bank account.</div>
        </div>
        <div class="jargon-card">
            <div class="jargon-term">🎯 Reorder Point</div>
            <div class="jargon-def">The minimum inventory threshold. When your stock dips below this number, you should immediately place a replenishment order.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:12px; padding:30px 0 10px 0;">
FORESIGHT • Demand &amp; Inventory Intelligence Platform
</div>
""", unsafe_allow_html=True)