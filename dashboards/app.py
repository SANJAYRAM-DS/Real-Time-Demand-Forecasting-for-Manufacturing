import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="M5 Demand Forecasting Dashboard",
    layout="wide",
    page_icon="*"
)

st.title("M5 Demand Forecasting Dashboard")

# -----------------------------
# Dynamic Path Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
REPORT_PATH = os.path.join(BASE_DIR, "reports")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    """Safely load features and prediction data, handling missing files."""
    features_path = os.path.join(DATA_PATH, "m5_features.parquet")
    preds_path = os.path.join(REPORT_PATH, "predictions.parquet")

    if not os.path.exists(features_path):
        st.error(f"Missing file: {features_path}\n\nPlease run your pipeline first.")
        st.stop()

    hist = pd.read_parquet(features_path)[['id', 'date', 'sales', 'cat_id', 'store_id']]

    if os.path.exists(preds_path):
        pred = pd.read_parquet(preds_path)
        df = hist.merge(pred, on=['id', 'date'], how='left')
        st.success("Predictions loaded successfully.")
    else:
        st.warning("No predictions found — showing only actual sales.")
        df = hist.copy()
        df['yhat'] = None

    return df


# -----------------------------
# Load & Preprocess
# -----------------------------
df = load_data()
df['date'] = pd.to_datetime(df['date'])
latest_date = df['date'].max().date()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
cat = st.sidebar.selectbox("Category", sorted(df['cat_id'].unique()))
store = st.sidebar.selectbox("Store", sorted(df['store_id'].unique()))
sku_list = sorted(df[df['cat_id'] == cat]['id'].unique())
sku = st.sidebar.selectbox("SKU", sku_list)

filtered = df[(df['cat_id'] == cat) & (df['store_id'] == store) & (df['id'] == sku)]

# -----------------------------
# Metrics Display
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    st.metric("Latest Date", str(latest_date))
with col2:
    mae = abs(filtered['sales_x'] - filtered['yhat']).mean() if 'yhat' in filtered else None
    st.metric("MAE (SKU)", f"{mae:.2f}" if mae is not None else "N/A")

# -----------------------------
# SKU-level Time Series Plot
# -----------------------------
st.subheader(f"Actual vs Predicted — {sku}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(filtered['date'], filtered['sales_x'], label='Actual', color='tab:blue', linewidth=2)
if filtered['yhat'].notna().any():
    ax.plot(filtered['date'], filtered['yhat'], label='Predicted', color='tab:orange', linewidth=2)
ax.set_title(f"Demand Trend — {sku}")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
st.pyplot(fig)

# -----------------------------
# Category-level Performance
# -----------------------------
st.divider()
st.subheader("Category-level Forecast Performance")

if 'yhat' in df.columns and df['yhat'].notna().any():
    err_cat = df.groupby('cat_id').apply(lambda g: abs(g['sales_x'] - g['yhat']).mean()).sort_values()
    st.bar_chart(err_cat)
else:
    st.info("ℹ Run the prediction pipeline to view category-level errors.")

# -----------------------------
# Overall Summary / Metadata
# -----------------------------
st.divider()
st.markdown(f"""
### Data Summary
- **Total records:** {len(df):,}
- **Unique SKUs:** {df['id'].nunique():,}
- **Date range:** {df['date'].min().date()} → {df['date'].max().date()}


""")

st.caption("Developed for the M5 Demand Forecasting project — interactive and auto-refresh ready.")
