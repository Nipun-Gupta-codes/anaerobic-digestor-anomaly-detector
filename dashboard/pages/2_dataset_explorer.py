"""
Page 2: Dataset Explorer
=========================
Exploratory Data Analysis (EDA) for Real Anaerobic Digestion Datasets
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set up project root in sys.path
ROOT = Path(__file__).resolve().parent
for parent in [ROOT] + list(ROOT.parents):
    if (parent / "src").is_dir() and (parent / "data").is_dir():
        ROOT = parent
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    st.set_page_config(page_title="Dataset Explorer", page_icon="🔍", layout="wide")
except Exception:
    pass

st.title("🔍 Raw Dataset Explorer & Sensor Telemetry")
st.caption("Interactive inspection of raw industrial and pilot sensor time series, data quality, and distributions.")

# -----------------------------------------------------------------------------
# Data Loading Functions
# -----------------------------------------------------------------------------
DATA_FILES = {
    "Pilot Digesters (D1 & D2 Merged)": ROOT / "data" / "raw" / "sensor_ad_pilot_merged.csv",
    "Full-Scale Digester": ROOT / "data" / "raw" / "sensor_ad_full_scale_digester_ccr.csv",
    "Full-Scale WWC Feed Stream": ROOT / "data" / "raw" / "sensor_ad_full_scale_wwc_ccr.csv",
}

@st.cache_data
def load_raw_dataset(dataset_name: str) -> pd.DataFrame:
    file_path = DATA_FILES.get(dataset_name)
    if not file_path or not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    # Standardize unnamed index column if present
    if "Unnamed: 0" in df.columns or "" in df.columns:
        col_to_drop = "Unnamed: 0" if "Unnamed: 0" in df.columns else ""
        df = df.drop(columns=[col_to_drop])
    
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
    return df

# -----------------------------------------------------------------------------
# Sidebar & Top Controls
# -----------------------------------------------------------------------------
col_sel, col_info = st.columns([1, 2])
with col_sel:
    dataset_name = st.selectbox(
        "Select Dataset to Inspect:",
        options=list(DATA_FILES.keys()),
        index=0
    )

try:
    df = load_raw_dataset(dataset_name)
    with col_info:
        st.success(
            f"**Loaded Successfully**: {dataset_name} | "
            f"**Rows**: {df.shape[0]} | **Columns**: {df.shape[1]} | "
            f"**Span**: {df['datetime'].dt.strftime('%Y-%m-%d').iloc[0]} to {df['datetime'].dt.strftime('%Y-%m-%d').iloc[-1]}"
        )
except Exception as e:
    st.warning(f"Unable to load raw dataset '{dataset_name}': {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Tabs Layout
# -----------------------------------------------------------------------------
tab_ts, tab_table, tab_stats, tab_missing, tab_corr = st.tabs([
    "📈 Time Series Plotter",
    "📄 Raw Data Table",
    "📊 Descriptive Statistics",
    "❓ Missing Values Analysis",
    "🔥 Correlation Heatmap"
])

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# -----------------------------------------------------------------------------
# TAB 1: Time Series Plotter
# -----------------------------------------------------------------------------
with tab_ts:
    st.subheader("Interactive Sensor & Lab Chemistry Time Series")
    st.caption("Examine temporal trends, seasonal fluctuations, sensor drift, and operational shock-loads.")

    default_var = None
    candidates = [
        "PD1_gas_generation_cfday", "biogas_produced_avg_cfday",
        "PD1_ph_su", "ph_su", "daily_ccr_digester", "daily_ccr_AD1"
    ]
    for c in candidates:
        if c in numeric_cols:
            default_var = c
            break
    if not default_var and numeric_cols:
        default_var = numeric_cols[0]

    ts_col1, ts_col2 = st.columns(2)
    with ts_col1:
        primary_var = st.selectbox(
            "Primary Variable (Left Y-Axis):",
            options=numeric_cols,
            index=numeric_cols.index(default_var) if default_var in numeric_cols else 0,
            key="primary_ts_var"
        )
    with ts_col2:
        secondary_options = ["None"] + [c for c in numeric_cols if c != primary_var]
        secondary_var = st.selectbox(
            "Secondary Variable (Right Y-Axis, Optional):",
            options=secondary_options,
            index=0,
            key="secondary_ts_var"
        )

    # Build Plotly Time Series Figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df[primary_var],
        mode="lines+markers",
        name=primary_var,
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=4),
        yaxis="y1"
    ))

    if secondary_var != "None":
        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=df[secondary_var],
            mode="lines+markers",
            name=secondary_var,
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(size=4),
            yaxis="y2"
        ))
        fig.update_layout(
            yaxis2=dict(
                title=f"{secondary_var}",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

    fig.update_layout(
        title=f"Time Series: {primary_var}" + (f" vs {secondary_var}" if secondary_var != "None" else ""),
        xaxis=dict(title="Date / Time", rangeslider=dict(visible=True)),
        yaxis=dict(title=f"{primary_var}"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: Raw Data Table
# -----------------------------------------------------------------------------
with tab_table:
    st.subheader("Data Viewer & Column Filter")
    cols_to_show = st.multiselect(
        "Select Columns to Display:",
        options=df.columns.tolist(),
        default=df.columns.tolist()[:8]
    )
    st.dataframe(df[cols_to_show] if cols_to_show else df, use_container_width=True)
    
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name=f"{dataset_name.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# TAB 3: Descriptive Statistics
# -----------------------------------------------------------------------------
with tab_stats:
    st.subheader("Statistical Summary of Process Telemetry")
    if numeric_cols:
        desc_df = df[numeric_cols].describe().T
        desc_df["missing_count"] = df[numeric_cols].isnull().sum()
        desc_df["missing_pct"] = (desc_df["missing_count"] / len(df) * 100).round(2)
        desc_df["skewness"] = df[numeric_cols].skew().round(3)
        st.dataframe(desc_df.style.format("{:.2f}", subset=["mean", "std", "min", "25%", "50%", "75%", "max", "skewness"]), use_container_width=True)

        st.markdown("#### Variable Distribution Inspector")
        dist_var = st.selectbox("Select Variable for Histogram & Box Plot:", options=numeric_cols, key="dist_var_select")
        col_hist, col_box = st.columns(2)
        with col_hist:
            fig_hist = px.histogram(
                df, x=dist_var, nbins=30,
                marginal="rug",
                title=f"Histogram: {dist_var}",
                template="plotly_white",
                color_discrete_sequence=["#2ca02c"]
            )
            fig_hist.update_layout(xaxis_title=dist_var, yaxis_title="Observation Count")
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_box:
            fig_box = px.box(
                df, y=dist_var,
                points="all",
                title=f"Boxplot & Outliers: {dist_var}",
                template="plotly_white",
                color_discrete_sequence=["#d62728"]
            )
            fig_box.update_layout(yaxis_title=dist_var)
            st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: Missing Values Analysis
# -----------------------------------------------------------------------------
with tab_missing:
    st.subheader("Missing Value Frequency & Sparsity Analysis")
    st.markdown("""
    Anaerobic digestion facilities often employ daily automatic sensors (pH, gas, temperature) 
    alongside intermittent laboratory titration tests (VFA, Alkalinity, COD) executed only every 2 to 3 days.
    """)
    missing_series = df.isnull().mean() * 100
    missing_counts = df.isnull().sum()
    missing_df = pd.DataFrame({
        "Variable": missing_series.index,
        "Missing_Pct": missing_series.values,
        "Missing_Count": missing_counts.values
    }).sort_values(by="Missing_Pct", ascending=False)

    top_missing = missing_df[missing_df["Missing_Count"] > 0]
    if not top_missing.empty:
        fig_miss = px.bar(
            top_missing,
            x="Variable",
            y="Missing_Pct",
            text="Missing_Pct",
            title=f"Missing Telemetry Rates in {dataset_name} (%)",
            color="Missing_Pct",
            color_continuous_scale="Reds",
            template="plotly_white"
        )
        fig_miss.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_miss.update_layout(
            xaxis=dict(tickangle=-45, title="Process Variable"),
            yaxis=dict(title="Percentage Missing (%)", range=[0, 110]),
            margin=dict(b=100)
        )
        st.plotly_chart(fig_miss, use_container_width=True)
    else:
        st.success("No missing values identified in this dataset!")

# -----------------------------------------------------------------------------
# TAB 5: Correlation Heatmap
# -----------------------------------------------------------------------------
with tab_corr:
    st.subheader("Sensor Cross-Correlation Heatmap")
    st.caption("Pearson correlation coefficients across continuous variables (filtered for >= 30% valid observations).")

    valid_cols = [c for c in numeric_cols if df[c].notna().mean() >= 0.30]
    if len(valid_cols) > 1:
        corr_matrix = df[valid_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(x="Variable", y="Variable", color="Pearson r"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=f"Cross-Correlation Matrix ({len(valid_cols)} Variables)",
            template="plotly_white"
        )
        fig_corr.update_layout(
            width=900,
            height=800,
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Insufficient populated numeric columns to compute correlation matrix.")
