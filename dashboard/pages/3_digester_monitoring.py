"""
Page 3: Digester Monitoring
============================
Harmonised State Telemetry, Biochemical Stability Indicators & Sensor Bands
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up project root in sys.path
ROOT = Path(__file__).resolve().parent
for parent in [ROOT] + list(ROOT.parents):
    if (parent / "src").is_dir() and (parent / "data").is_dir():
        ROOT = parent
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    st.set_page_config(page_title="Digester Monitoring", page_icon="📈", layout="wide")
except Exception:
    pass

st.title("📈 Harmonised Digester Condition Monitoring")
st.caption("Standardised SI-unit biochemical telemetry, stability indicators, and sensor-health tracking.")

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
SYSTEMS = {
    "Pilot Digester 1 (PD1)": ROOT / "data" / "processed" / "pilot_D1_harmonised.csv",
    "Pilot Digester 2 (PD2)": ROOT / "data" / "processed" / "pilot_D2_harmonised.csv",
    "Full-Scale Digester": ROOT / "data" / "processed" / "full_harmonised.csv",
}

@st.cache_data
def load_harmonised(system_name: str) -> pd.DataFrame:
    path = SYSTEMS.get(system_name)
    if not path or not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)

# -----------------------------------------------------------------------------
# Controls
# -----------------------------------------------------------------------------
col_sys, col_meta = st.columns([1, 2])
with col_sys:
    selected_system = st.selectbox(
        "Select Anaerobic Digester System:",
        options=list(SYSTEMS.keys()),
        index=0
    )

try:
    df = load_harmonised(selected_system)
except Exception as e:
    st.warning(f"Unable to load harmonised dataset for '{selected_system}': {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Current State Summary Box
# -----------------------------------------------------------------------------
def get_latest_valid(series):
    valid = series.dropna()
    return valid.iloc[-1] if not valid.empty else np.nan

latest_dt = df["datetime"].dropna().iloc[-1]
latest_ph = get_latest_valid(df["ph"]) if "ph" in df.columns else np.nan
latest_vfa = get_latest_valid(df["vfa_mgl"]) if "vfa_mgl" in df.columns else np.nan
latest_alk = get_latest_valid(df["alk_mgl"]) if "alk_mgl" in df.columns else np.nan
latest_va_alk = get_latest_valid(df["va_alk"]) if "va_alk" in df.columns else np.nan
latest_biogas = get_latest_valid(df["biogas_m3d"]) if "biogas_m3d" in df.columns else np.nan
latest_ccr = get_latest_valid(df["ccr"]) if "ccr" in df.columns else np.nan

is_warning = (latest_va_alk > 0.30) or (latest_vfa > 500) or (latest_ph < 6.8)

st.markdown("### 🎛️ Most Recent Operating Telemetry")
if is_warning:
    st.error(f"🚨 **STATUS: OPERATIONAL WARNING / ELEVATED RATIO** on {latest_dt.strftime('%Y-%m-%d')}")
else:
    st.success(f"✅ **STATUS: NORMAL / STABLE OPERATION** on {latest_dt.strftime('%Y-%m-%d')}")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("pH", f"{latest_ph:.2f}" if not np.isnan(latest_ph) else "N/A", delta="Normal: 6.8-7.6")
m2.metric("VFA (mg/L)", f"{latest_vfa:.1f}" if not np.isnan(latest_vfa) else "N/A", delta="Warn > 500")
m3.metric("Alkalinity (mg/L)", f"{latest_alk:.0f}" if not np.isnan(latest_alk) else "N/A", delta="Safe > 1000")
m4.metric("VA/Alk Ratio", f"{latest_va_alk:.3f}" if not np.isnan(latest_va_alk) else "N/A", delta="Warn > 0.30")
m5.metric("Biogas (m³/day)", f"{latest_biogas:.1f}" if not np.isnan(latest_biogas) else "N/A")
m6.metric("CCR Sensor", f"{latest_ccr:.1f}" if not np.isnan(latest_ccr) else "N/A")

st.markdown("---")

# -----------------------------------------------------------------------------
# Stacked Subplots: Biochemical Indicators
# -----------------------------------------------------------------------------
st.subheader("🧪 Biochemical Stability Telemetry (Stacked Profile)")
st.caption("Synchronized temporal progression of pH buffer equilibrium, acid accumulation, and stability ratios.")

fig_stack = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=[
        "1. Digester pH (Safe Buffer Range: 6.8 - 7.6)",
        "2. Volatile Fatty Acids (VFA, mg/L as Acetic Acid)",
        "3. Total Alkalinity (mg/L as CaCO3)",
        "4. VA/Alk Stability Ratio (Threshold: 0.30 Early Warning)"
    ]
)

# Row 1: pH
if "ph" in df.columns:
    fig_stack.add_trace(go.Scatter(
        x=df["datetime"], y=df["ph"],
        mode="lines+markers", name="pH",
        line=dict(color="#1f77b4", width=2), marker=dict(size=4)
    ), row=1, col=1)
    # Target band
    fig_stack.add_hrect(
        y0=6.8, y1=7.6, fillcolor="rgba(44, 160, 44, 0.15)",
        line_width=0, row=1, col=1
    )

# Row 2: VFA
if "vfa_mgl" in df.columns:
    fig_stack.add_trace(go.Scatter(
        x=df["datetime"], y=df["vfa_mgl"],
        mode="lines+markers", name="VFA (mg/L)",
        line=dict(color="#ff7f0e", width=2), marker=dict(size=4)
    ), row=2, col=1)
    fig_stack.add_hline(
        y=500.0, line_dash="dash", line_color="red",
        annotation_text="High VFA Alert (500 mg/L)", row=2, col=1
    )

# Row 3: Alkalinity
if "alk_mgl" in df.columns:
    fig_stack.add_trace(go.Scatter(
        x=df["datetime"], y=df["alk_mgl"],
        mode="lines+markers", name="Alkalinity (mg/L)",
        line=dict(color="#2ca02c", width=2), marker=dict(size=4)
    ), row=3, col=1)
    fig_stack.add_hline(
        y=1000.0, line_dash="dash", line_color="orange",
        annotation_text="Min Buffer (1000 mg/L)", row=3, col=1
    )

# Row 4: VA/Alk Ratio
if "va_alk" in df.columns:
    fig_stack.add_trace(go.Scatter(
        x=df["datetime"], y=df["va_alk"],
        mode="lines+markers", name="VA/Alk Ratio",
        line=dict(color="#d62728", width=2), marker=dict(size=5)
    ), row=4, col=1)
    # Threshold line
    fig_stack.add_hline(
        y=0.30, line_dash="solid", line_color="crimson", line_width=2,
        annotation_text="Early Warning Threshold (0.30)", row=4, col=1
    )
    # Shaded bands
    fig_stack.add_hrect(y0=0.0, y1=0.23, fillcolor="rgba(31, 119, 180, 0.10)", line_width=0, row=4, col=1)
    fig_stack.add_hrect(y0=0.23, y1=0.30, fillcolor="rgba(44, 160, 44, 0.15)", line_width=0, row=4, col=1)
    fig_stack.add_hrect(y0=0.30, y1=0.40, fillcolor="rgba(255, 127, 14, 0.20)", line_width=0, row=4, col=1)
    fig_stack.add_hrect(y0=0.40, y1=1.50, fillcolor="rgba(214, 39, 40, 0.25)", line_width=0, row=4, col=1)

fig_stack.update_layout(
    height=850,
    showlegend=False,
    template="plotly_white",
    margin=dict(l=50, r=30, t=50, b=40),
    hovermode="x unified"
)
fig_stack.update_yaxes(title_text="pH [su]", row=1, col=1)
fig_stack.update_yaxes(title_text="VFA [mg/L]", row=2, col=1)
fig_stack.update_yaxes(title_text="Alk [mg/L]", row=3, col=1)
fig_stack.update_yaxes(title_text="VA / Alk [-]", row=4, col=1)
fig_stack.update_xaxes(title_text="Observation Date", row=4, col=1)

st.plotly_chart(fig_stack, use_container_width=True)

# -----------------------------------------------------------------------------
# Biogas & CCR Trends
# -----------------------------------------------------------------------------
st.subheader("⚡ Biogas Production & Sensor Dynamics")
c_bio, c_ccr = st.columns(2)

with c_bio:
    if "biogas_m3d" in df.columns:
        fig_bio = go.Figure()
        # Raw points
        fig_bio.add_trace(go.Scatter(
            x=df["datetime"], y=df["biogas_m3d"],
            mode="lines+markers", name="Daily Biogas (m³/d)",
            line=dict(color="#9467bd", width=1.5), marker=dict(size=3)
        ))
        # 7-day rolling mean
        roll_bio = df["biogas_m3d"].rolling(window=7, min_periods=3).mean()
        fig_bio.add_trace(go.Scatter(
            x=df["datetime"], y=roll_bio,
            mode="lines", name="7-Day Rolling Mean",
            line=dict(color="#17becf", width=3)
        ))
        # Highlight zero-gas sensor faults if flagged
        if "flag_gas_zero" in df.columns:
            zero_gas = df[df["flag_gas_zero"] == 1]
            if not zero_gas.empty:
                fig_bio.add_trace(go.Scatter(
                    x=zero_gas["datetime"], y=zero_gas["biogas_m3d"],
                    mode="markers", name="Meter Reset / Zero Fault",
                    marker=dict(symbol="x", size=9, color="red")
                ))

        fig_bio.update_layout(
            title="Biogas Generation Profile (m³/day)",
            xaxis_title="Date",
            yaxis_title="Biogas Yield [m³/day]",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bio, use_container_width=True)

with c_ccr:
    if "ccr" in df.columns:
        fig_ccr = go.Figure()
        fig_ccr.add_trace(go.Scatter(
            x=df["datetime"], y=df["ccr"],
            mode="lines+markers", name="Digester CCR",
            line=dict(color="#8c564b", width=1.5), marker=dict(size=3)
        ))
        if "ccr_influent" in df.columns:
            fig_ccr.add_trace(go.Scatter(
                x=df["datetime"], y=df["ccr_influent"],
                mode="lines", name="Influent CCR",
                line=dict(color="#e377c2", width=1.5, dash="dot")
            ))
        elif "ccr_slurry" in df.columns:
            fig_ccr.add_trace(go.Scatter(
                x=df["datetime"], y=df["ccr_slurry"],
                mode="lines", name="Slurry CCR",
                line=dict(color="#e377c2", width=1.5, dash="dot")
            ))

        fig_ccr.update_layout(
            title="Continuous Sensor Telemetry: CCR",
            xaxis_title="Date",
            yaxis_title="CCR Sensor Signal",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ccr, use_container_width=True)

# -----------------------------------------------------------------------------
# Data Quality & Domain Notes
# -----------------------------------------------------------------------------
st.info("""
**Data Quality & Preprocessing Notice:**
- In Pilot D1, alkalinity measurements from Oct 17 to Dec 15, 2019 were reported at ~85 mg/L while pH was ~7.6 and gas production was healthy. This was identified in Phase 1 as an instrument or unit conversion artifact and has been masked (`flag_alk_suspect = 1`) to preserve scientific validity.
- The Full-Scale Digester operates under heavy bicarbonate buffering; its VA/Alk ratio remains below 0.03 at all times, demonstrating exceptional stability without organic souring.
""")
