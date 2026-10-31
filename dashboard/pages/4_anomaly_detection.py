"""
Page 4: Anomaly Detection
==========================
Unsupervised Novelty Detection & Chemical Evidence Cross-Validation
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
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
    st.set_page_config(page_title="Anomaly Detection", page_icon="⚠️", layout="wide")
except Exception:
    pass

st.title("⚠️ Unsupervised Anomaly & Novelty Detection")
st.caption("Novelty detection on real operational plant telemetry using Isolation Forest, LOF, and One-Class SVM.")

# -----------------------------------------------------------------------------
# Data Mapping
# -----------------------------------------------------------------------------
SCORE_FILES = {
    ("Pilot D1", "A: Logged Telemetry (pH, Temp, Gas, CCR)"): ROOT / "data" / "processed" / "anomaly_scores_pilot_D1_A_logged.csv",
    ("Pilot D1", "B: Full Chemistry (A + VFA, Alk, VA/Alk)"): ROOT / "data" / "processed" / "anomaly_scores_pilot_D1_B_chemistry.csv",
    ("Pilot D2", "A: Logged Telemetry (pH, Temp, Gas, CCR)"): ROOT / "data" / "processed" / "anomaly_scores_pilot_D2_A_logged.csv",
    ("Pilot D2", "B: Full Chemistry (A + VFA, Alk, VA/Alk)"): ROOT / "data" / "processed" / "anomaly_scores_pilot_D2_B_chemistry.csv",
    ("Full-Scale", "A: Logged Telemetry (pH, Temp, Gas, CCR)"): ROOT / "data" / "processed" / "anomaly_scores_full_scale_A_logged.csv",
    ("Full-Scale", "B: Full Chemistry (A + VFA, Alk, VA/Alk)"): ROOT / "data" / "processed" / "anomaly_scores_full_scale_B_chemistry.csv",
}

HARMONISED_FILES = {
    "Pilot D1": ROOT / "data" / "processed" / "pilot_D1_harmonised.csv",
    "Pilot D2": ROOT / "data" / "processed" / "pilot_D2_harmonised.csv",
    "Full-Scale": ROOT / "data" / "processed" / "full_harmonised.csv",
}

@st.cache_data
def load_anomaly_scores(sys_name: str, feat_set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_path = SCORE_FILES.get((sys_name, feat_set))
    if not score_path or not score_path.exists():
        raise FileNotFoundError(f"Score file not found: {score_path}")
    scores_df = pd.read_csv(score_path, parse_dates=["datetime"])
    
    harm_path = HARMONISED_FILES.get(sys_name)
    harm_df = pd.DataFrame()
    if harm_path and harm_path.exists():
        harm_df = pd.read_csv(harm_path, parse_dates=["datetime"])
    
    return scores_df.sort_values("datetime").reset_index(drop=True), harm_df

@st.cache_data
def load_benchmark_metrics() -> pd.DataFrame:
    metrics_path = ROOT / "reports" / "anomaly_detection_results.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return pd.DataFrame()

# -----------------------------------------------------------------------------
# Controls
# -----------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    selected_system = st.selectbox(
        "Digester System:",
        options=["Pilot D1", "Pilot D2", "Full-Scale"],
        index=0
    )
with c2:
    selected_features = st.selectbox(
        "Feature Configuration:",
        options=[
            "A: Logged Telemetry (pH, Temp, Gas, CCR)",
            "B: Full Chemistry (A + VFA, Alk, VA/Alk)"
        ],
        index=0
    )
with c3:
    selected_model = st.selectbox(
        "Detector Algorithm:",
        options=["IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
        index=0
    )

try:
    scores_df, harm_df = load_anomaly_scores(selected_system, selected_features)
except Exception as e:
    st.warning(f"Error loading anomaly scores: {e}")
    st.stop()

# Merge for richer inspection
merged_df = scores_df.copy()
if not harm_df.empty:
    cols_to_merge = ["datetime"] + [c for c in ["ph", "vfa_mgl", "alk_mgl", "va_alk", "biogas_m3d", "ccr"] if c in harm_df.columns]
    merged_df = pd.merge(merged_df, harm_df[cols_to_merge], on="datetime", how="left")

score_col = f"{selected_model}_score"
flag_col = f"{selected_model}_flag"

# -----------------------------------------------------------------------------
# Overview KPI Cards
# -----------------------------------------------------------------------------
n_total = len(scores_df)
n_flagged = int(scores_df[flag_col].sum()) if flag_col in scores_df.columns else 0
flag_pct = (n_flagged / n_total * 100) if n_total > 0 else 0
n_evidence = int(scores_df["evidence"].sum()) if "evidence" in scores_df.columns else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Evaluated Days", n_total)
kpi2.metric("Flagged Outliers", f"{n_flagged} ({flag_pct:.1f}%)")
kpi3.metric("Chemical Instability Days", n_evidence)
kpi4.metric("Novelty Cutoff", str(scores_df["train_cutoff"].iloc[0])[:10] if "train_cutoff" in scores_df.columns else "N/A")

# -----------------------------------------------------------------------------
# Time Series Plot of Anomaly Scores
# -----------------------------------------------------------------------------
st.subheader(f"📊 {selected_model} Anomaly Score Profile")
st.caption("Higher score denotes greater statistical novelty. Red diamonds indicate algorithm-flagged outliers; gold stars indicate chemical instability evidence.")

fig_score = go.Figure()

# Split inliers and outliers
inliers = merged_df[merged_df[flag_col] == 0]
outliers = merged_df[merged_df[flag_col] == 1]

# Normal Inliers
fig_score.add_trace(go.Scatter(
    x=inliers["datetime"], y=inliers[score_col],
    mode="markers",
    name="Normal Operation (Inlier)",
    marker=dict(color="#1f77b4", size=6, opacity=0.7)
))

# Flagged Outliers
fig_score.add_trace(go.Scatter(
    x=outliers["datetime"], y=outliers[score_col],
    mode="markers",
    name="Flagged Anomaly",
    marker=dict(color="#d62728", symbol="diamond", size=9, line=dict(width=1, color="black"))
))

# Chemical Evidence Overlay
if "evidence" in merged_df.columns:
    chem_ev = merged_df[merged_df["evidence"] == 1.0]
    if not chem_ev.empty:
        fig_score.add_trace(go.Scatter(
            x=chem_ev["datetime"], y=chem_ev[score_col],
            mode="markers",
            name="Chemical Evidence (VA/Alk>0.3 or VFA>500)",
            marker=dict(symbol="star", size=11, color="#ff7f0e", line=dict(width=1, color="darkred"))
        ))

# Training cutoff vertical line
if "train_cutoff" in merged_df.columns and pd.notna(merged_df["train_cutoff"].iloc[0]):
    cutoff_date = pd.to_datetime(merged_df["train_cutoff"].iloc[0])
    fig_score.add_vline(
        x=cutoff_date.timestamp() * 1000,
        line_dash="dash", line_color="gray", line_width=2,
        annotation_text="Novelty Train / Test Cutoff",
        annotation_position="top left"
    )

fig_score.update_layout(
    title=f"{selected_model} Score Progression ({selected_system} - {selected_features[:1]} Set)",
    xaxis_title="Observation Date",
    yaxis_title=f"{selected_model} Anomaly Score",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="closest"
)
st.plotly_chart(fig_score, use_container_width=True)

# -----------------------------------------------------------------------------
# Algorithmic Benchmark Metrics Table
# -----------------------------------------------------------------------------
st.subheader("🎯 Quantitative Evaluation Against Chemical Instability Proxy")
st.caption("Performance metrics on holdout test partition evaluated against heuristic chemical ground-truth proxy.")

metrics_df = load_benchmark_metrics()
if not metrics_df.empty:
    sys_filter = "pilot_D1" if "D1" in selected_system else ("pilot_D2" if "D2" in selected_system else "full_scale")
    feat_filter = "A_logged" if "A:" in selected_features else "B_chemistry"
    
    sub_metrics = metrics_df[
        (metrics_df["system"].str.contains(sys_filter)) &
        (metrics_df["system"].str.contains(feat_filter)) &
        (metrics_df["part"] == "holdout")
    ]
    if not sub_metrics.empty:
        display_m = sub_metrics[["detector", "n_rows", "n_flagged", "n_evidence_days", "auc", "precision", "recall"]].copy()
        display_m = display_m.rename(columns={
            "detector": "Algorithm",
            "n_rows": "Holdout Rows",
            "n_flagged": "Flagged Anomalies",
            "n_evidence_days": "Evidence Days",
            "auc": "ROC-AUC Score",
            "precision": "Precision",
            "recall": "Recall"
        })
        st.dataframe(display_m.style.format({
            "ROC-AUC Score": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}"
        }), use_container_width=True, hide_index=True)
    else:
        st.dataframe(metrics_df.head(10), use_container_width=True)
else:
    st.info("No precomputed metrics found in reports/anomaly_detection_results.csv.")

# -----------------------------------------------------------------------------
# Flagged Days Context Table
# -----------------------------------------------------------------------------
st.subheader("📋 Flagged Anomaly Events Table")
st.caption("Detailed biochemical context on days flagged by the selected detector.")

flagged_rows = merged_df[merged_df[flag_col] == 1].copy()
if not flagged_rows.empty:
    cols_display = ["datetime", score_col, "evidence"] + [c for c in ["ph", "vfa_mgl", "alk_mgl", "va_alk", "biogas_m3d", "ccr"] if c in flagged_rows.columns]
    flagged_rows = flagged_rows[cols_display].sort_values(by=score_col, ascending=False).reset_index(drop=True)
    st.dataframe(flagged_rows.style.format({
        score_col: "{:.3f}",
        "evidence": "{:.0f}",
        "ph": "{:.2f}",
        "vfa_mgl": "{:.1f}",
        "alk_mgl": "{:.0f}",
        "va_alk": "{:.3f}",
        "biogas_m3d": "{:.1f}",
        "ccr": "{:.1f}"
    }), use_container_width=True)
else:
    st.info("No days were flagged by this detector under the current threshold settings.")

# -----------------------------------------------------------------------------
# Scientific Warning
# -----------------------------------------------------------------------------
st.warning("""
⚠️ **Academic Honesty & Methodological Caution**:
Unsupervised novelty algorithms detect data points residing in sparse regions of the multidimensional sensor space. 
An anomaly flag represents a statistical departure from the training baseline; it is **NOT confirmed biological souring**. 
In full-scale facilities, flagged days frequently correspond to feed composition shifts or pump scheduling variations rather than biological inhibition.
""")
