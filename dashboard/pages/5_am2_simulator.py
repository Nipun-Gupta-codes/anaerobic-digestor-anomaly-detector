"""
Page 5: AM2 Kinetic Simulator
==============================
Mechanistic Simulation of Two-Step Acidogenesis-Methanogenesis Kinetics
Bernard et al. (2001) Dynamical Model
"""
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
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
    st.set_page_config(page_title="AM2 Kinetic Simulator", page_icon="⚙️", layout="wide")
except Exception:
    pass

from src.am2_model import solve_am2, DEFAULT_PARAMS, mu1, mu2

st.title("⚙️ AM2 Kinetic Simulator (Bernard et al., 2001)")
st.caption("First-principles two-step mathematical simulation of acidogenesis, methanogenesis, and Haldane substrate inhibition.")

# -----------------------------------------------------------------------------
# Scenario Presets via Session State
# -----------------------------------------------------------------------------
PRESETS = {
    "Normal Operation": {
        "S1_0": 2.0, "X1_0": 0.5, "S2_0": 2.0, "X2_0": 0.5,
        "D": 0.50, "S1_in": 10.0, "alk_in": 25.0, "duration": 50
    },
    "Organic Overloading": {
        "S1_0": 2.0, "X1_0": 0.5, "S2_0": 2.0, "X2_0": 0.5,
        "D": 0.60, "S1_in": 28.0, "alk_in": 25.0, "duration": 60
    },
    "Severe Souring / Washout": {
        "S1_0": 2.0, "X1_0": 0.4, "S2_0": 8.0, "X2_0": 0.3,
        "D": 0.85, "S1_in": 45.0, "alk_in": 12.0, "duration": 70
    }
}

for k in ["S1_0", "X1_0", "S2_0", "X2_0", "D", "S1_in", "alk_in", "duration"]:
    if k not in st.session_state:
        st.session_state[k] = PRESETS["Normal Operation"][k]

def apply_preset(preset_name: str):
    p = PRESETS[preset_name]
    for k, v in p.items():
        st.session_state[k] = v

st.subheader("📋 Scenario Presets")
pcol1, pcol2, pcol3 = st.columns(3)
with pcol1:
    if st.button("🟢 Preset: Normal Steady State", use_container_width=True):
        apply_preset("Normal Operation")
with pcol2:
    if st.button("🟡 Preset: Organic Overload", use_container_width=True):
        apply_preset("Organic Overloading")
with pcol3:
    if st.button("🔴 Preset: Severe Souring", use_container_width=True):
        apply_preset("Severe Souring / Washout")

st.markdown("---")

# -----------------------------------------------------------------------------
# Interactive Sliders & Parameters
# -----------------------------------------------------------------------------
st.subheader("🎛️ Simulation Controls & Operating Parameters")

c_in1, c_in2, c_in3, c_in4 = st.columns(4)
with c_in1:
    s1_0 = st.slider("Initial Substrate S1(0) [g COD/L]:", 0.1, 20.0, float(st.session_state["S1_0"]), 0.5, key="S1_0")
    x1_0 = st.slider("Initial Acidogens X1(0) [g VS/L]:", 0.05, 3.0, float(st.session_state["X1_0"]), 0.05, key="X1_0")
with c_in2:
    s2_0 = st.slider("Initial VFA S2(0) [mmol/L]:", 0.1, 50.0, float(st.session_state["S2_0"]), 0.5, key="S2_0")
    x2_0 = st.slider("Initial Methanogens X2(0) [g VS/L]:", 0.05, 3.0, float(st.session_state["X2_0"]), 0.05, key="X2_0")
with c_in3:
    D_val = st.slider("Dilution Rate D [1/day]:", 0.05, 1.5, float(st.session_state["D"]), 0.05, key="D")
    s1_in = st.slider("Influent Substrate S1_in [g COD/L]:", 1.0, 60.0, float(st.session_state["S1_in"]), 1.0, key="S1_in")
with c_in4:
    alk_in = st.slider("Influent Buffer Alk_in [mmol/L]:", 5.0, 50.0, float(st.session_state["alk_in"]), 1.0, key="alk_in")
    duration = st.slider("Simulation Span [days]:", 10, 150, int(st.session_state["duration"]), 5, key="duration")

with st.expander("🔬 Advanced Kinetic Parameters (Bernard et al., 2001 Defaults)"):
    kcol1, kcol2, kcol3 = st.columns(3)
    with kcol1:
        st.write(f"• $\\mu_{{1,max}}$ = {DEFAULT_PARAMS['mu1_max']} 1/d (Max acidogen growth)")
        st.write(f"• $K_{{S1}}$ = {DEFAULT_PARAMS['K_S1']} g COD/L (Monod half-velocity)")
        st.write(f"• $k_1$ = {DEFAULT_PARAMS['k1']} g COD/g VS (Yield ratio)")
    with kcol2:
        st.write(f"• $\\mu_{{2,max}}$ = {DEFAULT_PARAMS['mu2_max']} 1/d (Max methanogen growth)")
        st.write(f"• $K_{{S2}}$ = {DEFAULT_PARAMS['K_S2']} mmol/L (Substrate affinity)")
        st.write(f"• $K_I$ = {DEFAULT_PARAMS['K_I']} mmol/L (Haldane VFA inhibition)")
    with kcol3:
        st.write(f"• $k_2$ = {DEFAULT_PARAMS['k2']} mmol VFA/g VS")
        st.write(f"• $k_3$ = {DEFAULT_PARAMS['k3']} mmol VFA/g VS")
        st.write(f"• $k_6$ = {DEFAULT_PARAMS['k6']} mmol CH4/g VS")

# -----------------------------------------------------------------------------
# Execute Simulation
# -----------------------------------------------------------------------------
run_sim = st.button("▶️ Solve AM2 ODE System", type="primary", use_container_width=True)

# Run automatically or on button click
try:
    t_eval = np.arange(0, duration + 0.5, 0.5)
    res = solve_am2(
        t_span=(0.0, float(duration)),
        y0=[s1_0, x1_0, s2_0, x2_0],
        D_func=lambda t: D_val,
        S1_in_func=lambda t: s1_in,
        alk_in_mmol=alk_in,
        t_eval=t_eval
    )
except Exception as e:
    st.error(f"Numerical integration error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Dynamic Simulation Trajectories")

tab_states, tab_biogas, tab_chem, tab_haldane = st.tabs([
    "🧬 State Variables (S1, X1, S2, X2)",
    "⚡ Methane Yield & Production",
    "🧪 pH, VFA & Simulated FOS/TAC",
    "🎯 Haldane Kinetic Inhibition Curve"
])

# TAB 1: States
with tab_states:
    fig_states = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Soluble Substrate S1 [g COD/L]",
            "Acidogenic Biomass X1 [g VS/L]",
            "Volatile Fatty Acids S2 [mmol/L]",
            "Methanogenic Biomass X2 [g VS/L]"
        ]
    )
    fig_states.add_trace(go.Scatter(x=res.t, y=res.S1, mode="lines", name="S1", line=dict(color="#1f77b4", width=2.5)), row=1, col=1)
    fig_states.add_trace(go.Scatter(x=res.t, y=res.X1, mode="lines", name="X1", line=dict(color="#2ca02c", width=2.5)), row=1, col=2)
    fig_states.add_trace(go.Scatter(x=res.t, y=res.S2, mode="lines", name="S2", line=dict(color="#d62728", width=2.5)), row=2, col=1)
    fig_states.add_trace(go.Scatter(x=res.t, y=res.X2, mode="lines", name="X2", line=dict(color="#9467bd", width=2.5)), row=2, col=2)
    
    fig_states.update_layout(height=600, showlegend=False, template="plotly_white", margin=dict(t=50, b=40))
    fig_states.update_xaxes(title_text="Time [days]")
    st.plotly_chart(fig_states, use_container_width=True)

# TAB 2: Biogas / Methane
with tab_biogas:
    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        fig_ch4 = go.Figure()
        fig_ch4.add_trace(go.Scatter(
            x=res.t, y=res.q_CH4_Ld,
            mode="lines", name="q_CH4 (L/L·d)",
            line=dict(color="#ff7f0e", width=3)
        ))
        fig_ch4.update_layout(
            title="Methane Volumetric Generation Rate",
            xaxis_title="Time [days]",
            yaxis_title="q_CH4 [L CH4 / L reactor · day]",
            template="plotly_white"
        )
        st.plotly_chart(fig_ch4, use_container_width=True)
    with col_b2:
        st.markdown("#### Gas Production Metrics")
        max_ch4 = np.max(res.q_CH4_Ld)
        final_ch4 = res.q_CH4_Ld[-1]
        mean_ch4 = np.mean(res.q_CH4_Ld)
        st.metric("Peak Methane Rate", f"{max_ch4:.3f} L/L·d")
        st.metric("Final Steady Rate", f"{final_ch4:.3f} L/L·d")
        st.metric("Mean Production", f"{mean_ch4:.3f} L/L·d")
        if final_ch4 < 0.005:
            st.error("🚨 Washout detected: Methanogenic population has collapsed.")
        else:
            st.success("✅ Sustainable biomethane production.")

# TAB 3: Chemistry & FOS/TAC
with tab_chem:
    fig_chem = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "pH Profile (Carbonate Buffer Equilibrium)",
            "VFA (mg/L Acetic Acid Eq.) & Total Alkalinity (mg/L CaCO3)",
            "Simulated FOS/TAC Proxy (VFA mg/L / Alkalinity mg/L)"
        ]
    )
    # pH
    fig_chem.add_trace(go.Scatter(x=res.t, y=res.pH, mode="lines", name="pH", line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig_chem.add_hrect(y0=6.8, y1=7.6, fillcolor="rgba(44, 160, 44, 0.15)", line_width=0, row=1, col=1)

    # VFA & Alk
    fig_chem.add_trace(go.Scatter(x=res.t, y=res.vfa_mgl, mode="lines", name="VFA (mg/L)", line=dict(color="#d62728", width=2)), row=2, col=1)
    fig_chem.add_trace(go.Scatter(x=res.t, y=res.alkalinity_mgl, mode="lines", name="Alkalinity (mg/L)", line=dict(color="#2ca02c", width=2)), row=2, col=1)

    # FOS/TAC proxy
    sim_ratio = res.vfa_mgl / np.clip(res.alkalinity_mgl, 1.0, None)
    fig_chem.add_trace(go.Scatter(x=res.t, y=sim_ratio, mode="lines", name="Sim FOS/TAC Proxy", line=dict(color="#ff7f0e", width=2.5)), row=3, col=1)
    fig_chem.add_hline(y=0.30, line_dash="dash", line_color="orange", annotation_text="Warning (0.30)", row=3, col=1)
    fig_chem.add_hline(y=0.40, line_dash="solid", line_color="red", annotation_text="High Risk (0.40)", row=3, col=1)

    fig_chem.update_layout(height=750, template="plotly_white", hovermode="x unified", margin=dict(t=50, b=30))
    fig_chem.update_yaxes(title_text="pH [su]", row=1, col=1)
    fig_chem.update_yaxes(title_text="Concentration [mg/L]", row=2, col=1)
    fig_chem.update_yaxes(title_text="Proxy Ratio [-]", row=3, col=1)
    fig_chem.update_xaxes(title_text="Time [days]", row=3, col=1)
    st.plotly_chart(fig_chem, use_container_width=True)

# TAB 4: Haldane Curve
with tab_haldane:
    st.markdown("#### Methanogen Kinetics: Haldane Substrate Inhibition")
    s2_sweep = np.linspace(0.1, 150.0, 300)
    mu2_vals = [mu2(s, DEFAULT_PARAMS) for s in s2_sweep]
    current_s2 = res.S2[-1]
    current_mu2 = mu2(current_s2, DEFAULT_PARAMS)

    fig_hal = go.Figure()
    fig_hal.add_trace(go.Scatter(
        x=s2_sweep, y=mu2_vals,
        mode="lines", name="Haldane Growth Rate mu2(S2)",
        line=dict(color="#2ca02c", width=3)
    ))
    fig_hal.add_trace(go.Scatter(
        x=[current_s2], y=[current_mu2],
        mode="markers", name=f"Final State: S2={current_s2:.1f} mmol/L",
        marker=dict(color="red", size=12, symbol="star")
    ))
    fig_hal.update_layout(
        title="Methanogen Specific Growth Rate mu2 vs VFA Concentration S2",
        xaxis_title="VFA Concentration S2 [mmol/L]",
        yaxis_title="Specific Growth Rate mu2 [1/day]",
        template="plotly_white"
    )
    st.plotly_chart(fig_hal, use_container_width=True)
    st.caption("Notice how mu2 increases initially up to an optimal VFA concentration, beyond which severe substrate inhibition suppresses methanogenic growth.")

# -----------------------------------------------------------------------------
# Attribution & Scientific Integrity
# -----------------------------------------------------------------------------
st.info("""
📖 **Scientific Literature Reference:**
Bernard, O., Hadj-Sadok, Z., Dochain, D., Genovesi, A., & Steyer, J.-P. (2001).
*Dynamical model development and parameter identification for an anaerobic wastewater treatment process.*
Biotechnology and Bioengineering, 75(4), 424–438. https://doi.org/10.1002/bit.10036

*Disclaimer: This is an uncalibrated theoretical kinetic simulation demonstrating biochemical non-linearities and washout phenomena. It does not represent any specific commercial plant.*
""")
