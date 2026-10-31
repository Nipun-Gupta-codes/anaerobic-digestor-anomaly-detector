"""
Anaerobic Digester Anomaly Detection & Early Warning System
===========================================================
Streamlit Multi-Page Dashboard Entry Point
Final-Year Chemical Engineering Project
"""
import sys
from pathlib import Path
import streamlit as st

# Robustly ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
for parent in [ROOT] + list(ROOT.parents):
    if (parent / "src").is_dir() and (parent / "data").is_dir():
        ROOT = parent
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Anaerobic Digester Anomaly Detection",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Overview & Data": [
        st.Page("pages/1_overview.py", title="Project Overview", icon="📋", default=True),
        st.Page("pages/2_dataset_explorer.py", title="Dataset Explorer", icon="🔍"),
        st.Page("pages/3_digester_monitoring.py", title="Digester Monitoring", icon="📈"),
    ],
    "Anomaly & Kinetics": [
        st.Page("pages/4_anomaly_detection.py", title="Anomaly Detection", icon="⚠️"),
        st.Page("pages/5_am2_simulator.py", title="AM2 Kinetic Simulator", icon="⚙️"),
    ],
    "Forecasting & Control": [
        st.Page("pages/6_lstm_forecasting.py", title="Time-Series Forecasting", icon="🔮"),
        st.Page("pages/7_early_warning.py", title="Early Warning System", icon="🚨"),
        st.Page("pages/8_mpc_simulation.py", title="MPC Simulation", icon="🎛️"),
    ],
}

# Sidebar metadata / branding
with st.sidebar:
    st.markdown("### 🧪 Anaerobic Digestion AD-EWS")
    st.caption("Chemical Engineering Capstone Project")
    st.markdown("---")
    st.markdown(
        "**Key Systems:**\n"
        "- **Pilot D1 & D2**: 309 days (US units)\n"
        "- **Full-Scale**: 326 days (Municipal)\n"
        "- **AM2 Kinetic**: 6 synthetic regimes"
    )
    st.markdown("---")
    st.info(
        "🔬 **Academic Honesty Notice**:\n\n"
        "Real operational data and synthetic AM2 kinetics are kept separate. "
        "Real data contains no confirmed souring events (full-scale is highly stable). "
        "No FOS/TAC was measured in real data — only mass VA/Alk."
    )

pg = st.navigation(pages)
pg.run()
