"""
Page 1: Project Overview & Scientific Framework
================================================
Anaerobic Digester Anomaly Detection & Early Warning System
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Set up project root in sys.path
ROOT = Path(__file__).resolve().parent
for parent in [ROOT] + list(ROOT.parents):
    if (parent / "src").is_dir() and (parent / "data").is_dir():
        ROOT = parent
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    st.set_page_config(page_title="Project Overview", page_icon="📋", layout="wide")
except Exception:
    pass

st.title("📋 Project Overview: Anaerobic Digester Anomaly Detection & Control")
st.caption("Chemical Engineering Capstone Project | Machine Learning & Kinetic Process Modeling")

st.markdown("""
Anaerobic digestion (AD) is a complex, multi-stage biochemical process where consortia of 
acidogenic bacteria and methanogenic archaea convert organic matter into renewable biogas 
(primarily $CH_4$ and $CO_2$). Because methanogens have substantially slower specific growth rates 
than acidogens and are susceptible to volatile fatty acid (VFA) inhibition, operational imbalances 
can lead to **reactor souring** (acidification, pH depression, and biological washout).

This project designs, tests, and evaluates a multi-tier monitoring, anomaly detection, stability forecasting, 
and predictive control architecture grounded in both **real plant telemetry** and **first-principles kinetics**.
""")

# Key Project Objectives
st.subheader("🎯 Core Objectives")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("#### 1. Real Telemetry Audit")
    st.write(
        "Examine 600+ days of operational data across two pilot digesters (D1 & D2) and a municipal "
        "full-scale co-digestion facility to quantify real-world data quality and sensor gaps."
    )
with col2:
    st.markdown("#### 2. Unsupervised Novelty")
    st.write(
        "Detect abnormal operating days without ground-truth labels using Isolation Forest, "
        "Local Outlier Factor (LOF), and One-Class SVM on chronological operational splits."
    )
with col3:
    st.markdown("#### 3. AM2 Mechanistic ODEs")
    st.write(
        "Simulate controlled organic overload and biological souring via the Bernard et al. (2001) "
        "two-step kinetic model to generate synthetic labelled regimes and validate early warning rules."
    )
with col4:
    st.markdown("#### 4. Control & Mitigation")
    st.write(
        "Formulate early warning indicator bands and demonstrate closed-loop Model Predictive "
        "Control (MPC) to modulate organic feed loading and prevent fatal reactor acidification."
    )

st.markdown("---")

# Architectural Workflow Diagram
st.subheader("🔄 End-to-End Methodology & Data Pipeline")
st.markdown("""
```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               1. DATA SOURCES & INGESTION                               │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│ Real Operational Telemetry               │ First-Principles Kinetic Model              │
│ • Pilot D1 & D2 (309 days, 73 cols)      │ • AM2 Two-Step ODEs (Bernard et al., 2001)  │
│ • Full-Scale Digester (326 days, 47 cols)│ • 6 Labelled Regimes (Normal to Souring)    │
│ • Full-Scale WWC Feed (319 days, 33 cols)│ • Synthetic FOS/TAC Proxy Benchmark         │
└────────────────────┬─────────────────────┴──────────────────────┬──────────────────────┘
                     │                                            │
                     ▼                                            ▼
┌──────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│     2. PREPROCESSING & HARMONISATION     │ │     4. SUPERVISED KINETIC BENCHMARK       │
│ • Unit conversions (US -> SI metric)     │ │ • Train RF, GB, and XGBoost on AM2 data   │
│ • Sensor artifact & quality filtering    │ │ • Feature importance (S2, X2, pH drivers) │
│ • Missing data audit & imputation checks │ │ • Exploratory transfer test on pilot data │
└────────────────────┬─────────────────────┘ └───────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│     3. UNSUPERVISED ANOMALY DETECTION    │ │      5. MULTI-STEP STABILITY FORECAST     │
│ • Train on 60% chronological baseline    │ │ • Persistence vs HistGB vs LSTM (h=1,3,7) │
│ • Score holdout days (Novelty detection) │ │ • Evaluation against real sparse sampling │
│ • Validate against chemical proxy events │ │ • Quantification of practical limits      │
└────────────────────┬─────────────────────┘ └────────────────────┬──────────────────────┘
                     │                                            │
                     └─────────────────────┬──────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             6. DECISION SUPPORT & CONTROL                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ • Multi-Tier Early Warning System: Classification into Underfed, Optimal, Early Warning│
│ • Closed-Loop Model Predictive Control (MPC): Receding-horizon dilution rate control   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```
""")

st.markdown("---")

# Dataset Summary Table
st.subheader("📊 Dataset Summary Table")

dataset_summary = pd.DataFrame([
    {
        "Dataset": "Pilot Digesters (D1 & D2)",
        "Raw File": "sensor_ad_pilot_merged.csv",
        "Rows x Cols": "309 x 73",
        "Period": "2019-10-17 to 2020-08-31",
        "Biogas Scale": "~57 cf/day (0.07 m³/h)",
        "Key Telemetry": "pH, Temp, Gas, CCR, Alk, VFA, Feed flow (US units)",
        "Observed Dynamics": "Contains genuine souring/overload events in May–June 2020",
        "Completeness": "Intermittent lab chemistry (VFA/Alk ~60%, 11 missing days)"
    },
    {
        "Dataset": "Full-Scale Digester",
        "Raw File": "sensor_ad_full_scale_digester_ccr.csv",
        "Rows x Cols": "326 x 47",
        "Period": "2021-03-12 to 2022-01-31",
        "Biogas Scale": "~199,000 cf/day (235 m³/h)",
        "Key Telemetry": "Biogas, Methane flow, Temp, CCR, Feed loadings, Lab VFA/Alk",
        "Observed Dynamics": "Continuously stable municipal operation (zero souring events)",
        "Completeness": "Daily biogas (98%), but pH is only 21% and Alk is 40%"
    },
    {
        "Dataset": "Full-Scale WWC Feed",
        "Raw File": "sensor_ad_full_scale_wwc_ccr.csv",
        "Rows x Cols": "319 x 33",
        "Period": "2021-03-12 to 2022-01-24",
        "Biogas Scale": "N/A (Feed stream only)",
        "Key Telemetry": "Influent CCR, Slurry CCR, TS%, VS%, Feed stream volumes",
        "Observed Dynamics": "Feed characterization & sensor correlation only",
        "Completeness": "No digester state or biogas telemetry"
    },
    {
        "Dataset": "AM2 Kinetic Simulation",
        "Raw File": "am2_synthetic_labelled.csv",
        "Rows x Cols": "546 x 16",
        "Period": "Continuous ODE Days 0–545",
        "Biogas Scale": "0.00 – 1.80 L CH4/(L·d)",
        "Key Telemetry": "S1, X1, S2, X2, q_CH4, pH, Alkalinity, FOS/TAC proxy",
        "Observed Dynamics": "6 controlled regimes: Normal, Overloading, Acidification, Souring",
        "Completeness": "100% complete daily synthetic simulation"
    }
])

st.dataframe(dataset_summary, use_container_width=True, hide_index=True)

st.markdown("---")

# Key Findings Carousel (Tabs)
st.subheader("💡 Key Research Findings & Engineering Insights")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧪 1. FOS/TAC vs Real VA/Alk",
    "⚖️ 2. Plant Stability Asymmetry",
    "⚠️ 3. Sensor Quality & Artifacts",
    "📈 4. Forecasting Constraints",
    "🎛️ 5. AM2 Simulation & MPC"
])

with tab1:
    st.markdown("#### The Critical Distinction: Titration FOS/TAC vs Gravimetric VA/Alk")
    st.markdown("""
    - **No Titration FOS/TAC in Real Data**: Neither the pilot nor the full-scale industrial facilities 
      measured FOS/TAC via the standard two-point Nordmann titration method (which reports volatile organic acids in 
      $mg/L$ acetic acid equivalent relative to buffer capacity in $mg/L$ $CaCO_3$ equivalent).
    - **True Nature of Real Ratio**: In the real datasets, the ratio is strictly a gravimetric mass-to-mass quotient:
      $$\\text{Ratio} = \\frac{\\text{Volatile Acids } (mg/L)}{\\text{Total Alkalinity } (mg/L)}$$
    - **Academic Consequence**: Canonical literature thresholds (e.g. Lossie & Pütz, 2008: $0.23$ underfed, $0.23-0.30$ optimal, 
      $0.30-0.40$ warning, $>0.40$ souring) **cannot be directly equated** without clear caveat. In this work, 
      thresholds applied to real data are designated as *indicative project thresholds*, while true FOS/TAC analysis 
      is reserved for the AM2 kinetic benchmark.
    """)

with tab2:
    st.markdown("#### Operational Asymmetry: Pilot Instability vs Full-Scale Resilience")
    st.markdown("""
    - **Full-Scale Digester Never Soured**: The full-scale municipal facility maintained high buffering capacity throughout 
      its 326 monitored days ($pH = 7.08 - 7.39$, maximum VFA $= 114\\ mg/L$, recomputed $VA/Alk \\le 0.03$). 
      It is a prime example of an exceptionally well-operated commercial plant with **no real biological souring**.
    - **Pilot Digesters Exhibited Genuine Upsets**: In contrast, Pilot Digester 2 (and D1) experienced severe biological stress 
      between May 14 and June 8, 2020, where VFA climbed above $2,300\\ mg/L$, $VA/Alk$ peaked at $0.82$, and biogas dropped by over $80\%$.
    - **Engineering Takeaway**: The pilot provides the only ground-truth empirical testing ground for physical upset recovery, 
      while the full-scale dataset serves as a benchmark for baseline false-positive stability.
    """)

with tab3:
    st.markdown("#### Data Quality & Physical Disconnects")
    st.markdown("""
    - **Pilot D1 Alkalinity Anomaly (Oct–Dec 2019)**: Recomputing VA/Alk from raw columns yielded deceptive values around $1.0$ 
      because alkalinity was logged as a near-constant $\\sim 85\\ mg/L$ while pH was $7.4 - 7.7$ and biogas was normal. 
      This was identified as a laboratory/entry artifact, flagged as suspect (`flag_alk_suspect`), and masked to prevent false alarms.
    - **Flowmeter Counter Resets**: Multiple zero-gas generation days occurred in the pilot data accompanied by maintenance notes 
      documenting gas meter mechanical failure and counter resets.
    - **Lesson**: Naive anomaly detection without domain-aware chemistry checks flags sensor glitches rather than biochemical state changes.
    """)

with tab4:
    st.markdown("#### Sequential Forecasting Feasibility")
    st.markdown("""
    - **Intermittent Lab Sampling**: Because manual titrations and distillation tests were conducted only every 2 to 3 days, 
      the maximum continuous run of daily observations was 13 days in D1 and only 6 days in D2.
    - **LSTM Feasibility**: Multi-day ahead deep learning models (e.g., 7-day ahead LSTM) trained on real chemistry suffer 
      from massive observational gaps. 
    - **Result**: On real 1-day ahead forecasts, a simple **persistence baseline** ($y_t = y_{t-1}$) achieves $R^2 = 0.32$ to $0.71$, 
      outperforming deep models that suffer from small-sample overfitting.
    """)

with tab5:
    st.markdown("#### AM2 Mechanistic ODEs & Model Predictive Control (MPC)")
    st.markdown("""
    - **Mechanistic Grounding**: The Bernard et al. (2001) AM2 model provides a rigorous mathematical bridge, coupling 
      acidogenic Monod kinetics with methanogenic Haldane substrate-inhibition kinetics:
      $$\\mu_2(S_2) = \\mu_{2,max} \\frac{S_2}{K_{S2} + S_2 + S_2^2 / K_I}$$
    - **Closed-Loop MPC Demonstration**: A receding-horizon controller was formulated to optimize feed dilution rate $D(t)$ 
      during acute shock-loadings. By penalizing $S_2$ accumulation above inhibition thresholds, the MPC successfully restricts 
      peak VFA by up to $60\\%$ compared to an uncontrolled digester.
    """)

st.markdown("---")

# Scientific Integrity Statement
st.markdown("### 🛡️ Scientific Integrity Statement")
st.warning("""
**Commitment to Transparent Chemical Engineering Research:**
1. **No Fictitious Labels**: Unsupervised anomaly detection methods are evaluated against transparent statistical baselines 
   and heuristic chemical evidence proxies; no unverified 'ground truth' souring labels were manufactured.
2. **Strict Siloing**: Real plant measurements and synthetic ODE simulations are maintained in distinct data pipelines. 
   Simulation models are explicitly designated as uncalibrated literature simulations (Bernard et al., 2001).
3. **Transparent Limitations**: All instrument failures, sampling gaps, and unit conversion assumptions are fully disclosed 
   in the accompanying project audit reports (`reports/phase1_data_audit.md` and `reports/phase2_phase3_report.md`).
""")
