# Advanced Machine Learning for Early Anomaly Detection in Anaerobic Digesters

**Final-Year Chemical Engineering Project**
*Predicting Reactor Souring via Kinetic Modelling*

---

## Overview

This project implements a hybrid ML + mechanistic modelling framework for early detection of anaerobic digester souring. It combines:

- Real-world operational data from two anaerobic digestion systems (pilot and full-scale)
- The AM2 two-step kinetic model (Bernard et al., 2001) for synthetic scenario generation
- Multiple machine learning models: Random Forest, Gradient Boosting, Isolation Forest, LOF, One-Class SVM
- LSTM-based time-series forecasting (GradientBoosting fallback when TF unavailable)
- A rule-based early warning system using VA/Alk and simulated FOS/TAC ratios
- A simplified MPC demonstration using AM2 as a plant proxy

## Project Structure

```
anaerobic-digester-anomaly-detection/
├── data/
│   ├── raw/                          # Three original datasets
│   ├── processed/                    # Harmonised per-reactor CSVs + anomaly scores
│   └── synthetic/                    # AM2-generated labelled dataset
├── src/
│   ├── data_loader.py               # Dataset loading with schema validation
│   ├── preprocessing.py             # Unit harmonisation, quality flags, splits
│   ├── feature_engineering.py       # Lags, rolling stats, trend features
│   ├── eda.py                       # EDA figures (18 plots)
│   ├── audit.py                     # Phase 1 data audit
│   ├── biogas_prediction.py         # Real-data biogas prediction models
│   ├── anomaly_detection.py         # Unsupervised anomaly detection (IF, LOF, OCSVM)
│   ├── am2_model.py                 # AM2 ODE model (Bernard et al., 2001)
│   ├── simulation.py                # Scenario generation + synthetic dataset
│   ├── supervised_classifier.py     # 6-class synthetic data classification
│   ├── lstm_forecasting.py          # Time-series forecasting (LSTM / HistGB fallback)
│   ├── early_warning.py             # Warning system with FOS/TAC thresholds
│   └── mpc_simulator.py             # Simplified MPC demonstration
├── dashboard/
│   ├── app.py                       # Streamlit entry point
│   └── pages/                       # 8 dashboard pages
├── notebooks/                       # Jupyter notebooks (01-09)
├── models/                          # Saved trained models (.pkl)
├── reports/
│   ├── figures/                     # EDA figures (18 PNG files)
│   ├── phase1_data_audit.md
│   ├── phase2_phase3_report.md
│   ├── biogas_prediction_results.csv
│   ├── anomaly_detection_results.csv
│   ├── classification_summary.csv
│   ├── lstm_metrics_real.csv
│   ├── lstm_metrics_synthetic.csv
│   ├── early_warnings_pilot_D1.csv
│   ├── mpc_controlled.csv
│   └── mpc_baseline.csv
├── tests/                           # Unit tests
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.10+ (3.12 recommended)
- `uv` package manager (recommended) or `pip`

### Setup

```bash
# Clone or download the repository
cd anaerobic-digester-anomaly-detection

# Create virtual environment and install dependencies
uv venv .venv
uv pip install -r requirements.txt --python .venv/Scripts/python.exe

# OR with pip:
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### Data Files (Required)
Place the following files in `data/raw/`:
- `sensor_ad_pilot_merged.csv`
- `sensor_ad_full_scale_digester_ccr.csv`
- `sensor_ad_full_scale_wwc_ccr.csv`

## Running the Project

### Full pipeline (all phases in sequence):
```bash
.venv\Scripts\python.exe run_all.py
```

### Individual phases:
```bash
# Phase 1: Data audit
.venv\Scripts\python.exe src/audit.py

# Phase 2-3: EDA, preprocessing, biogas prediction, anomaly detection
.venv\Scripts\python.exe src/eda.py
.venv\Scripts\python.exe src/run_biogas.py
.venv\Scripts\python.exe src/run_anomaly.py

# Phase 4-5: AM2 simulation + supervised classification
.venv\Scripts\python.exe src/simulation.py
.venv\Scripts\python.exe src/supervised_classifier.py

# Phase 6: LSTM forecasting
.venv\Scripts\python.exe src/lstm_forecasting.py

# Phase 7: Early warning system
.venv\Scripts\python.exe src/early_warning.py

# Phase 8: MPC demonstration
.venv\Scripts\python.exe src/mpc_simulator.py

# Phase 9: Interactive dashboard
.venv\Scripts\streamlit.exe run dashboard/app.py
```

### Run tests:
```bash
.venv\Scripts\pytest.exe tests/ -v
```

## Key Results

### Phase 1 – Data Audit
| Dataset | Rows x Cols | Period |
|---|---|---|
| pilot | 309 x 73 | Oct 2019 – Aug 2020 |
| full_digester | 326 x 47 | Mar 2021 – Jan 2022 |
| full_wwc | 319 x 33 | Mar 2021 – Jan 2022 |

Key findings:
- No FOS/TAC measurements in any dataset; only VA/Alk ratio available
- One real instability episode: pilot May–June 2020 (VA/Alk up to 0.82)
- Full-scale digester is stable throughout (max VA/Alk = 0.027)

### Phase 3 – Biogas Prediction (Real Data)
| System | Persistence MAE | Best ML MAE |
|---|---|---|
| Pilot D1 | 0.166 m³/d | 0.197 (RF + AR) |
| Pilot D2 | 0.137 m³/d | 0.304 (RF + AR) |
| Full-scale | 1379 m³/d | 1620 (HistGB + AR) |

**Conclusion**: No tested ML model beats next-day persistence. Level drift between training and test periods is the dominant prediction error.

### Phase 3 – Unsupervised Anomaly Detection
| System | Detector | AUC |
|---|---|---|
| Pilot D2 | IsolationForest | 0.93 |
| Pilot D2 | LOF | 0.94 |
| Pilot D1 | LOF | 0.85 |
| Full-scale | — | N/A (no instability episodes) |

### Phase 4 – AM2 Simulation
- 546 synthetic time-series observations across 5 scenarios
- All 6 classes represented: Normal (151), Overloading (55), Early Acidification (76), High VFA (70), Severe Souring (131), Recovery (63)
- MPC demonstration: Baseline max VFA = 3,231 mg/L → MPC max VFA = 120 mg/L

### Phase 5 – Supervised Classification (Synthetic Data Only)
| Model | Accuracy |
|---|---|
| GradientBoosting | 0.558 |
| RandomForest | 0.442 |

Note: Low accuracy reflects chronological test split where some classes only appear in training scenarios. This is a fundamental small-dataset limitation.

### Phase 6 – LSTM Forecasting
- TensorFlow not installed; HistGradientBoosting used as fallback
- Real data (pilot D1, VA/Alk, h=1): Persistence MAE = 0.011, HistGB MAE = 0.017
- Persistence remains the best model for real sparse VA/Alk data
- Synthetic AM2 (h=1): HistGB R² = 0.034 (marginal improvement over train-mean)

## Scientific Integrity Notes

1. **VA/Alk ≠ FOS/TAC**: The real datasets contain VA/Alk mass ratios. FOS/TAC is a titration-based ratio. The thresholds (0.23/0.30/0.40/1.00) are defined for FOS/TAC and are applied only to simulated data, labelled as project-specific.

2. **Synthetic data**: AM2 simulations use literature parameters from Bernard et al. (2001). The model is NOT calibrated against any specific plant. Results are labelled as simulations.

3. **One real instability episode**: Only the May–June 2020 pilot events represent confirmed operational instability. Anomaly detection results have high uncertainty.

4. **MPC disclaimer**: The MPC demonstration is a simplified illustration. It has NOT been validated on a real plant.

5. **Domain shift**: Classifiers trained on AM2 synthetic data cannot be directly applied to real data without careful feature alignment and validation.

## References

1. Bernard, O., et al. (2001). Dynamical model development and parameter identification for an anaerobic wastewater treatment process. *Biotechnology & Bioengineering*, 75(4), 424–438.

2. Lossie, U., & Pütz, P. (2008). *Targeted control of biogas plants with FOS/TAC*. Practice Report, HACH LANGE.

3. Méndez-Acosta, H. O., et al. (2010). A robust control scheme to improve the stability of anaerobic digestion processes. *Journal of Process Control*, 20(4), 375–383.

4. Jimenez, J., et al. (2015). Instrumentation and control of anaerobic digestion processes: a review. *Water Science & Technology*, 72(12), 2102–2114.

## License

Academic project. Not for industrial use without validation.

