"""
run_all.py – Master pipeline runner
====================================
Executes all phases in order, checks for pre-requisites, and provides
a final summary of results. Run from the project root.

Usage:
    python run_all.py
    python run_all.py --skip-eda   (skip EDA figure generation)
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def _ok(msg): print(f"  [OK] {msg}")
def _run(fn, label):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)
    try:
        fn()
        _ok(f"{label} done in {time.time()-t0:.1f}s")
        return True
    except Exception as exc:
        print(f"  [FAILED] {label}: {exc}")
        import traceback; traceback.print_exc()
        return False

def phase1():
    from src.audit import run as audit_run
    audit_run()

def phase2_eda():
    from src.eda import run_all
    run_all()

def phase2_preprocess():
    """Preprocessing already done; processed files are in data/processed/."""
    for fn in ['pilot_D1_harmonised.csv', 'pilot_D2_harmonised.csv', 'full_harmonised.csv']:
        assert (ROOT / 'data' / 'processed' / fn).exists(), f"Missing: {fn}"
    _ok("Processed files verified")

def phase3_biogas():
    from src.run_biogas import main
    main()

def phase3_anomaly():
    from src.run_anomaly import main
    main()

def phase4_simulation():
    from src.simulation import run
    run()

def phase5_classifier():
    from src.supervised_classifier import run_supervised_classification
    run_supervised_classification()

def phase6_lstm():
    from src.lstm_forecasting import run_real_data, run_synthetic_data
    from pathlib import Path
    models = ROOT / 'models'; models.mkdir(exist_ok=True)
    real = run_real_data(horizons=[1, 3, 7], seq_len=7, save_dir=models)
    if not real.empty:
        real.to_csv(ROOT / 'reports' / 'lstm_metrics_real.csv', index=False)
    syn = run_synthetic_data(horizons=[1, 3, 7], seq_len=7, save_dir=models)
    if not syn.empty:
        syn.to_csv(ROOT / 'reports' / 'lstm_metrics_synthetic.csv', index=False)

def phase7_early_warning():
    from src.early_warning import scan_warnings, warnings_to_df, add_warning_column
    import pandas as pd
    for ds in ['pilot_D1', 'pilot_D2']:
        path = ROOT / 'data' / 'processed' / f'{ds}_harmonised.csv'
        if not path.exists(): continue
        df = pd.read_csv(path, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
        df = add_warning_column(df, 'va_alk', 'warning_category')
        ws = scan_warnings(df, 'va_alk', data_source='real', target_label='VA/Alk ratio (real; NOT FOS/TAC)')
        wdf = warnings_to_df(ws)
        wdf.to_csv(ROOT / 'reports' / f'early_warnings_{ds}.csv', index=False)
        _ok(f"Warnings for {ds}: {len(wdf)} rows")

def phase8_mpc():
    from src.mpc_simulator import run_mpc_simulation
    mpc_df, bl_df = run_mpc_simulation(t_total=100)
    mpc_df.to_csv(ROOT / 'reports' / 'mpc_controlled.csv', index=False)
    bl_df.to_csv(ROOT / 'reports' / 'mpc_baseline.csv', index=False)
    print(f"  Baseline max VFA: {bl_df['vfa_mgl'].max():.1f} mg/L")
    print(f"  MPC max VFA:      {mpc_df['vfa_mgl'].max():.1f} mg/L")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-eda', action='store_true', help='Skip EDA figure generation')
    args = parser.parse_args()

    phases = [
        (phase1,           "Phase 1: Data Audit"),
        (phase2_preprocess,"Phase 2: Preprocessing verification"),
    ]
    if not args.skip_eda:
        phases.append((phase2_eda, "Phase 2: EDA figures"))
    phases += [
        (phase3_biogas,    "Phase 3A: Biogas prediction (real data)"),
        (phase3_anomaly,   "Phase 3B: Unsupervised anomaly detection (real data)"),
        (phase4_simulation,"Phase 4: AM2 simulation + synthetic data"),
        (phase5_classifier,"Phase 5: Supervised classification (synthetic)"),
        (phase6_lstm,      "Phase 6: LSTM / time-series forecasting"),
        (phase7_early_warning,"Phase 7: Early warning system"),
        (phase8_mpc,       "Phase 8: MPC demonstration"),
    ]

    print("\nAnaerobic Digester Anomaly Detection – Full Pipeline")
    print("=" * 60)
    ok = sum(1 for fn, lbl in phases if _run(fn, lbl))
    print(f"\nCompleted {ok}/{len(phases)} phases.")
    print("Run 'streamlit run dashboard/app.py' to launch the dashboard.")

