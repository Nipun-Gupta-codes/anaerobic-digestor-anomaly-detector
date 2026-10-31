"""
Synthetic scenario simulation using the AM2 model.
===================================================
Generates labelled time-series for 6 operating classes (Bernard et al., 2001):

  Class 0 – Normal operation
  Class 1 – Organic overloading
  Class 2 – Early acidification
  Class 3 – High VFA accumulation
  Class 4 – Severe reactor souring
  Class 5 – Recovery

Each scenario transitions between classes in a physically meaningful sequence.
The full dataset is produced by concatenating multiple scenario runs.

Scientific integrity notes
--------------------------
* All kinetic equations and nominal parameter values come from Bernard et al. (2001).
* pH and alkalinity are estimated via a simplified carbonate-buffer model (qualitative).
* This is a SIMULATION, not a calibrated plant model.
* No real measurement is mixed with synthetic data.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.am2_model import solve_am2, DEFAULT_PARAMS

SEED = 42
rng  = np.random.default_rng(SEED)

# Use Radau for stiff scenarios (more robust than LSODA under extreme VFA)
_STIFF_METHOD = 'Radau'

LABEL_MAP = {
    0: 'Normal',
    1: 'Overloading',
    2: 'Early Acidification',
    3: 'High VFA',
    4: 'Severe Souring',
    5: 'Recovery',
}

# ---------------------------------------------------------------------------
# Helper: step-function utilities
# ---------------------------------------------------------------------------
def _step(t, t0, v_before, v_after):
    return v_before if t < t0 else v_after

def _ramp(t, t0, t1, v0, v1):
    if t <= t0: return v0
    if t >= t1: return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
def _scenario_A(seed_offset=0):
    """Normal stable operation (80 days)."""
    y0 = [5.0, 0.5, 2.0, 0.5]
    # Deterministic S1_in for ODE stability; noise added post-solve
    return solve_am2((0, 80), y0,
                     D_func=lambda t: 0.5,
                     S1_in_func=lambda t: 10.0,
                     alk_in_mmol=25.0,
                     label_func=lambda t: 0,
                     label_map=LABEL_MAP)

def _scenario_B(seed_offset=0):
    """Normal -> overloading -> early acidification (120 days)."""
    y0 = [5.0, 0.6, 3.0, 0.55]
    def S1i(t): return 10.0 if t < 40 else 22.0
    def lf(t):
        if t < 40: return 0
        if t < 75: return 1
        return 2
    return solve_am2((0, 120), y0,
                     D_func=lambda t: 0.5,
                     S1_in_func=S1i,
                     alk_in_mmol=22.0,
                     label_func=lf,
                     label_map=LABEL_MAP)

def _scenario_C(seed_offset=0):
    """High VFA accumulation -> Severe souring (100 days)."""
    y0 = [8.0, 0.7, 10.0, 0.35]
    # Use lower dilution than Bernard nominal to cause VFA build-up
    def lf(t):
        if t < 20: return 2
        if t < 50: return 3
        return 4
    return solve_am2((0, 100), y0,
                     D_func=lambda t: 0.55,
                     S1_in_func=lambda t: 26.0,
                     alk_in_mmol=18.0,
                     label_func=lf,
                     label_map=LABEL_MAP)

def _scenario_D(seed_offset=0):
    """Severe souring -> recovery (140 days, split at day 80 for alk dosing)."""
    y0 = [10.0, 0.5, 25.0, 0.2]
    def D(t): return 0.5 if t < 80 else _ramp(t, 80, 100, 0.5, 0.25)
    def S1i(t): return 26.0 if t < 80 else _ramp(t, 80, 100, 26.0, 10.0)
    def lf(t):
        if t < 20: return 3
        if t < 60: return 4
        return 5
    r1 = solve_am2((0, 80),   y0, D, S1i, alk_in_mmol=18.0, label_func=lf, label_map=LABEL_MAP)
    y1 = [float(r1.S1[-1]), float(r1.X1[-1]), float(r1.S2[-1]), float(r1.X2[-1])]
    r2 = solve_am2((0, 60),   y1, D, S1i, alk_in_mmol=30.0, label_func=lf, label_map=LABEL_MAP)
    return _concat_results(r1, r2)

def _scenario_E(seed_offset=0):
    """Normal -> mild stress -> recovery (100 days)."""
    y0 = [5.0, 0.5, 2.0, 0.5]
    def S1i(t):
        if t < 30: return 10.0
        if t < 60: return 16.0
        return 10.0
    def lf(t):
        if t < 30: return 0
        if t < 50: return 1
        if t < 60: return 2
        return 5
    return solve_am2((0, 100), y0,
                     D_func=lambda t: 0.5,
                     S1_in_func=S1i,
                     alk_in_mmol=24.0,
                     label_func=lf,
                     label_map=LABEL_MAP)


def _concat_results(r1, r2):
    """Concatenate two AM2Result objects (r2 times offset so they are sequential)."""
    from src.am2_model import AM2Result
    offset = r1.t[-1]
    return AM2Result(
        t=np.concatenate([r1.t, r2.t + offset - r2.t[0] + 1]),
        S1=np.concatenate([r1.S1, r2.S1]),
        X1=np.concatenate([r1.X1, r2.X1]),
        S2=np.concatenate([r1.S2, r2.S2]),
        X2=np.concatenate([r1.X2, r2.X2]),
        q_CH4_mmol=np.concatenate([r1.q_CH4_mmol, r2.q_CH4_mmol]),
        q_CH4_Ld=np.concatenate([r1.q_CH4_Ld, r2.q_CH4_Ld]),
        vfa_mgl=np.concatenate([r1.vfa_mgl, r2.vfa_mgl]),
        alkalinity_mgl=np.concatenate([r1.alkalinity_mgl, r2.alkalinity_mgl]),
        pH=np.concatenate([r1.pH, r2.pH]),
        D=np.concatenate([r1.D, r2.D]),
        label=np.concatenate([r1.label, r2.label]),
        label_name=np.concatenate([r1.label_name, r2.label_name]),
    )


# ---------------------------------------------------------------------------
# Small helper: result -> DataFrame
# ---------------------------------------------------------------------------
def result_to_df(res, scenario_id: int = 0) -> pd.DataFrame:
    """Convert AM2Result to a tidy DataFrame. Adds day-index and FOS/TAC-like proxy."""
    df = pd.DataFrame({
        'day':             res.t,
        'S1_g_COD_L':      res.S1,
        'X1_g_VS_L':       res.X1,
        'S2_mmol_L':       res.S2,
        'X2_g_VS_L':       res.X2,
        'q_CH4_mmol_L_d':  res.q_CH4_mmol,
        'q_CH4_L_L_d':     res.q_CH4_Ld,
        'vfa_mgl':         res.vfa_mgl,
        'alkalinity_mgl':  res.alkalinity_mgl,
        'pH':              res.pH,
        'D_per_day':       res.D,
        'label':           res.label,
        'label_name':      res.label_name,
        'scenario_id':     scenario_id,
    })
    # FOS/TAC-like proxy: VFA (mg/L as acetic acid) / alkalinity (mg/L as CaCO3)
    # This is NOT real FOS/TAC (which uses titration). Labelled explicitly.
    df['sim_fos_tac_proxy'] = df['vfa_mgl'] / df['alkalinity_mgl'].clip(lower=1.0)
    return df


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------
def generate_all_scenarios() -> pd.DataFrame:
    """Run all five scenario functions and return a single labelled DataFrame."""
    print("Running AM2 scenarios…")
    scenarios = [
        (_scenario_A, 0, "A_normal"),
        (_scenario_B, 1, "B_overload_acid"),
        (_scenario_C, 2, "C_souring"),
        (_scenario_D, 3, "D_souring_recovery"),
        (_scenario_E, 4, "E_mild_stress"),
    ]
    parts = []
    for fn, sid, sname in scenarios:
        print(f"  {sname} … ", end='', flush=True)
        try:
            res = fn()
            df  = result_to_df(res, scenario_id=sid)
            df['scenario_name'] = sname
            parts.append(df)
            print(f"OK  ({len(df)} rows, labels={sorted(df.label.unique())})")
        except Exception as exc:
            print(f"FAILED – {exc}")
    combined = pd.concat(parts, ignore_index=True)
    print(f"\nTotal synthetic rows: {len(combined)}   "
          f"Class distribution:\n{combined.groupby(['label', 'label_name']).size()}")
    return combined


# ---------------------------------------------------------------------------
# Reproducible noise addition (simulates measurement error for ML features)
# ---------------------------------------------------------------------------
def add_measurement_noise(df: pd.DataFrame, seed=SEED) -> pd.DataFrame:
    """Add small Gaussian noise to state variables to approximate sensor noise."""
    rn = np.random.default_rng(seed)
    out = df.copy()
    out['pH']           += rn.normal(0, 0.05, len(out))
    out['pH']            = out['pH'].clip(4.5, 9.0)
    out['vfa_mgl']      *= (1 + rn.normal(0, 0.03, len(out)))
    out['alkalinity_mgl'] *= (1 + rn.normal(0, 0.02, len(out)))
    out['q_CH4_L_L_d']  *= (1 + rn.normal(0, 0.04, len(out)))
    return out


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def run(save: bool = True) -> pd.DataFrame:
    df = generate_all_scenarios()
    df = add_measurement_noise(df)
    df['sim_fos_tac_proxy'] = df['vfa_mgl'] / df['alkalinity_mgl'].clip(lower=1.0)
    if save:
        out = ROOT / 'data' / 'synthetic'
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'am2_synthetic_labelled.csv'
        df.to_csv(path, index=False)
        print(f"\nSaved -> {path}")
    return df


if __name__ == '__main__':
    run()
