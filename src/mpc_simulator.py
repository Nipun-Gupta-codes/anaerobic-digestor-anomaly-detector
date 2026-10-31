"""
Simplified Model Predictive Control (MPC) Demonstration
=========================================================
This module demonstrates how predictive modelling could support anaerobic digester
process optimisation. It is a SIMULATION ONLY.

DISCLAIMER: This is NOT an industrially validated controller. It has NOT been tested
on a real digester. The purpose is to demonstrate the MPC concept in the context of
this final-year project.

Approach
--------
1. Use the AM2 ODE model as a "plant simulator" (proxy for the real reactor).
2. Define a simplified objective function that penalises:
   - High VFA accumulation (instability indicator)
   - High FOS/TAC proxy
   - Deviation from target methane production rate
   - Control action fluctuations (move suppression)
3. At each control step, optimise the feed rate (D) using scipy.optimize.minimize
   to minimise the objective over a short prediction horizon.
4. Apply the first optimal action (receding-horizon principle).

This is a steady-state / single-step simplified MPC for illustration.
A production controller would use a rolling Kalman filter, state estimator, and
multi-step nonlinear optimisation (NMPC).

References
----------
Méndez-Acosta, H. O. et al. (2010). A robust control scheme to improve the stability
of anaerobic digestion processes. Journal of Process Control, 20(4), 375-383.
https://doi.org/10.1016/j.jprocont.2010.01.006

Bernard, O. et al. (2001). Dynamical model development and parameter identification
for an anaerobic wastewater treatment process. Biotechnology & Bioengineering, 75(4).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.am2_model import solve_am2, DEFAULT_PARAMS, mu2


# --------------------------------------------------------------------------
# Objective function
# --------------------------------------------------------------------------
def mpc_objective(
    D_candidate: np.ndarray,
    y_current: np.ndarray,
    S1_in: float,
    alk_in_mmol: float,
    params: dict,
    target_ch4: float = 0.015,  # L CH4/L/day – target methane rate
    w_vfa: float = 1.0,
    w_ch4: float = 0.5,
    w_move: float = 0.2,
    D_prev: float = 0.5,
    horizon: int = 5,
) -> float:
    """
    Objective penalising:
      w_vfa  * mean(S2/K_I)          – VFA relative to inhibition threshold
      w_ch4  * (q_CH4 - target)^2   – deviation from desired methane rate
      w_move * (D - D_prev)^2       – move suppression (avoid large changes)
    """
    D_val = float(D_candidate[0])
    D_val = max(0.05, min(D_val, 1.5))   # physical bounds

    try:
        res = solve_am2(
            t_span=(0, horizon),
            y0=list(y_current),
            D_func=lambda t: D_val,
            S1_in_func=lambda t: S1_in,
            alk_in_mmol=alk_in_mmol,
            params=params,
            t_eval=np.arange(0, horizon + 1, 1.0),
        )
        vfa_pen  = w_vfa  * np.mean(res.S2 / params.get('K_I', 256.0))
        ch4_pen  = w_ch4  * np.mean((res.q_CH4_Ld - target_ch4) ** 2)
        move_pen = w_move * (D_val - D_prev) ** 2
        return float(vfa_pen + ch4_pen + move_pen)
    except Exception:
        return 1e6   # penalise infeasible point


# --------------------------------------------------------------------------
# Single-step MPC action
# --------------------------------------------------------------------------
def mpc_step(
    y_current: np.ndarray,
    S1_in: float,
    alk_in_mmol: float,
    D_prev: float,
    params: dict = None,
    D_bounds: tuple = (0.05, 1.2),
    target_ch4: float = 0.015,
    w_vfa: float = 1.0,
    w_ch4: float = 0.5,
    w_move: float = 0.2,
    horizon: int = 5,
) -> dict:
    """
    Compute the optimal dilution rate for the next period.

    Returns dict with:
        D_opt       : optimal dilution rate (1/day)
        obj_value   : objective function value at optimum
        success     : bool
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    result = minimize(
        mpc_objective,
        x0=[D_prev],
        args=(y_current, S1_in, alk_in_mmol, params, target_ch4,
              w_vfa, w_ch4, w_move, D_prev, horizon),
        method='L-BFGS-B',
        bounds=[D_bounds],
        options=dict(maxiter=50, ftol=1e-8),
    )
    return dict(D_opt=float(result.x[0]), obj_value=float(result.fun), success=bool(result.success))


# --------------------------------------------------------------------------
# Closed-loop MPC simulation
# --------------------------------------------------------------------------
def run_mpc_simulation(
    t_total: int = 100,
    control_interval: int = 2,
    y0: list = None,
    S1_in_normal: float = 10.0,
    S1_in_overload: float = 25.0,
    overload_start: int = 30,
    overload_end: int = 60,
    alk_in_mmol: float = 25.0,
    params: dict = None,
    D_init: float = 0.5,
    target_ch4: float = 0.015,
    w_vfa: float = 1.5,
    w_ch4: float = 0.4,
    w_move: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a closed-loop MPC run vs. an uncontrolled (fixed-D) baseline.

    Returns (mpc_df, baseline_df) both with columns:
        day, S1, X1, S2, X2, q_CH4_Ld, vfa_mgl, alkalinity_mgl, pH,
        D_applied, sim_fos_tac_proxy
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if y0 is None:
        y0 = [5.0, 0.5, 2.0, 0.5]

    def _S1in(t):
        if overload_start <= t < overload_end:
            return S1_in_overload
        return S1_in_normal

    # --- Baseline (fixed D) ---
    bl_res = solve_am2(
        t_span=(0, t_total),
        y0=y0,
        D_func=lambda t: D_init,
        S1_in_func=_S1in,
        alk_in_mmol=alk_in_mmol,
        params=params,
        t_eval=np.arange(0, t_total + 1, 1.0),
    )
    baseline_df = _result_to_df(bl_res, D_init)

    # --- MPC closed loop ---
    mpc_rows = []
    y_state  = np.array(y0, dtype=float)
    D_prev   = D_init

    for t_start in range(0, t_total, control_interval):
        t_end = min(t_start + control_interval, t_total)
        S1_in = _S1in(t_start)

        # Compute optimal D for next interval
        step = mpc_step(
            y_current=y_state, S1_in=S1_in, alk_in_mmol=alk_in_mmol,
            D_prev=D_prev, params=params, target_ch4=target_ch4,
            w_vfa=w_vfa, w_ch4=w_ch4, w_move=w_move, horizon=control_interval,
        )
        D_applied = step['D_opt']

        # Simulate interval with D_applied
        res_int = solve_am2(
            t_span=(0, t_end - t_start),
            y0=list(y_state),
            D_func=lambda t: D_applied,
            S1_in_func=lambda t: S1_in,
            alk_in_mmol=alk_in_mmol,
            params=params,
            t_eval=np.arange(0, t_end - t_start + 1, 1.0),
        )
        # Collect rows
        for i, tt in enumerate(res_int.t):
            mpc_rows.append(dict(
                day=t_start + tt,
                S1=res_int.S1[i], X1=res_int.X1[i],
                S2=res_int.S2[i], X2=res_int.X2[i],
                q_CH4_Ld=res_int.q_CH4_Ld[i],
                vfa_mgl=res_int.vfa_mgl[i],
                alkalinity_mgl=res_int.alkalinity_mgl[i],
                pH=res_int.pH[i],
                D_applied=D_applied,
                sim_fos_tac_proxy=res_int.vfa_mgl[i] / max(res_int.alkalinity_mgl[i], 1.0),
            ))

        # Advance state to end of interval
        y_state = np.array([res_int.S1[-1], res_int.X1[-1],
                            res_int.S2[-1], res_int.X2[-1]])
        D_prev  = D_applied

    mpc_df = pd.DataFrame(mpc_rows)
    return mpc_df, baseline_df


def _result_to_df(res, D_val: float) -> pd.DataFrame:
    return pd.DataFrame({
        'day':             res.t,
        'S1':              res.S1,
        'X1':              res.X1,
        'S2':              res.S2,
        'X2':              res.X2,
        'q_CH4_Ld':        res.q_CH4_Ld,
        'vfa_mgl':         res.vfa_mgl,
        'alkalinity_mgl':  res.alkalinity_mgl,
        'pH':              res.pH,
        'D_applied':       D_val,
        'sim_fos_tac_proxy': res.vfa_mgl / res.alkalinity_mgl.clip(1.0),
    })


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == '__main__':
    print("Running MPC demonstration…")
    mpc_df, bl_df = run_mpc_simulation(t_total=100)
    out = ROOT / 'reports'
    mpc_df.to_csv(out / 'mpc_controlled.csv', index=False)
    bl_df.to_csv(out / 'mpc_baseline.csv', index=False)
    print(f"MPC simulation complete.")
    print(f"  Baseline max VFA:   {bl_df['vfa_mgl'].max():.1f} mg/L")
    print(f"  MPC max VFA:        {mpc_df['vfa_mgl'].max():.1f} mg/L")
    print(f"  Baseline mean FOS:  {bl_df['sim_fos_tac_proxy'].mean():.3f}")
    print(f"  MPC mean FOS:       {mpc_df['sim_fos_tac_proxy'].mean():.3f}")
    print("Saved: reports/mpc_controlled.csv, reports/mpc_baseline.csv")

