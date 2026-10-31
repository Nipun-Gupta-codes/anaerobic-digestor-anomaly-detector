"""
Early Warning System for Anaerobic Digester Souring
====================================================
Monitors predicted stability indicators (VA/Alk or simulated FOS/TAC proxy)
and generates structured warnings with contributing variable explanations.

FOS/TAC thresholds (project-specific; based on project synopsis and common literature guidance):
  < 0.23           : Underfed / low VFA loading
  0.23 – 0.30      : Optimal stability zone
  0.30 – 0.40      : Early warning
  0.40 – 1.00      : High risk
  > 1.00           : Severe / fatal souring

IMPORTANT: These thresholds are defined for TITRATION-BASED FOS/TAC.
The VA/Alk ratio from real data is related but NOT equivalent.
All warnings based on real VA/Alk are labelled accordingly.
See: Lossie, U., & Pütz, P. (2008). Targeted control of biogas plants with FOS/TAC.
     Practice Report. HACH LANGE, Düsseldorf.

Suggested corrective actions are SIMULATED responses, not validated industrial protocols.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------
# Warning thresholds (project-specific)
# --------------------------------------------------------------------------
THRESHOLDS = {
    'underfed':   (None, 0.23),
    'optimal':    (0.23, 0.30),
    'warning':    (0.30, 0.40),
    'high_risk':  (0.40, 1.00),
    'souring':    (1.00, None),
}

LEVEL_NAMES = {
    'underfed':  'INFO️ Underfed',
    'optimal':   '[OK] Optimal Stability',
    'warning':   'WARNING️ Early Warning',
    'high_risk': '🔴 High Risk',
    'souring':   '🚨 Severe Souring',
}

SUGGESTED_ACTIONS = {
    'underfed': [
        "Consider increasing organic feed rate moderately.",
        "Monitor VFA to confirm low-loading condition.",
        "Check feed composition for sufficient COD loading.",
    ],
    'optimal': [
        "Continue current operating conditions.",
        "Maintain regular monitoring of VA/Alk or FOS/TAC.",
    ],
    'warning': [
        "Reduce organic feed rate by 10–20% (simulation only).",
        "Increase monitoring frequency of VFA and alkalinity.",
        "Check feed composition for inhibitory compounds.",
        "Consider reducing dilution rate to stabilise residence time.",
    ],
    'high_risk': [
        "Reduce organic loading rate significantly (20–40% reduction).",
        "Add bicarbonate or lime for alkalinity supplementation (simulate NaHCO3 dosing).",
        "Increase hydraulic retention time by reducing feed volume.",
        "Halt addition of inhibitory co-substrates (FOG, etc.).",
        "Increase monitoring frequency to at least twice daily.",
    ],
    'souring': [
        "URGENT: Reduce or halt organic loading immediately.",
        "Dose alkalinity (bicarbonate, Na2CO3) to raise buffering capacity.",
        "Consider reactor bypass or partial dilution with fresh water.",
        "Consult process specialist before resuming normal operation.",
        "Monitor pH, VFA, alkalinity every 4–6 hours.",
    ],
}


# --------------------------------------------------------------------------
# Classify a single stability indicator value
# --------------------------------------------------------------------------
def classify_ratio(ratio: float) -> str:
    """Return the warning category name for a given stability ratio."""
    if np.isnan(ratio):
        return 'unknown'
    if ratio < 0.23:
        return 'underfed'
    if ratio < 0.30:
        return 'optimal'
    if ratio < 0.40:
        return 'warning'
    if ratio < 1.00:
        return 'high_risk'
    return 'souring'


@dataclass
class Warning:
    """Structured early warning output."""
    timestamp: str
    ratio: float
    category: str
    level_name: str
    contributing_variables: dict = field(default_factory=dict)
    suggested_actions: list[str] = field(default_factory=list)
    data_source: str = ''       # 'real' or 'synthetic'
    target_label: str = ''      # what the ratio actually is
    note: str = ''


def build_warning(
    ratio: float,
    timestamp,
    context: dict = None,
    data_source: str = 'real',
    target_label: str = 'VA/Alk ratio (real; NOT FOS/TAC)',
) -> Warning:
    """
    Build a Warning object for a given ratio value and context variables.

    Parameters
    ----------
    ratio       : stability ratio value (VA/Alk or sim FOS/TAC proxy)
    timestamp   : observation date/time or day index
    context     : dict of variable_name -> value (e.g. {'ph': 6.8, 'vfa_mgl': 800})
    data_source : 'real' or 'synthetic'
    target_label: human-readable label for the ratio
    """
    if context is None:
        context = {}
    cat   = classify_ratio(ratio)
    level = LEVEL_NAMES.get(cat, 'Unknown')
    actions = SUGGESTED_ACTIONS.get(cat, [])

    # Identify contributing variables that suggest instability
    contrib = {}
    if 'vfa_mgl' in context and not np.isnan(context.get('vfa_mgl', np.nan)):
        v = context['vfa_mgl']
        contrib['VFA (mg/L)'] = f"{v:.1f}" + (' WARNING️ elevated' if v > 500 else '')
    if 'ph' in context and not np.isnan(context.get('ph', np.nan)):
        p = context['ph']
        contrib['pH'] = f"{p:.2f}" + (' WARNING️ low' if p < 6.8 else '')
    if 'alkalinity_mgl' in context and not np.isnan(context.get('alkalinity_mgl', np.nan)):
        a = context['alkalinity_mgl']
        contrib['Alkalinity (mg/L)'] = f"{a:.1f}" + (' WARNING️ low buffer' if a < 1000 else '')
    if 'biogas_m3d' in context and not np.isnan(context.get('biogas_m3d', np.nan)):
        b = context['biogas_m3d']
        contrib['Biogas (m³/day)'] = f"{b:.2f}"
    if 'q_CH4_L_L_d' in context:
        contrib['CH4 rate (sim, L/L/d)'] = f"{context['q_CH4_L_L_d']:.4f}"
    if 'ccr' in context and not np.isnan(context.get('ccr', np.nan)):
        contrib['CCR'] = f"{context['ccr']:.2f}"

    note = (
        "WARNING️ Thresholds are project-specific and defined for FOS/TAC (titration-based). "
        "This ratio is VA/Alk (mass/mass) – related but not identical. "
        "Treat as indicative only."
        if data_source == 'real' else
        "INFO️ Based on simulated AM2 model data. Thresholds applied to a synthetic FOS/TAC-like "
        "proxy (VFA mg/L / alkalinity mg/L). NOT a real plant measurement."
    )

    return Warning(
        timestamp=str(timestamp),
        ratio=float(ratio),
        category=cat,
        level_name=level,
        contributing_variables=contrib,
        suggested_actions=actions,
        data_source=data_source,
        target_label=target_label,
        note=note,
    )


# --------------------------------------------------------------------------
# Batch warning scan
# --------------------------------------------------------------------------
def scan_warnings(
    df: pd.DataFrame,
    ratio_col: str,
    timestamp_col: str = 'datetime',
    data_source: str = 'real',
    target_label: str = 'VA/Alk (real)',
) -> list[Warning]:
    """
    Scan a DataFrame for warning conditions.

    Returns a list of Warning objects for every row where ratio is not NaN.
    """
    warnings = []
    for _, row in df.iterrows():
        r = row.get(ratio_col, np.nan)
        if pd.isna(r):
            continue
        ts  = row.get(timestamp_col, 'unknown')
        ctx = {c: row.get(c, np.nan) for c in
               ['vfa_mgl', 'ph', 'alkalinity_mgl', 'biogas_m3d',
                'q_CH4_L_L_d', 'ccr', 'va_alk', 'sim_fos_tac_proxy']}
        warnings.append(build_warning(r, ts, ctx, data_source, target_label))
    return warnings


def warnings_to_df(warnings: list[Warning]) -> pd.DataFrame:
    """Convert list of Warning objects to a summary DataFrame."""
    rows = []
    for w in warnings:
        rows.append({
            'timestamp':  w.timestamp,
            'ratio':      w.ratio,
            'category':   w.category,
            'level':      w.level_name,
            'source':     w.data_source,
            'target':     w.target_label,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Add warning categories to any DataFrame with a ratio column
# --------------------------------------------------------------------------
def add_warning_column(df: pd.DataFrame, ratio_col: str,
                        new_col: str = 'warning_category') -> pd.DataFrame:
    df = df.copy()
    df[new_col] = df[ratio_col].apply(classify_ratio)
    return df


# --------------------------------------------------------------------------
# Predicted-future warning (post-LSTM)
# --------------------------------------------------------------------------
def forecast_warning(
    predicted_ratio: float,
    horizon_days: int,
    current_context: dict = None,
    data_source: str = 'synthetic',
    target_label: str = 'Predicted sim FOS/TAC proxy',
) -> Warning:
    """
    Generate a warning from a model-predicted future stability ratio.

    Parameters
    ----------
    predicted_ratio : model output for t + horizon_days
    horizon_days    : forecast horizon (1, 3, or 7 days)
    current_context : current measured/simulated values for context
    """
    w = build_warning(
        ratio=predicted_ratio,
        timestamp=f'Forecast (t+{horizon_days}d)',
        context=current_context or {},
        data_source=data_source,
        target_label=target_label,
    )
    w.note = (
        f"PREDICTED value at t+{horizon_days} days via ML model. "
        + w.note
    )
    return w


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # Demo on pilot D1 harmonised data
    path = ROOT / 'data' / 'processed' / 'pilot_D1_harmonised.csv'
    if path.exists():
        df = pd.read_csv(path, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
        df = add_warning_column(df, 'va_alk', 'warning_category')
        print("Pilot D1 warning distribution:\n", df['warning_category'].value_counts())

        ws = scan_warnings(df, 'va_alk', data_source='real',
                           target_label='VA/Alk ratio (real; NOT FOS/TAC)')
        wdf = warnings_to_df(ws)
        non_ok = wdf[wdf['category'].isin(['warning', 'high_risk', 'souring'])]
        print(f"\n{len(non_ok)} rows with warning-level or higher VA/Alk ratio:")
        print(non_ok.head(10).to_string(index=False))
        wdf.to_csv(ROOT / 'reports' / 'early_warnings_pilot_D1.csv', index=False)
        print("\nSaved -> reports/early_warnings_pilot_D1.csv")
    else:
        print("pilot_D1_harmonised.csv not found.")
