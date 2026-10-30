"""Phase 1 data audit: per-dataset schema, missingness, continuity and key-variable availability."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_all

KEY_VARS = {
    "pilot": {
        "pH": ["PD1_ph_su", "PD2_ph_su"],
        "VFA (mg/L)": ["PD1_volatile_acids_mgl", "PD2_volatile_acids_mgl"],
        "Alkalinity (mg/L)": ["PD1_alkalinity_mgl", "PD2_alkalinity_mgl"],
        "VA/Alk ratio": ["PD1_va_alk_ratio_mgl", "PD2_va_alk_ratio_mgl"],
        "Organic loading rate (%)": ["PD_organic_loading_rate_pct", "PD_organic_loading_rate_7dayavg_pct"],
        "Temperature (degF)": ["PD1_temperature_degf", "PD2_temperature_degf"],
        "COD (mg/L)": ["PD1_COD_mgl", "PD2_COD_mgl", "PD1_sCOD_mgl", "PD2_sCOD_mgl"],
        "VS (%)": ["PD1_VS_pct", "PD2_VS_pct"],
        "CCR": ["daily_ccr_AD1", "daily_ccr_AD2", "daily_ccr_AD2_Influent"],
        "Biogas (cf/day)": ["PD1_gas_generation_cfday", "PD2_gas_generation_cfday"],
        "Methane (%)": ["PD1_methane_pct", "PD2_methane_pct"],
        "HRT": [],
    },
    "full_digester": {
        "pH": ["ph_su"], "VFA (mg/L)": ["volatile_acids_mgl"], "Alkalinity (mg/L)": ["alkalinity_mgl"],
        "VA/Alk ratio": ["va_alk_ratio"],
        "Organic loading rate (%)": ["primary_VS_loading_pct", "twas_VS_loading_pct", "external_organic_loading_pct"],
        "Temperature (degC)": ["temperature_deg_c"], "COD (mg/kg)": ["waste_provider_cod_wet_weight_mgkg"],
        "VS (%)": ["volatile_solids_pct"], "CCR": ["daily_ccr_digester", "daily_ccr_slurry"],
        "Biogas (cf/day)": ["biogas_produced_avg_cfday"],
        "Methane (cf/day | cf/min)": ["methane_produced_total_cfday", "methane_produced_avg_cfmin"],
        "HRT (days)": ["detention_time_1_dig_in_service_days"],
    },
    "full_wwc": {
        "pH / VFA / Alk": [], "Biogas / methane": [], "CCR": ["daily_ccr_digester", "daily_ccr_slurry"],
        "Feed TS/VS (%)": ["AD_primary_sludge_VS_pct", "AD_twas_VS_pct", "AD_slurry_VS_pct"],
        "Feed flow (mgd)": ["AD_total_feed_flow_mgd"], "FOG load (lbs)": ["AD_FOG_VS_load_lbs"],
    },
}

def availability(df: pd.DataFrame, cols: list[str]) -> str:
    if not cols:
        return "NOT AVAILABLE"
    parts = []
    for c in cols:
        if c not in df.columns:
            parts.append(f"{c}: column missing")
        else:
            s = df[c]
            parts.append(f"{c}: n={s.notna().sum()} ({s.notna().mean()*100:.0f}%)")
    return "; ".join(parts)

def longest_gap(s: pd.Series) -> int:
    """Longest run of consecutive NaN."""
    m = s.isna().astype(int)
    runs = m.groupby((m != m.shift()).cumsum()).sum()
    return int(runs.max()) if len(runs) else 0

def main(out_dir: Path | None = None):
    out_dir = out_dir or Path(__file__).resolve().parents[1] / "reports"
    out_dir.mkdir(exist_ok=True)
    D = load_all()
    rows = []
    for name, df in D.items():
        for c in df.columns:
            s = df[c]
            num = pd.api.types.is_numeric_dtype(s)
            rows.append(dict(dataset=name, column=c, dtype=str(s.dtype), n_obs=int(s.notna().sum()),
                             pct_missing=round(s.isna().mean()*100, 1), n_unique=int(s.nunique()),
                             longest_nan_run=longest_gap(s),
                             min=s.min() if num else None, median=s.median() if num else None,
                             max=s.max() if num else None))
    pd.DataFrame(rows).to_csv(out_dir / "data_audit_columns.csv", index=False)
    for name, df in D.items():
        print(f"\n##### {name}: {df.shape}")
        for var, cols in KEY_VARS[name].items():
            print(f"  {var:28s} -> {availability(df, cols)}")

if __name__ == "__main__":
    main()
