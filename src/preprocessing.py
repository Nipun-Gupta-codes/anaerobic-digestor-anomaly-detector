"""Preprocessing: unit harmonisation, per-reactor tables, leakage-safe scaling and chronological splits.

Design rules (see reports/phase1_data_audit.md):
* No dataset is concatenated with another. Pilot D1 and D2 are separate reactors; full-scale is its own system.
* Lab variables (VFA, alkalinity, pH, COD, VS) are NEVER interpolated. Only continuously logged variables
  (gas, temperature, CCR) may be bridged, and only across short gaps (<= MAX_FILL_DAYS).
* VA/Alk is recomputed from VA and alkalinity (full-scale reported ratio is rounded to 2 decimals).
* Outliers are flagged, never deleted.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

CF_TO_M3 = 0.028316846592
MAX_FILL_DAYS = 2
ALK_SUSPECT_BELOW = 500.0   # mg/L. Alkalinity this low is incompatible with the recorded pH ~7.5 and VFA ~100 mg/L
PH_SUSPECT_BELOW = 5.5      # su. Isolated reading simultaneous in both pilot digesters with normal VFA/alkalinity

def f_to_c(f): return (f - 32.0) * 5.0 / 9.0

# ---------------------------------------------------------------- pilot
PILOT_SHARED = [
    "fog_loading_rate_lb1000ft3xd", "food_loading_rate_lb1000ft3xd", "PS_loading_rate_lb1000ft3xd",
    "TWAS_loading_rate_lb1000ft3xd", "PD_target_organic_loading_rate_pct",
    "PD_organic_loading_rate_pct", "PD_organic_loading_rate_7dayavg_pct",
    "fog_feed_volume_gal", "food_feed_volume_gal", "PS_feed_volume_gal", "TWAS_feed_volume_gal",
]

def pilot_reactor(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Return a harmonised table for pilot digester `d` (1 or 2) with common column names."""
    if d not in (1, 2): raise ValueError("d must be 1 or 2")
    p = f"PD{d}_"
    need = [p + "ph_su", p + "volatile_acids_mgl", p + "alkalinity_mgl", p + "gas_generation_cfday",
            p + "temperature_degf", f"daily_ccr_AD{d}"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise KeyError(f"pilot columns missing: {miss}")
    out = pd.DataFrame({"datetime": df["datetime"]})
    out["ph_raw"] = df[p + "ph_su"]
    out["vfa_mgl"] = df[p + "volatile_acids_mgl"]
    out["alk_mgl_raw"] = df[p + "alkalinity_mgl"]
    # Quality flags: raw values are kept; flagged values are masked in the modelling columns (not deleted).
    out["flag_alk_suspect"] = (out["alk_mgl_raw"] < ALK_SUSPECT_BELOW).astype(int)
    out["flag_ph_suspect"] = (out["ph_raw"] < PH_SUSPECT_BELOW).astype(int)
    out["ph"] = out["ph_raw"].mask(out["flag_ph_suspect"] == 1)
    out["alk_mgl"] = out["alk_mgl_raw"].mask(out["flag_alk_suspect"] == 1)
    out["va_alk"] = out["vfa_mgl"] / out["alk_mgl"]
    out["temp_c"] = f_to_c(df[p + "temperature_degf"])
    out["biogas_m3d_raw"] = df[p + "gas_generation_cfday"] * CF_TO_M3
    # zero gas = start-up / documented meter faults (pilot comments), implausible at pH ~7.5 -> flag and mask
    out["flag_gas_zero"] = (out["biogas_m3d_raw"] == 0).astype(int)
    out["biogas_m3d"] = out["biogas_m3d_raw"].mask(out["flag_gas_zero"] == 1)
    out["methane_pct"] = df.get(p + "methane_pct")
    out["cod_mgl"] = df.get(p + "COD_mgl")
    out["vs_pct"] = df.get(p + "VS_pct")
    out["ts_pct"] = df.get(p + "TS_pct")
    out["ccr"] = df[f"daily_ccr_AD{d}"]
    out["ccr_influent"] = df.get("daily_ccr_AD2_Influent")
    for c in PILOT_SHARED:
        if c in df.columns: out[c] = df[c]
    out["system"] = f"pilot_D{d}"
    return out

# ---------------------------------------------------------------- full scale
def full_scale_reactor(df: pd.DataFrame) -> pd.DataFrame:
    need = ["ph_su", "volatile_acids_mgl", "alkalinity_mgl", "biogas_produced_avg_cfday",
            "temperature_deg_c", "daily_ccr_digester"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise KeyError(f"full-scale columns missing: {miss}")
    out = pd.DataFrame({"datetime": df["datetime"]})
    out["ph"] = df["ph_su"]
    out["vfa_mgl"] = df["volatile_acids_mgl"]
    out["alk_mgl"] = df["alkalinity_mgl"]
    out["va_alk"] = out["vfa_mgl"] / out["alk_mgl"]          # recomputed, NOT the rounded reported ratio
    out["temp_c"] = df["temperature_deg_c"]
    out["biogas_m3d"] = df["biogas_produced_avg_cfday"] * CF_TO_M3
    out["methane_m3min"] = df["methane_produced_avg_cfmin"] * CF_TO_M3
    out["ccr"] = df["daily_ccr_digester"]
    out["ccr_slurry"] = df["daily_ccr_slurry"]
    out["vs_pct"] = df["volatile_solids_pct"]
    out["hrt_d"] = df["detention_time_1_dig_in_service_days"]
    for c in ["primary_VS_loading_pct", "twas_VS_loading_pct", "external_organic_loading_pct",
              "primary_VS_loading_7dayavg_pct", "twas_VS_loading_7dayavg_pct",
              "external_organic_loading_7dayavg_pct", "total_feed_flow_mgd", "fog_flow_pct",
              "primary_flow_pct", "twas_flow_pct", "vslr_lbs_vs_cf_digester_volumeday",
              "volatile_solids_reduction_avg_pct"]:
        if c in df.columns: out[c] = df[c]
    out["system"] = "full_scale"
    return out

def bridge_short_gaps(s: pd.Series, limit: int = MAX_FILL_DAYS) -> pd.Series:
    """Linear interpolation ONLY across gaps of at most `limit` consecutive missing days.
    Longer gaps are left entirely NaN (pandas' own `limit=` would partially fill them)."""
    filled = s.interpolate(method="linear", limit_area="inside")
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    keep_nan = isna & (run_len > limit)
    return filled.mask(keep_nan)

def regularise_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Insert explicit NaN rows for missing calendar days so lags are true day-lags."""
    d = df.set_index("datetime")
    d = d[~d.index.duplicated()].asfreq("D")
    return d.rename_axis("datetime").reset_index()

def flag_outliers_iqr(s: pd.Series, k: float = 1.5, fit_idx=None) -> pd.Series:
    """IQR flag. Bounds learned on fit_idx (training rows) only when given."""
    ref = s.loc[fit_idx] if fit_idx is not None else s
    q1, q3 = ref.quantile([.25, .75]); i = q3 - q1
    return ((s < q1 - k * i) | (s > q3 + k * i)).astype(int)

# ---------------------------------------------------------------- splitting / scaling
@dataclass
class ChronoSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    def sizes(self): return len(self.train), len(self.val), len(self.test)

def chrono_split(df: pd.DataFrame, train=0.6, val=0.2, gap: int = 0) -> ChronoSplit:
    """Strictly chronological split. `gap` rows are dropped between blocks so that lag/rolling
    windows built from earlier rows cannot straddle a boundary."""
    if not df["datetime"].is_monotonic_increasing: raise ValueError("df must be sorted by datetime")
    n = len(df); a = int(n * train); b = int(n * (train + val))
    return ChronoSplit(df.iloc[:a].copy(), df.iloc[a + gap:b].copy(), df.iloc[b + gap:].copy())

class TrainOnlyPreprocessor:
    """Median imputation + standardisation, fitted on training data ONLY."""
    def __init__(self): self.med_ = None; self.mu_ = None; self.sd_ = None; self.cols_ = None
    def fit(self, X: pd.DataFrame):
        self.cols_ = list(X.columns)
        self.med_ = X.median().fillna(0.0)   # all-NaN training column -> 0
        Xi = X.fillna(self.med_)
        self.mu_ = Xi.mean(); self.sd_ = Xi.std().replace(0, 1.0).fillna(1.0)
        return self
    def transform(self, X: pd.DataFrame, scale=True) -> pd.DataFrame:
        Xi = X[self.cols_].fillna(self.med_).fillna(0.0)
        return (Xi - self.mu_) / self.sd_ if scale else Xi
    def fit_transform(self, X, scale=True): return self.fit(X).transform(X, scale)
