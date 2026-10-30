"""Real-data biogas prediction with chronological train/val/test and train-only preprocessing.

Two feature regimes are evaluated per system:
  'process'  : same-day operating variables + trailing statistics (NO gas history) -> what drives gas
  'process+ar': additionally yesterday's gas (autoregressive) -> compared against persistence
XGBoost is not installed in this environment; HistGradientBoostingRegressor (sklearn) is used as a
gradient-boosting stand-in and is labelled as such.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.preprocessing import chrono_split, TrainOnlyPreprocessor
from src.feature_engineering import add_rolling, last_observed
SEED = 42
TARGET = "biogas_m3d"

PILOT_BASE = ["ph", "temp_c", "ccr", "ccr_influent", "vs_pct", "cod_mgl",
              "fog_loading_rate_lb1000ft3xd", "food_loading_rate_lb1000ft3xd", "PS_loading_rate_lb1000ft3xd",
              "TWAS_loading_rate_lb1000ft3xd", "PD_organic_loading_rate_pct", "PD_organic_loading_rate_7dayavg_pct"]
FULL_BASE = ["temp_c", "ccr", "ccr_slurry", "hrt_d", "ph", "vs_pct", "primary_VS_loading_pct", "twas_VS_loading_pct",
             "external_organic_loading_pct", "primary_VS_loading_7dayavg_pct", "external_organic_loading_7dayavg_pct",
             "total_feed_flow_mgd", "fog_flow_pct"]

def build_features(t: pd.DataFrame, base: list[str], ar: bool) -> pd.DataFrame:
    base = [c for c in base if c in t.columns and t[c].notna().sum() > 0.2 * len(t)]
    d = t.copy()
    d = last_observed(d, ["vfa_mgl", "alk_mgl", "va_alk", "ph", "vs_pct", "cod_mgl"], max_age=7)
    d = add_rolling(d, ["ccr"], windows=(7,))
    feats = base + [c for c in d.columns if c.endswith(("_last", "_age")) or c.startswith("ccr_rmean")]
    # 'ph'/'vs_pct'/'cod_mgl' appear both raw and as *_last; keep the raw version only where it is dense
    feats = list(dict.fromkeys(feats))
    if ar:
        d["biogas_lag1"] = d[TARGET].shift(1); feats.append("biogas_lag1")
    keep = [f for f in feats if d[f].notna().sum() > 0.2 * len(d)]
    return d[["datetime", TARGET] + keep]

def metrics(y, p): return dict(MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), R2=r2_score(y, p))

def model_zoo():
    return {
        "LinearRegression": (lambda: LinearRegression(), True),
        "RandomForest": (lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1), False),
        "GradientBoosting": (lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=.05, subsample=.8, random_state=SEED), False),
        "HistGB (XGBoost stand-in)": (lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=200, random_state=SEED), None),
    }

def run_system(name: str, t: pd.DataFrame, base: list[str], ar: bool, gap=7):
    d = build_features(t, base, ar)
    d = d[d[TARGET].notna()].reset_index(drop=True)
    sp = chrono_split(d, 0.6, 0.2, gap=gap)
    feats = [c for c in d.columns if c not in ("datetime", TARGET)]
    rows, imps = [], {}
    # persistence baseline (yesterday's value) on real consecutive days only
    full = t.set_index("datetime")[TARGET]
    test_idx = sp.test["datetime"]
    pers = full.shift(1).reindex(test_idx).values; ytest = sp.test[TARGET].values; ok = ~np.isnan(pers)
    rows.append(dict(system=name, regime="baseline", model="Persistence (y[t-1])", n_train=len(sp.train), n_test=int(ok.sum()), **metrics(ytest[ok], pers[ok])))
    mean_pred = np.full(len(ytest), sp.train[TARGET].mean())
    rows.append(dict(system=name, regime="baseline", model="Train-mean", n_train=len(sp.train), n_test=len(ytest), **metrics(ytest, mean_pred)))
    pre = TrainOnlyPreprocessor().fit(sp.train[feats])
    for mname, (mk, needs_scale) in model_zoo().items():
        if mname.startswith("HistGB"):   # handles NaN natively
            Xtr, Xte = sp.train[feats], sp.test[feats]
        else:
            Xtr, Xte = pre.transform(sp.train[feats], scale=bool(needs_scale)), pre.transform(sp.test[feats], scale=bool(needs_scale))
        m = mk().fit(Xtr, sp.train[TARGET])
        p = m.predict(Xte)
        rows.append(dict(system=name, regime="process+ar" if ar else "process", model=mname, n_train=len(sp.train), n_test=len(sp.test), **metrics(sp.test[TARGET], p)))
        if mname == "RandomForest":
            imps["rf_impurity"] = pd.Series(m.feature_importances_, index=feats)
            pi = permutation_importance(m, Xte, sp.test[TARGET], n_repeats=20, random_state=SEED, scoring="neg_mean_absolute_error")
            imps["rf_permutation_test"] = pd.Series(pi.importances_mean, index=feats)
            imps["_pred"] = pd.DataFrame({"datetime": sp.test["datetime"].values, "obs": sp.test[TARGET].values, "pred": p})
    return pd.DataFrame(rows), imps, sp
