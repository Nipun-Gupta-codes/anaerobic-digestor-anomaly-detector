"""Feature engineering. Every feature at row t uses information from rows <= t only (no look-ahead)."""
from __future__ import annotations
import numpy as np
import pandas as pd

def add_lags(df: pd.DataFrame, cols, lags=(1, 2, 3, 7)) -> pd.DataFrame:
    out = {}
    for c in cols:
        if c in df.columns:
            for l in lags: out[f"{c}_lag{l}"] = df[c].shift(l)
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)

def add_rolling(df: pd.DataFrame, cols, windows=(3, 7)) -> pd.DataFrame:
    """Trailing rolling mean/std. min_periods=2 so sparse lab series still produce values only when >=2 observations exist."""
    out = {}
    for c in cols:
        if c in df.columns:
            for w in windows:
                r = df[c].rolling(w, min_periods=2)
                out[f"{c}_rmean{w}"] = r.mean(); out[f"{c}_rstd{w}"] = r.std()
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)

def add_trends(df: pd.DataFrame, cols, window=7) -> pd.DataFrame:
    """Backward-looking slope (units/day) over `window` days via change between current and window-ago value (NaN-safe)."""
    out = {}
    for c in cols:
        if c in df.columns:
            out[f"{c}_trend{window}"] = (df[c] - df[c].shift(window)) / window
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["datetime"]
    out = df.copy()
    out["dayofweek"] = d.dt.dayofweek
    out["doy_sin"] = np.sin(2 * np.pi * d.dt.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * d.dt.dayofyear / 365.25)
    return out

def last_observed(df: pd.DataFrame, cols, max_age=7) -> pd.DataFrame:
    """Forward-carry the most recent *measured* lab value for at most max_age days, with an 'age' column.
    This reflects what an operator actually knows at time t (no interpolation toward future values)."""
    out = {}
    for c in cols:
        if c not in df.columns: continue
        s = df[c]; seen = s.notna()
        last = s.ffill(); grp = seen.cumsum()
        age = df.groupby(grp).cumcount().where(grp > 0)
        out[f"{c}_last"] = last.where(age <= max_age)
        out[f"{c}_age"] = age.where(age <= max_age)
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
