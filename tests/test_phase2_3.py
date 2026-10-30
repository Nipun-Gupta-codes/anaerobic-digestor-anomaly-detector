import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src.preprocessing import chrono_split, TrainOnlyPreprocessor, bridge_short_gaps, pilot_reactor
from src.feature_engineering import add_lags, add_rolling, last_observed
from src.data_loader import load_dataset

def _df(n=100):
    return pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=n), "x": np.arange(n, dtype=float)})

def test_chrono_split_order_and_gap():
    sp = chrono_split(_df(), gap=5)
    assert sp.train.datetime.max() < sp.val.datetime.min() < sp.test.datetime.min()
    assert (sp.val.datetime.min() - sp.train.datetime.max()).days > 5

def test_preprocessor_uses_train_only():
    tr = pd.DataFrame({"a": [1., 2., 3.]}); te = pd.DataFrame({"a": [1000.]})
    p = TrainOnlyPreprocessor().fit(tr)
    assert abs(p.mu_["a"] - 2.0) < 1e-9          # unaffected by the test value
    assert np.isfinite(p.transform(pd.DataFrame({"a": [np.nan]}))["a"].iloc[0])

def test_all_nan_training_column_is_safe():
    tr = pd.DataFrame({"a": [1., 2., 3.], "b": [np.nan] * 3})
    out = TrainOnlyPreprocessor().fit_transform(tr)
    assert np.isfinite(out.values).all()

def test_features_are_backward_looking():
    d = add_lags(_df(), ["x"], lags=(1,)); assert d.x_lag1.iloc[5] == 4
    r = add_rolling(_df(), ["x"], windows=(3,)); assert r.x_rmean3.iloc[10] == np.mean([8, 9, 10])
    # changing a FUTURE value must not change earlier features
    d2 = _df(); d2.loc[50, "x"] = -999; r2 = add_rolling(d2, ["x"], windows=(3,))
    assert r2.x_rmean3.iloc[:50].equals(r.x_rmean3.iloc[:50])

def test_last_observed_age_limit():
    d = _df(20); d["lab"] = np.nan; d.loc[2, "lab"] = 5.0
    o = last_observed(d, ["lab"], max_age=3)
    assert o.lab_last.iloc[4] == 5.0 and np.isnan(o.lab_last.iloc[10])

def test_bridge_gap_limit():
    s = pd.Series([1, np.nan, 3, np.nan, np.nan, np.nan, 7.0]); b = bridge_short_gaps(s, 2)
    assert b.iloc[1] == 2                      # 1-day gap bridged
    assert b.iloc[3:6].isna().all()            # 3-day gap left completely empty

def test_pilot_flags_keep_raw_values():
    df = load_dataset("pilot"); t = pilot_reactor(df, 1)
    assert t.flag_alk_suspect.sum() > 0 and t.alk_mgl.isna().sum() >= t.flag_alk_suspect.sum()
    assert t.alk_mgl_raw.notna().sum() >= df.PD1_alkalinity_mgl.notna().sum()
