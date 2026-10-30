"""Expanding-window (rolling-origin) evaluation: robust to the choice of a single split date."""
import sys; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from src.biogas_prediction import build_features, PILOT_BASE, FULL_BASE, TARGET, SEED
from src.preprocessing import TrainOnlyPreprocessor
T = {"pilot_D1": "pilot_D1", "pilot_D2": "pilot_D2", "full_scale": "full"}
rows = []
for n, f in T.items():
    t = pd.read_csv(ROOT/"data/processed"/f"{f}_harmonised.csv", parse_dates=["datetime"])
    base = FULL_BASE if n == "full_scale" else PILOT_BASE
    for ar in (False, True):
        d = build_features(t, base, ar); d = d[d[TARGET].notna()].reset_index(drop=True)
        feats = [c for c in d.columns if c not in ("datetime", TARGET)]
        N = len(d); edges = np.linspace(int(N*0.4), N, 6).astype(int)   # 5 expanding folds, each tests the next block
        for k in range(5):
            a, b = edges[k], edges[k+1]; gap = 7
            tr, te = d.iloc[:a-gap], d.iloc[a:b]
            pre = TrainOnlyPreprocessor().fit(tr[feats])
            res = {"persistence": None}
            full = t.set_index("datetime")[TARGET]; pers = full.shift(1).reindex(te.datetime).values; ok = ~np.isnan(pers)
            rows.append(dict(system=n, ar=ar, fold=k, model="persistence", MAE=mean_absolute_error(te[TARGET].values[ok], pers[ok]), test_start=te.datetime.min().date(), mean_train=tr[TARGET].mean(), mean_test=te[TARGET].mean()))
            rows.append(dict(system=n, ar=ar, fold=k, model="train-mean", MAE=mean_absolute_error(te[TARGET], np.full(len(te), tr[TARGET].mean())), test_start=te.datetime.min().date(), mean_train=tr[TARGET].mean(), mean_test=te[TARGET].mean()))
            for mn, mdl, sc in [("Ridge", RidgeCV(alphas=np.logspace(-2,3,12)), True), ("RandomForest", RandomForestRegressor(300, min_samples_leaf=3, random_state=SEED, n_jobs=-1), False)]:
                m = mdl.fit(pre.transform(tr[feats], scale=sc), tr[TARGET]); p = m.predict(pre.transform(te[feats], scale=sc))
                rows.append(dict(system=n, ar=ar, fold=k, model=mn, MAE=mean_absolute_error(te[TARGET], p), test_start=te.datetime.min().date(), mean_train=tr[TARGET].mean(), mean_test=te[TARGET].mean()))
            m = HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=200, random_state=SEED).fit(tr[feats], tr[TARGET]); p = m.predict(te[feats])
            rows.append(dict(system=n, ar=ar, fold=k, model="HistGB", MAE=mean_absolute_error(te[TARGET], p), test_start=te.datetime.min().date(), mean_train=tr[TARGET].mean(), mean_test=te[TARGET].mean()))
R = pd.DataFrame(rows); R.to_csv(ROOT/"reports/biogas_rolling_origin_folds.csv", index=False)
S = R.groupby(["system","ar","model"]).MAE.agg(["mean","std"]).round(3).unstack("model")
print(S.to_string())
print("\nlevel shift check (mean biogas train vs next test block), process regime:")
print(R[(R.model=="persistence")&(~R.ar)][["system","fold","test_start","mean_train","mean_test"]].round(3).to_string(index=False))
