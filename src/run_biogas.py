import sys; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.biogas_prediction import *
T = {n: pd.read_csv(ROOT/"data/processed"/f"{f}_harmonised.csv", parse_dates=["datetime"]) for n, f in [("pilot_D1","pilot_D1"),("pilot_D2","pilot_D2"),("full_scale","full")]}
res, IMP = [], {}
for ar in (False, True):
    for n, t in T.items():
        r, imps, sp = run_system(n, t, FULL_BASE if n == "full_scale" else PILOT_BASE, ar)
        res.append(r); IMP[(n, ar)] = imps
        print(n, "AR" if ar else "process", "split sizes", sp.sizes(), sp.train.datetime.min().date(), sp.train.datetime.max().date(), "|", sp.test.datetime.min().date(), sp.test.datetime.max().date())
R = pd.concat(res).drop_duplicates(subset=["system","regime","model"]).round(4)
R.to_csv(ROOT/"reports/biogas_prediction_results.csv", index=False)
print(R.to_string(index=False))
for (n, ar), imps in IMP.items():
    if ar: continue
    imp = pd.DataFrame({"rf_impurity": imps["rf_impurity"], "rf_permutation_test_dMAE": imps["rf_permutation_test"]}).sort_values("rf_impurity", ascending=False)
    imp.to_csv(ROOT/f"reports/biogas_feature_importance_{n}.csv"); print("\n", n, "top features (process regime)"); print(imp.head(8).round(4).to_string())
    fig, ax = plt.subplots(figsize=(7,4)); top = imp.head(10)[::-1]; ax.barh(top.index, top.rf_impurity); ax.set_xlabel("Random Forest impurity importance (association, not causation)"); ax.set_title(f"{n}: top biogas predictors (no gas history)"); fig.tight_layout(); fig.savefig(ROOT/f"reports/figures/fi_biogas_{n}.png"); plt.close(fig)
    p = imps["_pred"]; fig, ax = plt.subplots(figsize=(10,3.5)); ax.plot(p.datetime, p.obs, label="observed"); ax.plot(p.datetime, p.pred, label="RF prediction (test period)"); ax.set_xlabel("Date"); ax.set_ylabel("Biogas (m³/day)"); ax.set_title(f"{n}: held-out biogas prediction (chronological test block)"); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/f"reports/figures/pred_biogas_{n}.png"); plt.close(fig)
