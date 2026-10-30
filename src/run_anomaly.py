import sys; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.anomaly_detection import run
T = {n: pd.read_csv(ROOT/"data/processed"/f"{f}_harmonised.csv", parse_dates=["datetime"]) for n, f in [("pilot_D1","pilot_D1"),("pilot_D2","pilot_D2"),("full_scale","full")]}
CFG = {"pilot": {"A_logged": ["ph","temp_c","biogas_m3d","ccr"], "B_chemistry": ["ph","vfa_mgl","alk_mgl","va_alk","biogas_m3d","ccr"]},
       "full_scale": {"A_logged": ["temp_c","biogas_m3d","ccr","ccr_slurry"], "B_chemistry": ["vfa_mgl","alk_mgl","va_alk","biogas_m3d","ccr"]}}
allm, store = [], {}
for n, t in T.items():
    for cn, feats in CFG["full_scale" if n == "full_scale" else "pilot"].items():
        out, m = run(t, feats, f"{n}|{cn}"); allm.append(m); store[(n, cn)] = out
        out.to_csv(ROOT/f"data/processed/anomaly_scores_{n}_{cn}.csv", index=False)
M = pd.concat(allm).round(3); M.to_csv(ROOT/"reports/anomaly_detection_results.csv", index=False)
print(M[["system","detector","part","n_rows","n_flagged","n_with_chem","n_evidence_days","auc","precision","recall"]].to_string(index=False))
# figure: pilot isolation-forest scores vs chemical evidence
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, n in zip(axes, ["pilot_D1", "pilot_D2"]):
    o = store[(n, "A_logged")]; t = T[n]
    ax.plot(o.datetime, o.IsolationForest_score, lw=.8, label="Isolation Forest score (pH, temp, biogas, CCR only)")
    ev = o[o.evidence == 1]; ax.scatter(ev.datetime, ev.IsolationForest_score, c="r", s=18, label="chemical evidence (VA/Alk>0.3, VFA>500 mg/L, or pH<6.8)", zorder=3)
    ax.axvline(o.train_cutoff.iloc[0], c="k", ls="--", lw=.8, label="end of training period"); ax.set_ylabel("Anomaly score (higher = more unusual)"); ax.set_title(f"{n}: detector fitted on training period without any chemistry"); ax.legend(fontsize=7)
axes[-1].set_xlabel("Date"); fig.tight_layout(); fig.savefig(ROOT/"reports/figures/anomaly_pilot_scores.png"); plt.close(fig)
# what distinguishes top-flagged full-scale days
o = store[("full_scale","A_logged")]; t = T["full_scale"].merge(o[["datetime","IsolationForest_score","IsolationForest_flag"]], on="datetime")
print("\nfull-scale A_logged: flagged days", int(t.IsolationForest_flag.sum()), "| flagged mean vs others:")
print(t.groupby("IsolationForest_flag")[["biogas_m3d","ccr","temp_c","va_alk","vfa_mgl","ph"]].mean().round(3).to_string())
print("full-scale flagged date range by month:\n", t[t.IsolationForest_flag==1].datetime.dt.to_period("M").value_counts().sort_index().to_string())
