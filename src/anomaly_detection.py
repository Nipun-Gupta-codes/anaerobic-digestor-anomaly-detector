"""Unsupervised anomaly detection on REAL data (no verified souring labels exist).

Detectors are fitted on the training period only (first 60%, chronological) and score every row (novelty detection).
Two feature configurations:
  A 'logged'    : pH, temperature, biogas, CCR, (loading)  -- contains NO VFA/alkalinity, so chemistry is an independent check
  B 'chemistry' : A + VFA, alkalinity, VA/Alk              -- only on rows where chemistry was measured
'Chemical evidence of instability' is a PROXY, not ground truth: pilot VA/Alk > 0.30 or VFA > 500 mg/L or pH < 6.8.
Statistical outliers are not claimed to be souring.
"""
from __future__ import annotations
import sys; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_score, recall_score
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.preprocessing import TrainOnlyPreprocessor
SEED = 42

def evidence(t: pd.DataFrame, va_thr=0.30, vfa_thr=500, ph_thr=6.8) -> pd.Series:
    """1 = chemical evidence of instability, 0 = chemistry measured and normal, NaN = no chemistry measured that day."""
    known = t[["va_alk", "vfa_mgl", "ph"]].notna().any(axis=1)
    ev = ((t["va_alk"] > va_thr) | (t["vfa_mgl"] > vfa_thr) | (t["ph"] < ph_thr)).astype(float)
    return ev.where(known)

def make_detectors(contamination=0.05):
    return {
        "IsolationForest": IsolationForest(n_estimators=300, contamination=contamination, random_state=SEED),
        "LocalOutlierFactor": LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True),
        "OneClassSVM": OneClassSVM(kernel="rbf", gamma="scale", nu=contamination),
    }

def run(t: pd.DataFrame, feats: list[str], name: str, train_frac=0.6, contamination=0.05):
    d = t.dropna(subset=feats).reset_index(drop=True)
    ntr = int((t.datetime <= t.datetime.iloc[int(len(t) * train_frac)]).sum())
    cutoff = t.datetime.iloc[int(len(t) * train_frac)]
    tr_mask = d.datetime <= cutoff
    pre = TrainOnlyPreprocessor().fit(d.loc[tr_mask, feats])
    X = pre.transform(d[feats]); ev = evidence(d)
    out = d[["datetime"]].copy(); out["evidence"] = ev
    rows = []
    for dn, det in make_detectors(contamination).items():
        det.fit(X[tr_mask])
        score = -det.score_samples(X)            # higher = more anomalous
        flag = (det.predict(X) == -1).astype(int)
        out[f"{dn}_score"] = score; out[f"{dn}_flag"] = flag
        for part, mk in [("train", tr_mask), ("holdout", ~tr_mask)]:
            m = mk & ev.notna(); n_ev = int(ev[m].sum())
            r = dict(system=name, features=",".join(feats) if len(feats) < 5 else f"{len(feats)} feats", detector=dn, part=part, n_rows=int(mk.sum()),
                     n_flagged=int(flag[mk].sum()), n_with_chem=int(m.sum()), n_evidence_days=n_ev)
            if n_ev > 0 and n_ev < m.sum():
                r.update(auc=roc_auc_score(ev[m], score[m]), precision=precision_score(ev[m], flag[m], zero_division=0), recall=recall_score(ev[m], flag[m], zero_division=0))
            rows.append(r)
    out["train_cutoff"] = cutoff
    return out, pd.DataFrame(rows)
