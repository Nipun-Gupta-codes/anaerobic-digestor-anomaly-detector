"""
Supervised anomaly classification on AM2-generated synthetic data.
==================================================================
Trains and evaluates multiple classifiers on the 6-class labelled synthetic dataset.

Classes:
  0 – Normal
  1 – Overloading
  2 – Early Acidification
  3 – High VFA
  4 – Severe Souring
  5 – Recovery

Evaluation:
  - Accuracy, macro F1, per-class precision / recall / F1
  - Confusion matrix
  - Feature importance (RF impurity + permutation)

Important caveats:
  - Models trained on SYNTHETIC data; performance on real plant data is unknown.
  - No ground-truth labels exist in the real datasets.
  - A transfer experiment (applying the classifier to real pilot data) is conducted
    only for exploratory purposes and is clearly labelled as such.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.preprocessing import label_binarize
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import TrainOnlyPreprocessor

SEED = 42
LABEL_MAP = {
    0: 'Normal',
    1: 'Overloading',
    2: 'Early Acidification',
    3: 'High VFA',
    4: 'Severe Souring',
    5: 'Recovery',
}

# Try XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

FEATURE_COLS = [
    'S1_g_COD_L', 'X1_g_VS_L', 'S2_mmol_L', 'X2_g_VS_L',
    'q_CH4_mmol_L_d', 'q_CH4_L_L_d',
    'vfa_mgl', 'alkalinity_mgl', 'pH',
    'D_per_day', 'sim_fos_tac_proxy',
]


# --------------------------------------------------------------------------
# Load and split
# --------------------------------------------------------------------------
def load_synthetic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Use only columns that actually exist
    use = [c for c in FEATURE_COLS if c in df.columns]
    df['label'] = df['label'].astype(int)
    return df[use + ['label', 'scenario_id', 'day']].dropna(subset=use)


def chrono_split_synthetic(df: pd.DataFrame, train=0.60, val=0.20):
    """Chronological split within each scenario to avoid future leakage."""
    tr, va, te = [], [], []
    for sid in df['scenario_id'].unique():
        s = df[df['scenario_id'] == sid].sort_values('day').reset_index(drop=True)
        n = len(s)
        a = int(n * train)
        b = int(n * (train + val))
        tr.append(s.iloc[:a])
        va.append(s.iloc[a:b])
        te.append(s.iloc[b:])
    return (pd.concat(tr).reset_index(drop=True),
            pd.concat(va).reset_index(drop=True),
            pd.concat(te).reset_index(drop=True))


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------
def classifier_zoo():
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            class_weight='balanced', random_state=SEED, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=SEED),
    }
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, use_label_encoder=False,
            eval_metric='mlogloss', random_state=SEED)
    return models


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def per_class_report(y_true, y_pred, classes) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=list(classes.keys()),
                                 target_names=list(classes.values()),
                                 output_dict=True, zero_division=0)
    rows = []
    for k, v in rep.items():
        if isinstance(v, dict):
            rows.append({'class': k, **v})
    return pd.DataFrame(rows)


def roc_auc_multiclass(y_true, y_prob, classes):
    """One-vs-rest macro-average ROC-AUC."""
    y_bin = label_binarize(y_true, classes=list(classes.keys()))
    if y_bin.shape[1] == 1:
        return np.nan
    try:
        return float(roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro'))
    except Exception:
        return np.nan


# --------------------------------------------------------------------------
# Main training and evaluation
# --------------------------------------------------------------------------
def run_supervised_classification(save_models: bool = True) -> dict:
    syn_path = ROOT / 'data' / 'synthetic' / 'am2_synthetic_labelled.csv'
    if not syn_path.exists():
        raise FileNotFoundError(f"Synthetic data not found: {syn_path}. Run src/simulation.py first.")

    df = load_synthetic(syn_path)
    use_feats = [c for c in FEATURE_COLS if c in df.columns]
    print(f"Loaded {len(df)} synthetic rows. Features: {use_feats}")
    print(f"Class distribution:\n{df.groupby('label')['label'].count()}")

    train_df, val_df, test_df = chrono_split_synthetic(df)
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Fit scaler on training data only
    prep = TrainOnlyPreprocessor().fit(train_df[use_feats])
    X_tr = prep.transform(train_df[use_feats])
    X_te = prep.transform(test_df[use_feats])
    y_tr = train_df['label'].values
    y_te = test_df['label'].values

    results = {}
    models_dir = ROOT / 'models'
    models_dir.mkdir(exist_ok=True)

    for mname, clf in classifier_zoo().items():
        print(f"\n  Training {mname}…", end='', flush=True)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        print(f"  acc={acc:.3f}")

        # Per-class report
        classes_present = {k: v for k, v in LABEL_MAP.items() if k in np.unique(np.concatenate([y_tr, y_te]))}
        class_rep = per_class_report(y_te, y_pred, classes_present)

        # Confusion matrix
        cm = confusion_matrix(y_te, y_pred, labels=list(classes_present.keys()))

        # ROC-AUC
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_te)
            auc = roc_auc_multiclass(y_te, y_prob, classes_present)
        else:
            auc = np.nan

        # Feature importance
        fi_imp = None; fi_perm = None
        if hasattr(clf, 'feature_importances_'):
            fi_imp = pd.Series(clf.feature_importances_, index=use_feats).sort_values(ascending=False)
        try:
            pi = permutation_importance(clf, X_te, y_te, n_repeats=15,
                                         random_state=SEED, scoring='accuracy')
            fi_perm = pd.Series(pi.importances_mean, index=use_feats).sort_values(ascending=False)
        except Exception:
            pass

        # Save model
        if save_models:
            joblib.dump(clf, models_dir / f'clf_{mname.lower()}.pkl')

        results[mname] = dict(
            accuracy=acc, roc_auc=auc,
            class_report=class_rep, confusion_matrix=cm,
            fi_impurity=fi_imp, fi_permutation=fi_perm,
            y_test=y_te, y_pred=y_pred,
        )

        print(f"\n  {mname} confusion matrix:\n{cm}")
        print(f"\n  {mname} per-class report:\n{class_rep[['class','precision','recall','f1-score','support']].to_string(index=False)}")
        print(f"  ROC-AUC (OvR macro): {auc:.3f}")

    # Summary table
    summary = pd.DataFrame([
        {'model': k, 'accuracy': v['accuracy'], 'roc_auc': v['roc_auc']}
        for k, v in results.items()
    ])
    print(f"\nSummary:\n{summary.to_string(index=False)}")

    # Save preprocessor
    if save_models:
        joblib.dump(prep, models_dir / 'clf_preprocessor.pkl')

    return results, summary, test_df


# --------------------------------------------------------------------------
# Transfer experiment: apply synthetic-trained RF to real pilot data
# --------------------------------------------------------------------------
def transfer_to_real(rf_model=None, prep=None) -> pd.DataFrame:
    """
    Apply the synthetic-trained Random Forest to real pilot D1 data (exploratory only).

    IMPORTANT: No ground-truth labels exist in real data. Predictions indicate
    'AM2-model-space similarity', not confirmed souring events. Interpret with caution.
    """
    models_dir = ROOT / 'models'
    if rf_model is None:
        rf_path = models_dir / 'clf_randomforest.pkl'
        if not rf_path.exists():
            print("[transfer] No trained RF found.")
            return pd.DataFrame()
        rf_model = joblib.load(rf_path)
    if prep is None:
        pp_path = models_dir / 'clf_preprocessor.pkl'
        if not pp_path.exists():
            print("[transfer] No preprocessor found.")
            return pd.DataFrame()
        prep = joblib.load(pp_path)

    pilot_path = ROOT / 'data' / 'processed' / 'pilot_D1_harmonised.csv'
    if not pilot_path.exists():
        print("[transfer] pilot_D1_harmonised.csv not found.")
        return pd.DataFrame()

    df = pd.read_csv(pilot_path, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)

    # Map real columns to synthetic feature names where possible
    col_map = {
        'vfa_mgl': 'vfa_mgl',
        'ph': 'pH',
        'alk_mgl': 'alkalinity_mgl',
        'biogas_m3d': 'q_CH4_L_L_d',  # very rough proxy; different unit
        'va_alk': 'sim_fos_tac_proxy',
    }
    real_feats = list(set(FEATURE_COLS))
    mapped = {}
    for sf in real_feats:
        for rk, rv in col_map.items():
            if rv == sf and rk in df.columns:
                mapped[sf] = df[rk]
    mapped_df = pd.DataFrame(mapped, index=df.index)
    # fill missing synthetic features with column median
    for c in real_feats:
        if c not in mapped_df.columns:
            mapped_df[c] = np.nan

    mapped_df = mapped_df[real_feats]
    valid = mapped_df.notna().any(axis=1)
    X_real = prep.transform(mapped_df.loc[valid])

    preds = rf_model.predict(X_real)
    pred_names = [LABEL_MAP.get(p, str(p)) for p in preds]

    result = df.loc[valid, ['datetime', 'ph', 'vfa_mgl', 'alk_mgl', 'va_alk', 'biogas_m3d']].copy()
    result['transfer_pred_label']  = preds
    result['transfer_pred_name']   = pred_names
    result['CAUTION'] = ('Synthetic-model prediction. No ground truth. '
                         'Feature alignment is approximate. '
                         'Do not interpret as confirmed plant state.')
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == '__main__':
    results, summary, test_df = run_supervised_classification()

    # Save summary
    (ROOT / 'reports').mkdir(exist_ok=True)
    summary.to_csv(ROOT / 'reports' / 'classification_summary.csv', index=False)

    # Save per-class reports
    for mname, r in results.items():
        r['class_report'].to_csv(ROOT / 'reports' / f'class_report_{mname.lower()}.csv', index=False)

    # Transfer experiment
    print("\n--- Transfer experiment (EXPLORATORY, no ground truth in real data) ---")
    transfer_df = transfer_to_real()
    if not transfer_df.empty:
        print(transfer_df['transfer_pred_name'].value_counts())
        transfer_df.to_csv(ROOT / 'reports' / 'transfer_experiment.csv', index=False)
        print("Saved -> reports/transfer_experiment.csv")

    print("\nDone.")
