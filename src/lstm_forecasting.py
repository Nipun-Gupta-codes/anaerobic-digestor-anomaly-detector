"""
LSTM time-series forecasting of VA/Alk ratio (real data) and VFA proxy (synthetic AM2 data).
=============================================================================================
Implementation notes
--------------------
* All sequences are built from chronologically ordered data.
* Training uses only the train split; scaler is fitted on training rows only.
* The model is built with pure Keras (TensorFlow); if TF is unavailable, falls back
  to a sklearn-based gradient-boosting regressor with lag features (documented).
* Evaluated against a persistence baseline (y_hat = y[t-1]).
* Targets:
    - Real data  : 'va_alk'  (VA/Alk ratio recomputed from raw columns)
                   Clearly labelled as such, NOT claimed to be FOS/TAC.
    - Synthetic  : 'sim_fos_tac_proxy' (simulated FOS/TAC-like ratio from AM2)
                   Clearly labelled as a simulation.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 42
np.random.seed(SEED)

# --------------------------------------------------------------------------
# Try importing TensorFlow / Keras
# --------------------------------------------------------------------------
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[lstm_forecasting] TensorFlow not available. Using GB fallback.")

# GB fallback
from sklearn.ensemble import GradientBoostingRegressor


# --------------------------------------------------------------------------
# Sequence construction
# --------------------------------------------------------------------------
def make_sequences(arr: np.ndarray, seq_len: int, horizon: int = 1):
    """
    Create sliding-window sequences from a 2-D array.
    X: (n, seq_len, n_features)   y: (n,)  — the target at seq_len+horizon-1
    No look-ahead: each X[i] uses only rows i..i+seq_len-1.
    """
    X, y = [], []
    for i in range(len(arr) - seq_len - horizon + 1):
        X.append(arr[i : i + seq_len, :-1])       # all columns except target
        y.append(arr[i + seq_len + horizon - 1, -1])  # target column (last)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def persistence_forecast(df: pd.DataFrame, target: str, horizon: int = 1) -> np.ndarray:
    """Persistence: forecast = last observed value."""
    return df[target].shift(horizon).values


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred, name='model') -> dict:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return dict(model=name, n=len(yt), MAE=np.nan, RMSE=np.nan, R2=np.nan)
    return dict(
        model=name, n=len(yt),
        MAE=float(mean_absolute_error(yt, yp)),
        RMSE=float(np.sqrt(mean_squared_error(yt, yp))),
        R2=float(r2_score(yt, yp)),
    )


# --------------------------------------------------------------------------
# Keras LSTM model
# --------------------------------------------------------------------------
def build_lstm(seq_len: int, n_features: int, units: int = 64,
               dropout: float = 0.2) -> 'keras.Model':
    """One-hidden-layer LSTM for univariate regression."""
    inp = layers.Input(shape=(seq_len, n_features), name='input')
    x   = layers.LSTM(units, return_sequences=False, dropout=dropout)(inp)
    x   = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    m   = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return m


# --------------------------------------------------------------------------
# GB fallback model (lag-feature approach)
# --------------------------------------------------------------------------
def gb_with_lags(X_seq: np.ndarray, seq_len: int, n_features: int):
    """Flatten LSTM sequences -> 2-D for gradient boosting fallback."""
    return X_seq.reshape(len(X_seq), seq_len * n_features)


# --------------------------------------------------------------------------
# Main training + evaluation function
# --------------------------------------------------------------------------
def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int = 7,
    horizon: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    epochs: int = 80,
    batch_size: int = 16,
    lstm_units: int = 64,
    dropout: float = 0.2,
    save_dir: Path = None,
    system_name: str = 'unknown',
    target_label: str = 'VA/Alk (measured)',
) -> dict:
    """
    Train LSTM (or GB fallback) for seq_len-step sliding window.

    Returns dict with:
        metrics: DataFrame
        history: keras History object or None
        scaler : fitted StandardScaler
        model  : fitted model
        pred_df: DataFrame with dates, obs, pred for the test split
    """
    # Drop rows without target
    sub = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy().reset_index(drop=True)
    if len(sub) < seq_len + horizon + 10:
        return dict(metrics=pd.DataFrame(), model=None, pred_df=pd.DataFrame(),
                    note=f"Insufficient observations ({len(sub)}) for seq_len={seq_len}, horizon={horizon}")

    n = len(sub)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    # Fit scaler on training rows only
    train_data = sub.iloc[:n_train]
    scaler = StandardScaler().fit(train_data)
    scaled = scaler.transform(sub)

    X_all, y_all = make_sequences(scaled, seq_len, horizon)
    # Adjust index mapping: sequence i corresponds to rows i to i+seq_len-1
    n_train_seq = max(0, n_train - seq_len - horizon + 1)
    n_val_seq   = max(0, n_val)

    X_tr, y_tr = X_all[:n_train_seq], y_all[:n_train_seq]
    X_va, y_va = X_all[n_train_seq : n_train_seq + n_val_seq], y_all[n_train_seq : n_train_seq + n_val_seq]
    X_te, y_te = X_all[n_train_seq + n_val_seq:], y_all[n_train_seq + n_val_seq:]

    if len(X_tr) < 4 or len(X_te) < 2:
        return dict(metrics=pd.DataFrame(), model=None, pred_df=pd.DataFrame(),
                    note=f"Sequence splits too small (tr={len(X_tr)}, te={len(X_te)})")

    # Unscale target for metrics (last column of scaler)
    target_idx = len(feature_cols)
    t_mean = scaler.mean_[target_idx]
    t_std  = scaler.scale_[target_idx]
    def unscale_y(y_sc): return y_sc * t_std + t_mean

    y_te_orig = unscale_y(y_te)

    metrics_rows = []
    history = None
    model   = None

    # ---- Persistence baseline ----
    # On the test set: predict y[t] = y[t-horizon] in original space
    pers_idx = n_train_seq + n_val_seq + seq_len - 1   # first row index of test targets
    persist_vals = sub[target_col].shift(horizon).iloc[pers_idx + horizon : pers_idx + horizon + len(y_te)].values
    metrics_rows.append(regression_metrics(y_te_orig, persist_vals, name=f'Persistence (h={horizon})'))

    # ---- LSTM or GB ----
    if TF_AVAILABLE:
        n_feats = X_tr.shape[2]
        mdl = build_lstm(seq_len, n_feats, lstm_units, dropout)
        callbacks = [
            keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor='val_loss'),
        ]
        if save_dir is not None:
            ckpt_path = Path(save_dir) / f'lstm_{system_name}_h{horizon}.keras'
            callbacks.append(keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True))

        val_data = (X_va, y_va) if len(X_va) > 0 else None
        hist = mdl.fit(X_tr, y_tr, validation_data=val_data, epochs=epochs,
                       batch_size=batch_size, callbacks=callbacks, verbose=0)
        history = hist
        model   = mdl
        y_pred_sc = mdl.predict(X_te, verbose=0).flatten()
        y_pred    = unscale_y(y_pred_sc)
        metrics_rows.append(regression_metrics(y_te_orig, y_pred, name=f'LSTM (h={horizon})'))
        model_label = 'LSTM'
    else:
        # HistGB fallback -- handles NaN natively
        from sklearn.ensemble import HistGradientBoostingRegressor
        X_tr_flat = gb_with_lags(X_tr, seq_len, X_tr.shape[2])
        X_te_flat = gb_with_lags(X_te, seq_len, X_te.shape[2])
        gb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                           max_iter=200, random_state=SEED)
        gb.fit(X_tr_flat, y_tr)
        y_pred_sc = gb.predict(X_te_flat)
        y_pred    = unscale_y(y_pred_sc)
        model     = gb
        metrics_rows.append(regression_metrics(y_te_orig, y_pred, name=f'HistGB-fallback (h={horizon})'))
        model_label = 'HistGradientBoosting'

    # ---- Train-mean baseline ----
    train_mean = sub[target_col].iloc[:n_train].mean()
    metrics_rows.append(regression_metrics(y_te_orig, np.full_like(y_te_orig, train_mean),
                                           name='Train-mean'))

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df['system']        = system_name
    metrics_df['seq_len']       = seq_len
    metrics_df['horizon']       = horizon
    metrics_df['target']        = target_col
    metrics_df['target_label']  = target_label
    metrics_df['model_backend'] = 'TensorFlow/Keras' if TF_AVAILABLE else 'GradientBoosting (fallback)'

    # Build prediction DataFrame for test set
    test_rows = sub.iloc[pers_idx + horizon : pers_idx + horizon + len(y_te)]
    has_date = 'datetime' in sub.columns or 'day' in sub.columns
    date_col = 'datetime' if 'datetime' in sub.columns else ('day' if 'day' in sub.columns else None)
    pred_df = pd.DataFrame({
        (date_col or 'index'):   test_rows[date_col].values if date_col else np.arange(len(y_te)),
        'observed': y_te_orig,
        'predicted': y_pred,
        'persistence': persist_vals,
    })

    return dict(metrics=metrics_df, history=history, scaler=scaler,
                model=model, pred_df=pred_df, model_label=model_label,
                n_train=len(X_tr), n_val=len(X_va), n_test=len(X_te))


# --------------------------------------------------------------------------
# Run on real pilot data (VA/Alk target)
# --------------------------------------------------------------------------
def run_real_data(horizons=(1, 3, 7), seq_len=7, save_dir: Path = None) -> pd.DataFrame:
    """
    Train LSTM (or GB) on real pilot VA/Alk data.
    Uses pilot D1 harmonised CSV (the only dataset with a credible instability episode).
    """
    processed = ROOT / 'data' / 'processed'
    all_metrics = []

    for ds_name in ['pilot_D1', 'pilot_D2']:
        path = processed / f'{ds_name}_harmonised.csv'
        if not path.exists():
            print(f"[lstm] {path} not found, skipping.")
            continue
        df = pd.read_csv(path, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
        # Only rows with valid VA/Alk
        df_sub = df[df['va_alk'].notna()].copy()
        if len(df_sub) < 30:
            print(f"[lstm] {ds_name}: too few VA/Alk observations ({len(df_sub)})")
            continue

        feature_cols = [c for c in ['ph', 'temp_c', 'ccr', 'biogas_m3d', 'vfa_mgl'] if c in df_sub.columns]
        feature_cols = [c for c in feature_cols if df_sub[c].notna().sum() > 0.3 * len(df_sub)]
        all_feat = feature_cols + ['va_alk']

        df_model = df_sub[(['datetime'] if 'datetime' in df_sub.columns else []) + all_feat].copy()

        for h in horizons:
            print(f"[lstm] {ds_name} VA/Alk horizon={h} … ", end='', flush=True)
            result = train_and_evaluate(
                df=df_model, feature_cols=feature_cols, target_col='va_alk',
                seq_len=seq_len, horizon=h, system_name=ds_name,
                target_label='VA/Alk ratio (real measurement, NOT FOS/TAC)',
                save_dir=save_dir,
            )
            if result.get('metrics') is not None and len(result['metrics']) > 0:
                all_metrics.append(result['metrics'])
                print(f"OK  (train={result.get('n_train','?')}, test={result.get('n_test','?')})")
            else:
                print(f"SKIPPED – {result.get('note', 'unknown reason')}")

    return pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()


# --------------------------------------------------------------------------
# Run on synthetic AM2 data (sim_fos_tac_proxy target)
# --------------------------------------------------------------------------
def run_synthetic_data(horizons=(1, 3, 7), seq_len=7, save_dir: Path = None) -> pd.DataFrame:
    """
    Train LSTM (or GB) on synthetic AM2 sim_fos_tac_proxy target.
    """
    syn_path = ROOT / 'data' / 'synthetic' / 'am2_synthetic_labelled.csv'
    if not syn_path.exists():
        print("[lstm] Synthetic data not found. Run src/simulation.py first.")
        return pd.DataFrame()

    df = pd.read_csv(syn_path).sort_values('day').reset_index(drop=True)
    feature_cols = ['S1_g_COD_L', 'X1_g_VS_L', 'S2_mmol_L', 'X2_g_VS_L',
                    'pH', 'vfa_mgl', 'alkalinity_mgl', 'D_per_day', 'q_CH4_L_L_d']
    feature_cols = [c for c in feature_cols if c in df.columns]
    target_col   = 'sim_fos_tac_proxy'

    all_metrics = []
    for h in horizons:
        print(f"[lstm] synthetic AM2 horizon={h} … ", end='', flush=True)
        result = train_and_evaluate(
            df=df, feature_cols=feature_cols, target_col=target_col,
            seq_len=seq_len, horizon=h, system_name='AM2_synthetic',
            target_label='Simulated FOS/TAC proxy (NOT real measurement)',
            save_dir=save_dir,
        )
        if result.get('metrics') is not None and len(result['metrics']) > 0:
            all_metrics.append(result['metrics'])
            print(f"OK  (train={result.get('n_train','?')}, test={result.get('n_test','?')})")
        else:
            print(f"SKIPPED – {result.get('note','unknown')}")

    return pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == '__main__':
    models_dir = ROOT / 'models'
    models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("LSTM FORECASTING – REAL DATA (VA/Alk ratio, pilot only)")
    print("=" * 60)
    real_metrics = run_real_data(horizons=[1, 3, 7], seq_len=7, save_dir=models_dir)
    if not real_metrics.empty:
        print("\nReal-data metrics:\n", real_metrics.to_string(index=False))
        real_metrics.to_csv(ROOT / 'reports' / 'lstm_metrics_real.csv', index=False)

    print("\n" + "=" * 60)
    print("LSTM FORECASTING – SYNTHETIC AM2 DATA (simulated FOS/TAC proxy)")
    print("=" * 60)
    syn_metrics = run_synthetic_data(horizons=[1, 3, 7], seq_len=7, save_dir=models_dir)
    if not syn_metrics.empty:
        print("\nSynthetic metrics:\n", syn_metrics.to_string(index=False))
        syn_metrics.to_csv(ROOT / 'reports' / 'lstm_metrics_synthetic.csv', index=False)

    print("\nDone.")
