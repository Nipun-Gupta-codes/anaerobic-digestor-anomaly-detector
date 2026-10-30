# Phases 2-3: Preprocessing/EDA and real-data baselines

All numbers come from executed code (`src/run_biogas.py`, `src/run_biogas_cv.py`, `src/run_anomaly.py`); CSVs are in `reports/`, figures in `reports/figures/`.
Pilot D1 and D2, and full-scale, are always modelled **separately**. Everything here uses REAL measurements; nothing is simulated yet.

## Phase 2: preprocessing and features
* Per-reactor harmonised tables with common names and SI units (degF->degC, cf->m3): `data/processed/{pilot_D1,pilot_D2,full}_harmonised.csv`.
* Quality flags with raw values retained: D1 alkalinity < 500 mg/L (27 rows), pH < 5.5 (1 row per pilot digester, same day), zero-gas days (D1 6, D2 14; masked as NaN in the modelling column).
* VA/Alk recomputed from VFA and alkalinity everywhere. Lab variables are never interpolated; they are carried forward as "last measured value + age in days" (max 7 days), which is what an operator would know.
* Chronological splits with a 7-row gap, train-only median imputation and scaling (tests in `tests/test_phase2_3.py`: 7 pass).
* 18 EDA figures (missingness, correlations, time series, distributions, boxplots, biogas vs loading, CCR relationships, VFA vs alkalinity, pH vs VA/Alk).

## Phase 3A: biogas prediction (real data)
Chronological 60/20/20 split. Test blocks: pilot 2020-07-11 to 2020-08-31; full-scale 2021-11-28 to 2022-01-23. MAE in m3/day.

| System | Persistence (y[t-1]) | Train mean | Best process-only model | Best with yesterday's gas |
|---|---|---|---|---|
| pilot D1 | 0.166 (R2 0.61) | 0.567 | RandomForest 0.282 (R2 0.09) | RandomForest 0.197 (R2 0.49) |
| pilot D2 | 0.137 (R2 0.29) | 0.519 | RandomForest 0.577 (R2 -6.98) | RandomForest 0.304 (R2 -1.44) |
| full-scale | 1379 (R2 -0.56) | 1385 | HistGB 1654 (R2 -0.80) | HistGB 1620 (R2 -0.75) |

Rolling-origin check (5 expanding folds, mean MAE): persistence is best in every system, pilot D1 0.168 vs RF 0.683 (process-only), D2 0.163 vs 0.598, full-scale 1349 vs 1793.
**Conclusion:** with the variables available, no tested model (Linear/Ridge, Random Forest, Gradient Boosting, HistGB as an XGBoost stand-in) beats yesterday's gas value. Process-only models beat the training mean in only one of three single-split cases (pilot D1) and not in the pilot rolling-origin folds (D1 mean MAE 0.665-0.683 vs 0.624 for the training mean; D2 0.598-0.682 vs 0.522). Adding yesterday's gas lets them beat the training mean in the pilot (D1 0.32-0.37, D2 0.40-0.46) but still not persistence (0.17/0.16). In full-scale all models beat the training mean in the rolling folds (1,474-1,793 vs 2,355) only because that mean is stale after a level shift, and none beats persistence (1,349). Biogas level drifts between periods (full-scale: mean ~8,200 m3/d in the early training block vs ~3,500 in the next block; pilot D1 test-block means range 0.9-1.8 m3/d vs ~2.0 in training), and tree models cannot extrapolate a level they did not see. Plain Linear/Ridge regression was the worst model.

Feature importance (Random Forest) is an association only. Impurity importance ranks alkalinity first in pilot D1 (0.54) but permutation importance on the test block is ~0, and alkalinity declines steadily over the pilot period, so it is most likely acting as a time proxy. For full-scale, CCR (and its 7-day mean) leads, but permutation importance on the test block is negative, i.e. shuffling it did not hurt. **Do not present these as drivers of biogas.** SHAP was not run (library unavailable).

## Phase 3B: unsupervised anomaly detection (real data)
Detectors fitted on the first 60% of each series, scored on the rest. "Chemical evidence" proxy: VA/Alk > 0.30 or VFA > 500 mg/L or pH < 6.8. This is a proxy, not ground truth; days with only pH measured count as "normal" if pH is fine.

* Pilot D2 (18 evidence days in holdout), logged-variables-only features (no VFA/alkalinity used): AUC 0.93 (IsolationForest), 0.94 (LOF), 0.84 (One-Class SVM). Pilot D1 (20 evidence days): AUC 0.72 / 0.85 / 0.91.
* Precision is low (0.19-0.39 for logged features; base rate of evidence days in the holdout is 16% for D1) because the detectors also flag a lot of the holdout (e.g. IsolationForest flags 96 of 126 D1 holdout days): operating levels shifted after the training period, so "unusual vs. early operation" is not the same as "unstable". Recall is high (0.95-1.0).
* Full-scale: no day meets the evidence proxy (max VA/Alk 0.027, pH 7.08-7.39), so there is nothing to validate against; IsolationForest flags 12 days, mostly Mar-Jun 2021, with higher CCR. These are statistical outliers in a stable plant and are not reported as souring.
* Caveat: the pilot training window contains almost no instability, which favours novelty detection; with another split the numbers would change. Only one real instability episode (May-June 2020) exists, so these are small-sample results.

## Limitations discovered so far
1. A single real instability episode in the pilot; none in full-scale.
2. Sparse VFA/alkalinity sampling; flagged D1 alkalinity record in 2019.
3. Level shifts and drift between periods dominate prediction error.
4. XGBoost, SHAP, TensorFlow/PyTorch and Streamlit are not installed here.

## Next (Phase 4-5)
Implement AM2 (S1, X1, S2, X2) with scipy `solve_ivp`, build scenarios A-E and the labelled 6-class synthetic dataset, then the classifier with held-out simulated scenarios. I'd also like to confirm the AM2 formulation (Bernard et al., 2001) and parameter values against the literature before coding, which needs a web search.
