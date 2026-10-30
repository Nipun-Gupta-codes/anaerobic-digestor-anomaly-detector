# Phase 1 – Data Audit and Feature-Availability Report

All numbers below were produced by `src/audit.py` and ad-hoc checks on the raw CSVs (no values are assumed).
Full per-column table: `reports/data_audit_columns.csv`.

## 1. Dataset overview

| Dataset | Rows x cols | Period | Calendar gaps | Duplicates |
|---|---|---|---|---|
| pilot (`sensor_ad_pilot_merged.csv`) | 309 x 73 | 2019-10-17 to 2020-08-31 | 11 missing days (gaps of 2, 2, 5, 6 days) | none |
| full_digester | 326 x 47 | 2021-03-12 to 2022-01-31 | none (daily) | none |
| full_wwc | 319 x 33 | 2021-03-12 to 2022-01-24 | none (daily) | none |

* `.Rd` row/column counts match the CSVs (309/73, 326/47, 319/33). The CSV `full_digester` has one extra column
  (`waste_provider_VS_pct_user`) not listed in the Rd. The `.rda` could not be opened here (no R / pyreadr), so it was not
  used to cross-check values; the Rd documentation was used instead. It is the same data as the CSVs, so nothing is double-counted.
* The two full-scale files share the same dates and identical CCR columns (verified) -> they are two views of ONE plant and must not be stacked as independent observations.
* Pilot (2019-20) and full-scale (2021-22) do not overlap in time and differ ~3,500x in biogas magnitude
  (pilot ~57 cf/day, full-scale ~199,000 cf/day), so they must not be pooled in one regression without scale handling.

## 2. Key-variable availability

| Variable | Pilot (D1 / D2) | Full-scale digester | Full-scale WWC |
|---|---|---|---|
| pH | 100% / 100% | **21%** (isolated days) | none |
| VFA (mg/L) | 60% / 60% | 65% | none |
| Alkalinity (mg/L) | 61% / 54% | 40% (every 2-3 days) | none |
| VA/Alk ratio | 31% / 31% | 39% | none |
| Organic loading | 50% (actual), 76% (7-day avg) | 63% (VS loading by stream) | feed VS/TS loads, 31-38% |
| Temperature | 100% (degF) | 98% (degC) | none |
| COD | 61% / 54% (sCOD 17%) | **6%** (feed only) | none |
| VS (digester) | 61% / 60% | 38% | feed only |
| CCR | 100% (3 sensors) | 100% (2 sensors) | 100% (same 2) |
| Biogas | 96% (cf/day) | 98% | none |
| Methane | 29% (% only) | 98% (cf/min), 64% (cf/day) | none |
| HRT | **not available** | 73% (detention time, days) | none |

## 3. Findings that affect the project design

1. **Full-scale WWC has no digester chemistry or biogas.** It is feed composition + CCR only, so it can support feed-characterisation and CCR-vs-loading analysis, not biogas prediction or anomaly labels.
2. **Real VA/Alk ratio is not FOS/TAC.** In the pilot it equals VA (mg/L) / alkalinity (mg/L) exactly (max |diff| = 0). FOS/TAC is a titration-based ratio (organic acids as acetic-acid equivalents / buffer capacity in CaCO3 equivalents), so the two are related but not interchangeable, and the thresholds 0.23/0.30/0.40 cannot be applied to it without caveats. No dataset contains FOS or TAC.
3. **The full-scale `va_alk_ratio` is rounded to 2 decimals** (range 0.01-0.03, correlation with VA/alk recomputed from raw columns = 0.82). The ratio should be recomputed from `volatile_acids_mgl / alkalinity_mgl` instead of using the reported one. In full-scale the digester never leaves a very stable regime (pH 7.08-7.39, VFA max 114 mg/L), so it contains essentially **no real instability** and is a poor test for souring detection.
4. **Real instability events exist only in the pilot**, around May-June 2020 (e.g. D2 ratio up to 0.82 on 2020-05-14 with pH 6.58; D1 ratio 0.30-0.35 for 20 May-8 Jun with VFA ~1,600-2,300 mg/L; D2 gas fell to 8-20 cf/day). pH on 2020-01-14 (4.53 in both digesters) coincides in both reactors and with normal VFA/alkalinity, so it is likely a probe/logging problem rather than acidification; this needs confirmation, not assumption. Pilot comments mention a broken gas meter and counter resets, so zero gas days (D1: 6, D2: 14) may be instrument faults.
5. **Sparse sampling limits forecasting.** VFA and alkalinity are measured intermittently: longest consecutive observed run is 13 days (D1 VFA), 6 days for a computable ratio in the pilot, and 3 days for alkalinity in full-scale. A 7-day-ahead LSTM on the real VA/alk ratio is therefore not credible with sequence lengths of 7+ days of observed inputs; real-data forecasting should target a 1-3 day horizon with gaps handled explicitly, or be restricted to the pilot. The real-data LSTM is the part of your synopsis most at risk.
6. **Biogas is the best-populated target**: full-scale 318 consecutive days; pilot runs of 228 and 59 days. Pilot `gas_generation_cfhr` does not agree with `cfday` (corr -0.015), so `cfhr` should be excluded and `cfday` used.
7. Pilot units are US customary (degF, cf, lb/1000 ft3/d, gal) and full-scale mixes degC and cf. Convert to SI for any comparison with AM2.
8. Outliers: VFA and VA/Alk have many IQR outliers (32/186 and 18/97 in pilot D1), and these are mostly the genuine May-June events. They will be kept, not removed.
9. AM2 states (S1, X1, X2) are not measured anywhere; AM2 will be a literature-parameter simulation, not a calibrated plant model. Only pH, VFA, alkalinity, biogas/methane and loading can be compared against it.

## 3b. Correction added in Phase 2 (data-quality finding)

Phase 1 summarised the *reported* pilot ratio columns, which are blank in Oct-Dec 2019. Recomputing VA/Alk from the raw columns
gives values around 1.0 for pilot D1 in that period, because D1 alkalinity is ~85 mg/L (nearly constant, 27 observations) while pH is 7.4-7.7,
VFA ~50-150 mg/L and biogas is normal. This is inconsistent with a working bicarbonate-buffered digester and is treated as an
unreliable alkalinity record (unit/method/entry problem), NOT as souring. Raw values are kept (`alk_mgl_raw`), flagged
(`flag_alk_suspect`, alkalinity < 500 mg/L) and masked in the modelling column. The same treatment is applied to the isolated pH 4.5
reading (`flag_ph_suspect`) and zero-gas days (`flag_gas_zero`). The thresholds are documented rules, not validated limits.
The May-June 2020 events remain the only credible real instability period.

## 4. Consequences for the plan

* Real-data biogas prediction: pilot (D1 and D2 as separate reactors) and full-scale digester, trained and tested chronologically, separately.
* Real-data anomaly detection: pilot is informative; full-scale will mostly show stable operation, and that is a result.
* Real-data forecasting target: pilot VA/Alk (recomputed), short horizons, documented as "VA/Alk", not FOS/TAC.
* FOS/TAC thresholds will be applied to a simulated FOS/TAC-like indicator from AM2 only, and labelled as project-specific.

## 5. Environment limits found

No `xgboost`, `tensorflow`, `torch`, `streamlit` or `shap` here and the sandbox has no network, so they cannot be installed. NumPy/SciPy/pandas/scikit-learn/matplotlib/seaborn are present.
