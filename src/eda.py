"""EDA figures. Every plot has title, axis labels, units where known. Saves PNGs to reports/figures."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data_loader import load_all
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})

def save(fig, name): fig.tight_layout(); fig.savefig(FIG / name); plt.close(fig); print("saved", name)

def missing_heatmap(df, title, name, min_missing=0.0):
    m = df.drop(columns=["datetime"]).isna()
    keep = m.columns[m.mean() > min_missing]; keep = keep[keep != "comments_digester_1_ac"]
    fig, ax = plt.subplots(figsize=(14, max(4, 0.16 * len(keep))))
    sns.heatmap(m[keep].T, cbar=False, cmap=["#2a9d8f", "#e76f51"], ax=ax, xticklabels=False, yticklabels=True)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    ax.set_title(f"{title}: missing values (orange = missing; x = observations in time order)")
    ax.set_xlabel("Observation index (chronological, daily)"); ax.set_ylabel("Variable")
    save(fig, name)

def corr_heatmap(df, cols, title, name):
    c = df[cols].corr(min_periods=20)
    fig, ax = plt.subplots(figsize=(0.7*len(cols)+3, 0.6*len(cols)+2))
    sns.heatmap(c, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7})
    ax.set_title(f"{title}: Pearson correlation (pairwise, >=20 shared obs)")
    save(fig, name)

def timeseries_panel(tabs: dict, name):
    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=False)
    spec = [("ph", "pH (-)"), ("vfa_mgl", "VFA (mg/L)"), ("alk_mgl", "Alkalinity (mg/L)"),
            ("va_alk", "VA/Alk (mg/L per mg/L)"), ("biogas_m3d", "Biogas (m³/day)")]
    for ax, (c, lab) in zip(axes, spec):
        for k, t in tabs.items():
            if k == "full_scale" and c == "biogas_m3d": continue
            ax.plot(t["datetime"], t[c], ".-", ms=3, lw=.6, label=k)
        ax.set_ylabel(lab); ax.legend(fontsize=7, ncol=3)
        if c == "va_alk": ax.axhline(.3, ls="--", c="k", lw=.7)
    axes[0].set_title("Stability indicators over time (pilot D1/D2 and full-scale; dashed = 0.3 VA/Alk, project early-warning level)")
    axes[-1].set_xlabel("Date")
    save(fig, name)

def full_biogas_ts(t, name):
    fig, ax = plt.subplots(figsize=(12, 3.5)); ax.plot(t["datetime"], t["biogas_m3d"], lw=.8)
    ax.set_title("Full-scale biogas production over time"); ax.set_xlabel("Date"); ax.set_ylabel("Biogas (m³/day)")
    save(fig, name)

def distributions(tabs, cols, name):
    fig, axes = plt.subplots(len(cols), len(tabs), figsize=(4*len(tabs), 2.6*len(cols)))
    for j, (k, t) in enumerate(tabs.items()):
        for i, c in enumerate(cols):
            ax = axes[i, j]; s = t[c].dropna()
            if len(s): ax.hist(s, bins=30, color="#457b9d")
            ax.set_title(f"{k}: {c} (n={len(s)})", fontsize=8); ax.set_ylabel("Count"); ax.set_xlabel(c, fontsize=7)
    fig.suptitle("Variable distributions by system", y=1.0); save(fig, name)

def boxplots(tabs, cols, name):
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2*len(cols), 4))
    for ax, c in zip(axes, cols):
        data = [t[c].dropna() for t in tabs.values()]
        ax.boxplot(data, tick_labels=list(tabs), showfliers=True, flierprops=dict(markersize=3))
        ax.set_title(c); ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.suptitle("Boxplots (points beyond whiskers = IQR outliers; retained, not removed)"); save(fig, name)

def scatter_pair(tabs, x, y, xlabel, ylabel, title, name, logy=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, t in tabs.items():
        m = t[[x, y]].dropna()
        if len(m): ax.scatter(m[x], m[y], s=10, alpha=.6, label=f"{k} (n={len(m)})")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=7)
    if logy: ax.set_yscale("log")
    save(fig, name)

def run_all():
    D = load_all(); P, F, W = D["pilot"], D["full_digester"], D["full_wwc"]
    raw_sets = {"pilot": P, "full_digester": F, "full_wwc": W}
    for k, df in raw_sets.items(): missing_heatmap(df, k, f"missing_{k}.png")
    T = {n: pd.read_csv(ROOT / "data/processed" / f"{n}_harmonised.csv", parse_dates=["datetime"]) for n in ["pilot_D1", "pilot_D2", "full"]}
    T["full_scale"] = T.pop("full")
    base = ["ph", "vfa_mgl", "alk_mgl", "va_alk", "temp_c", "biogas_m3d", "ccr", "vs_pct", "cod_mgl"]
    corr_heatmap(T["pilot_D1"], [c for c in base + ["PD_organic_loading_rate_pct", "fog_loading_rate_lb1000ft3xd", "food_loading_rate_lb1000ft3xd", "methane_pct"] if c in T["pilot_D1"]], "Pilot D1", "corr_pilot_D1.png")
    corr_heatmap(T["pilot_D2"], [c for c in base + ["PD_organic_loading_rate_pct", "fog_loading_rate_lb1000ft3xd", "food_loading_rate_lb1000ft3xd"] if c in T["pilot_D2"]], "Pilot D2", "corr_pilot_D2.png")
    fc = [c for c in base + ["hrt_d", "primary_VS_loading_pct", "twas_VS_loading_pct", "external_organic_loading_pct", "methane_m3min", "ccr_slurry"] if c in T["full_scale"]]
    corr_heatmap(T["full_scale"], fc, "Full-scale digester", "corr_full_scale.png")
    wc = [c for c in W.columns if c not in ("datetime",)]
    corr_heatmap(W, ["AD_primary_sludge_VS_pct", "AD_twas_VS_pct", "AD_slurry_VS_pct", "AD_FOG_VS_load_lbs", "AD_total_feed_flow_mgd", "AD_FOG_flow_pct", "daily_ccr_digester", "daily_ccr_slurry"], "Full-scale WWC (feed composition + CCR)", "corr_full_wwc.png")
    timeseries_panel(T, "ts_stability_indicators.png"); full_biogas_ts(T["full_scale"], "ts_full_biogas.png")
    distributions(T, ["ph", "vfa_mgl", "va_alk", "biogas_m3d"], "dist_key_vars.png")
    boxplots(T, ["ph", "vfa_mgl", "va_alk", "ccr"], "box_key_vars.png")
    pil = {k: T[k] for k in ("pilot_D1", "pilot_D2")}
    scatter_pair(pil, "PD_organic_loading_rate_pct", "biogas_m3d", "Organic loading rate (% of target capacity; pilot)", "Biogas (m³/day)", "Pilot: biogas vs organic loading rate", "sc_pilot_biogas_vs_olr.png")
    scatter_pair({"full_scale": T["full_scale"]}, "primary_VS_loading_pct", "biogas_m3d", "Primary-sludge VS loading (%)", "Biogas (m³/day)", "Full-scale: biogas vs primary VS loading", "sc_full_biogas_vs_vsload.png")
    scatter_pair(pil, "ccr", "biogas_m3d", "Daily CCR (sensor units)", "Biogas (m³/day)", "Pilot: CCR vs biogas", "sc_pilot_ccr_vs_biogas.png")
    scatter_pair({"full_scale": T["full_scale"]}, "ccr", "biogas_m3d", "Daily CCR (sensor units)", "Biogas (m³/day)", "Full-scale: CCR vs biogas", "sc_full_ccr_vs_biogas.png")
    scatter_pair({**pil, "full_scale": T["full_scale"]}, "ccr", "vfa_mgl", "Daily CCR (sensor units)", "VFA (mg/L)", "CCR vs digester VFA (sensor detects volatile acids)", "sc_ccr_vs_vfa.png", logy=True)
    scatter_pair({**pil, "full_scale": T["full_scale"]}, "alk_mgl", "vfa_mgl", "Alkalinity (mg/L)", "VFA (mg/L)", "Volatile acids vs alkalinity (lines of constant ratio drawn in notebooks)", "sc_vfa_vs_alk.png", logy=True)
    scatter_pair({**pil}, "ph", "va_alk", "pH (-)", "VA/Alk (-)", "Pilot: pH vs VA/Alk (pH insensitive until late?)", "sc_ph_vs_vaalk.png", logy=True)

if __name__ == "__main__":
    run_all()
