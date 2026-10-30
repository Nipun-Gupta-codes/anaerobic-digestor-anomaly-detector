"""Dataset loading with schema validation. Paths are resolved relative to the project root."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

DATASETS = {
    "pilot": "sensor_ad_pilot_merged.csv",
    "full_digester": "sensor_ad_full_scale_digester_ccr.csv",
    "full_wwc": "sensor_ad_full_scale_wwc_ccr.csv",
}

def load_dataset(name: str, raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Load one raw CSV, drop the R row-index column, parse datetime, sort chronologically."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Choose from {list(DATASETS)}")
    path = Path(raw_dir) / DATASETS[name]
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, na_values=["NA", ""])
    first = df.columns[0]
    if first.startswith("Unnamed") or first == "":
        df = df.drop(columns=first)
    if "datetime" not in df.columns:
        raise ValueError(f"{name}: no datetime column")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    df.attrs["dataset"] = name
    return df

def load_all(raw_dir: Path | str = RAW_DIR) -> dict[str, pd.DataFrame]:
    return {k: load_dataset(k, raw_dir) for k in DATASETS}
