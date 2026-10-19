"""Shared helpers: dataset loading, cleaning, daily series and metrics."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "pune_aqi_csv.csv")   # <- put your CSV here
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

POLLUTANTS = ["SO2", "NOx", "RSPM", "SPM"]


def parse_dates(series):
    """Mixed formats: timestamped rows are MM-DD-YYYY, the rest DD-MM-YYYY."""
    raw = (series.astype(str).str.replace("\r", "", regex=False)
                 .str.replace("/", "-", regex=False).str.strip())
    has_time = raw.str.contains(":")
    d1 = pd.to_datetime(raw[has_time].str.slice(0, 10), format="%m-%d-%Y", errors="coerce")
    d2 = pd.to_datetime(raw[~has_time].str.slice(0, 10), format="%d-%m-%Y", errors="coerce")
    out = pd.concat([d1, d2]).reindex(raw.index)
    return out.fillna(pd.to_datetime(raw, errors="coerce", dayfirst=True))


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. "
                                "Copy your CSV into the 'data' folder as pune_aqi_csv.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().reset_index(drop=True)

    # 'SO2 µg/m3' -> 'SO2', 'Nox µg/m3' -> 'NOx', ...
    rename = {}
    for col in df.columns:
        parts = col.split()
        first = parts[0].lower() if parts else ""
        for target in ["Date", "AQI", "SO2", "NOx", "RSPM", "SPM"]:
            if first == target.lower() and target not in rename.values():
                rename[col] = target
    for col in df.columns:
        if col not in rename and any(k in col.lower() for k in ["location", "station", "area", "site"]):
            rename[col] = "Location"
            break
    df = df.rename(columns=rename)

    if "Date" not in df.columns or "AQI" not in df.columns:
        raise ValueError(f"Need 'Date' and 'AQI' columns. Found: {list(df.columns)}")

    df["Date"] = parse_dates(df["Date"])
    for c in ["AQI"] + POLLUTANTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")   # 'BDL - x' -> NaN
    return df


def daily_aqi(df):
    """City-wide daily average AQI -> DataFrame(Date, AQI)."""
    return (df.dropna(subset=["Date", "AQI"]).groupby("Date")["AQI"].mean()
              .reset_index().sort_values("Date").reset_index(drop=True))


def metrics(y_true, y_pred):
    return {"R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("Saved", path)