"""
Evaluate predictions.csv against actual day-ahead prices fetched live
from the Energy-Charts API for the evaluation window (8–9 May 2026).
"""
import requests
import numpy as np
import pandas as pd

EC_BASE = "https://api.energy-charts.info"

EVAL_START = "2026-05-08"
EVAL_END   = "2026-05-09"

# Evaluation window in UTC (matches predictions.csv)
EVAL_TIMESTAMPS = pd.date_range(
    start="2026-05-08 17:00", periods=30, freq="1h", tz="UTC"
)


def fetch_prices(bzn: str) -> pd.Series:
    r = requests.get(
        f"{EC_BASE}/price",
        params={"bzn": bzn, "start": EVAL_START, "end": EVAL_END},
        timeout=60,
    )
    r.raise_for_status()
    d = r.json()
    ts = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
    s  = pd.Series(d["price"], index=ts, name="actual")
    # Keep only hourly data (some zones return 15-min)
    s = s.resample("1h").first()
    return s


def pinball(y_true, y_pred, q):
    r = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.where(r >= 0, q * r, (q - 1) * r)))


def evaluate(zone_label, bzn, pred_p025, pred_p50, pred_p975, actual,
             actual_aligned, preds_full, label):
    mae  = float(np.mean(np.abs(actual.values - pred_p50.values)))
    rmse = float(np.sqrt(np.mean((actual.values - pred_p50.values) ** 2)))
    pb   = pinball(actual.values, pred_p50.values, 0.45)
    bias = float(np.mean(pred_p50.values - actual.values))
    inside = (actual.values >= pred_p025.values) & (actual.values <= pred_p975.values)
    coverage = inside.mean() * 100

    print(f"\n{'='*50}")
    print(f"  {zone_label}")
    print(f"{'='*50}")
    print(f"  Pinball (q=0.45) : {pb:.3f}")
    print(f"  MAE              : {mae:.2f} EUR/MWh")
    print(f"  RMSE             : {rmse:.2f} EUR/MWh")
    print(f"  Bias (pred-act)  : {bias:+.2f} EUR/MWh")
    print(f"  95% PI coverage  : {coverage:.1f}%  ({inside.sum()}/{len(inside)} hours)")

    print(f"\n  {'Timestamp (UTC)':25s}  {'Actual':>8}  {'p025':>8}  {'p50':>8}  {'p975':>8}  {'In PI':>6}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for ts in EVAL_TIMESTAMPS:
        a_val = actual_aligned.get(ts, np.nan)
        p0 = preds_full.loc[ts, f"{label} p025"]
        p5 = preds_full.loc[ts, f"{label} p50"]
        p9 = preds_full.loc[ts, f"{label} p975"]
        if np.isnan(a_val):
            print(f"  {str(ts)[:25]:25s}  {'N/A':>8}  {p0:8.2f}  {p5:8.2f}  {p9:8.2f}  {'N/A':>6}")
        else:
            hit = "YES" if p0 <= a_val <= p9 else "NO "
            print(f"  {str(ts)[:25]:25s}  {a_val:8.2f}  {p0:8.2f}  {p5:8.2f}  {p9:8.2f}  {hit:>6}")


def main():
    preds = pd.read_csv("predictions.csv")
    # Parse timestamps (CET +01:00) and convert to UTC
    preds["ts_utc"] = pd.to_datetime(preds["timestamp"], utc=True)
    preds = preds.set_index("ts_utc")
    preds.index = preds.index.tz_convert("UTC")

    zones = {
        "DE-LU": "DE-LU",
        "ES":    "ES",
    }

    for label, bzn in zones.items():
        print(f"\nFetching actual prices for {label} ({bzn})...")
        try:
            actual = fetch_prices(bzn)
        except Exception as e:
            print(f"  ERROR fetching {label}: {e}")
            continue

        # Align to evaluation window
        actual_aligned = actual.reindex(EVAL_TIMESTAMPS)
        n_missing = actual_aligned.isna().sum()
        if n_missing > 0:
            print(f"  WARNING: {n_missing} hours missing from API response")

        actual_clean = actual_aligned.dropna()
        ts_valid = actual_clean.index

        p025 = preds.loc[ts_valid, f"{label} p025"]
        p50  = preds.loc[ts_valid, f"{label} p50"]
        p975 = preds.loc[ts_valid, f"{label} p975"]

        evaluate(label, bzn, p025, p50, p975, actual_clean, actual_aligned, preds, label)

    print("\nDone.")


if __name__ == "__main__":
    main()
