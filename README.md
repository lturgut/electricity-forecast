# Electricity Price Forecast

Day-ahead electricity price forecasting for **DE-LU** (Germany-Luxembourg) and **ES** (Spain) using LightGBM quantile regression.

## Evaluation window

May 11, 2026 — 24 hourly slots (00:00–23:00 UTC), submitted as CEST timestamps in `predictions.csv`.

## Model

- **Algorithm**: LightGBM quantile regression at q=0.025, q=0.45, q=0.975
- **Ensemble**: 5 models with different random seeds, predictions averaged
- **Sample weighting**: Exponential decay with 180-day half-life (recent data weighted more)
- **Training data**: 2024-01-01 to 2026-05-10

### Features (43 total)

| Group | Features |
|-------|----------|
| Time | hour sin/cos, day-of-week sin/cos, month sin/cos, is_weekend, is_summer |
| Price lags | lag 24h, 48h, 72h, 120h, 168h, momentum |
| Rolling stats | mean/std over 24h, 48h, 72h, 168h windows; min over 24h/48h; p10 over 24h |
| Weather | temperature, wind speed, solar radiation, solar_rad_delta |
| Generation | renewable share, wind MW, solar MW, load MW, fossil gas MW |
| Solar surplus | solar_mw / load_mw ratio (key driver of near-zero prices) |
| Cross-zone | other zone's price lag 24h and 48h |
| Calendars | DE public holiday, ES public holiday |
| Commodities | gas price lag 1d/7d (Henry Hub via yfinance), CO2 price lag 1d/7d (KRBN ETF via yfinance) |

### Interval widening

For high-solar hours (solar radiation > 150 W/m²), prediction intervals are expanded proportionally to capture the increased uncertainty from solar surplus price crashes.

## Data sources

| Data | Source |
|------|--------|
| Day-ahead prices | [Energy-Charts API](https://api.energy-charts.info) |
| Generation mix | Energy-Charts API (`/public_power`) |
| Historical weather | Open-Meteo archive API |
| Weather forecast | Open-Meteo forecast API |
| Gas prices | Henry Hub (NG=F) via yfinance |
| CO2 prices | KRBN ETF via yfinance |

All data is cached locally in `data/` on first run.

## Files

| File | Purpose |
|------|---------|
| `run_forecast.py` | Main pipeline — fetches data, trains ensemble, writes `predictions.csv` |
| `backtest_may10.py` | Backtest on May 10 using data through May 9 — measures model quality |
| `evaluate.py` | Evaluates `predictions.csv` against actual prices from Energy-Charts API |
| `predictions.csv` | Submission file (24 rows, CEST timestamps) |
| `data/` | Cached CSVs for prices, generation, weather, gas, CO2 |

## Usage

```bash
# Generate predictions
python run_forecast.py

# Backtest on May 10
python backtest_may10.py

# Evaluate submitted predictions
python evaluate.py
```

## Requirements

```
lightgbm
pandas
numpy
requests
statsmodels
holidays
yfinance
```

## Metric

Pinball loss at q=0.025, q=0.45, and q=0.975 for each zone.

## Report

A short methods write-up with the full mathematical formulation and a backtest table is in [`report/report.pdf`](report/report.pdf) (source: `report/report.tex`).
