# Electricity Price Forecast — Team Higher Power

Day-ahead electricity price forecasting for **DE-LU** (Germany-Luxembourg) and **ES** (Spain) using LightGBM quantile regression. Submission for the ETH Zürich Frigg Hackathon.

## Evaluation window

May 11, 2026 — 24 hourly slots (02:00–2026-05-12 01:00 CEST), submitted as CEST timestamps in `higher_power_predictions.csv`.

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

## Files

| File | Purpose |
|------|---------|
| `higher_power_model.ipynb` | Self-contained notebook — loads data, trains ensemble, generates predictions |
| `higher_power_predictions.csv` | Submission file (24 rows, CEST timestamps, 7 columns) |
| `higher_power_data.zip` | All input CSVs with README.txt describing each source |
| `report/report.tex` | Methods write-up (LaTeX source) |
| `report/report.pdf` | Methods write-up (compiled PDF) |
| `data/` | Cached CSVs for prices, generation, weather, gas, CO2 |

## Usage

Open and run all cells in `higher_power_model.ipynb`. The notebook is self-contained:
- Downloads and caches all data on first run (skips if CSVs already present in `data/`)
- Trains the 30-model ensemble (2 zones × 3 quantiles × 5 seeds)
- Writes `higher_power_predictions.csv`

## Data sources

| Data | Source |
|------|--------|
| Day-ahead prices | [Energy-Charts API](https://api.energy-charts.info) |
| Generation mix | Energy-Charts API (`/public_power`) |
| Historical weather | Open-Meteo archive API |
| Weather forecast | Open-Meteo forecast API |
| Gas prices | Henry Hub (NG=F) via yfinance |
| CO2 prices | KRBN ETF via yfinance |

## Requirements

```
%pip install -q lightgbm statsmodels pandas numpy requests holidays yfinance
```

## Metric

Pinball loss at q=0.025, q=0.45, and q=0.975 for each zone.

## Report

Methods write-up with full mathematical formulation: [`report/report.pdf`](report/report.pdf)
