# Electricity Price Forecasting — DE-LU & ES

**Frigg / S2S / ETH Analytics Club challenge submission**
**Team: larat · ETH Zurich**

Probabilistic day-ahead electricity price forecasts for two European bidding zones:

| Zone | Market | Exchange |
|---|---|---|
| **DE-LU** | Germany–Luxembourg | EPEX SPOT |
| **ES** | Spain | OMIE |

**Evaluation window:** Mon 11 May 02:00 CEST → Tue 12 May 01:00 CEST (24 hourly slots)
**Metric:** Pinball loss at q = 0.025 / 0.45 / 0.975 (q=0.45 is asymmetric: overestimation penalised 1.22× more)

---

## Notebooks

### `model.ipynb` — Primary submission model

A three-layer regime-aware ensemble:

**Layer 1 — Global LightGBM** (one model per zone × quantile = 6 models)
- 22 features: cyclical calendar encodings, four price lags (t‑24h / t‑48h / t‑72h / t‑168h), rolling statistics (24h / 168h mean and std), weather (temperature, wind speed, solar radiation, solar×hour interaction), renewable share
- Two-stage training: probe run with early stopping on a held-out validation window to find the optimal number of trees → retrain final model on 100% of data at that iteration count
- Pinball loss objective (`alpha = q`) trains directly on the competition metric

**Layer 2 — Hour-Specific LightGBM (LEAR-style)** (24 hours × 3 quantiles × 2 zones = 144 models)
- Inspired by LEAR (Lago et al., 2021 — the academic benchmark for day-ahead EPF): a dedicated model for every hour of the day
- 23 features per model: same-hour price lags (t‑1d / t‑2d / t‑7d / t‑14d), 28-day rolling stats, previous-day summary statistics (mean / std / max / min / peak / off-peak), weather, calendar
- Training on only same-hour observations means each model specialises in its hour's structural drivers (e.g., the solar-noon model learns to suppress prices on high-radiation days; the 20:00 model learns the evening ramp)

**Layer 3 — STL Long-Term Fallback** (for horizons > 14 days)
- Seasonal–Trend decomposition using Loess (STL) extracts trend + weekly seasonality from historical prices
- Uncertainty bands scaled as √horizon to reflect increasing forecast uncertainty

**Ensemble:** 40% Global LightGBM + 60% Hour-Specific LightGBM (short-term); STL (long-term)

**Evaluation pipeline:**
1. *Bridge step*: predict May 10 prices first using history through May 9, so that `lag24h` is properly populated for May 11 (avoids NaN cascade in rolling features)
2. Inject May 10 predictions into the extended dataframe, then build May 11 features
3. Run ensemble predictor → enforce quantile monotonicity via sort

**Output:** `predictions.csv`

---

### `model_advanced.ipynb` — Advanced LEAR ensemble (experimental)

Implements the full academic LEAR feature set and compares three independent methods:

**LEAR feature set (60 features per hour-specific model)**

The key upgrade over `model.ipynb` is the *full price profile* as input instead of sparse lags:

- **D‑1 vector**: all 24 hourly prices from yesterday (`d1_h00` … `d1_h23`)
- **D‑7 vector**: all 24 hourly prices from the same weekday last week
- D‑1 min / max
- Weather: temperature, wind speed, solar radiation, renewable share
- Calendar: cyclical day-of-week and month encodings, weekend, Monday flags

Construction formula for hour *h*: `d1_h{j} = price.shift(h + 24 − j)` sampled at hour-*h* timestamps. This correctly maps `d1_h{h}` to yesterday's same hour and `d1_h{0..23}` to yesterday's full profile.

**Method A — LEAR-ElasticNet** (sklearn `QuantileRegressor`, 144 models)
- Standard scaling + L1-regularised linear quantile regression (α = 0.1, HiGHS LP solver)
- This is the original LEAR formulation from the academic literature
- Provides a linear, interpretable complement to the tree-based models

**Method B — LightGBM-LEAR-v2** (144 models)
- Same 60-feature LEAR input, LightGBM quantile objective
- Two-stage training identical to `model.ipynb`
- Parallelised over hours with `joblib.Parallel` for ~6–8× training speedup

**Method C — XGBoost-LEAR** (144 models, `objective='reg:quantileerror'`)
- Third diverse learner; particularly strong for the upper tail (p975)
- Gracefully skipped if XGBoost ≥ 2.0 is not available

**Validation results (pinball at q=0.45, last 8 weeks held out):**

| Method | DE-LU | ES |
|---|---|---|
| A — LEAR-ElasticNet | 9.48 | 6.03 |
| **B — LightGBM-LEAR-v2** | **8.49** | **5.07** |
| C — XGBoost | 8.67 | 5.35 |

**Ensemble weights** (validation-optimised via SLSQP for each hour × quantile):
- p025: LightGBM dominates (~85–90%)
- p50: LightGBM leads (~45–57%), XGBoost contributes (~20–32%), ElasticNet (~22%)
- p975: XGBoost dominates (~73–80%)

**Output:** `predictions_v2.csv`

---

## Data

All data is fetched automatically at runtime from public APIs. No manual downloads needed.

| File | Source | Period | Content |
|---|---|---|---|
| `de_lu_prices.csv` | Energy-Charts (Fraunhofer ISE) | 2022–2026-05-09 | DE-LU day-ahead prices (EUR/MWh) |
| `es_prices.csv` | Energy-Charts | 2022–2026-05-09 | ES day-ahead prices (EUR/MWh) |
| `de_lu_generation.csv` | Energy-Charts | 2022–2026-05-09 | Hourly generation mix + load (DE) |
| `es_generation.csv` | Energy-Charts | 2022–2026-05-09 | Hourly generation mix + load (ES) |
| `de_lu_weather_hist.csv` | Open-Meteo Archive | 2022–2026-05-04 | Temperature, wind, radiation (Germany centre) |
| `es_weather_hist.csv` | Open-Meteo Archive | 2022–2026-05-04 | Temperature, wind, radiation (Madrid) |
| `de_lu_weather_fcst.csv` | Open-Meteo Forecast | 2026-05-05 + 16 days | Weather forecast used as model input |
| `es_weather_fcst.csv` | Open-Meteo Forecast | 2026-05-05 + 16 days | Weather forecast used as model input |

Data CSVs are gitignored and re-fetched on each run from the API. The Open-Meteo archive API has a ~5-day lag; the forecast API with `past_days=16` fills the recent gap seamlessly.

---

## Predictions

| File | Model | DE-LU p50 mean | ES p50 mean |
|---|---|---|---|
| `predictions.csv` | `model.ipynb` ensemble | ~91 EUR/MWh | ~44 EUR/MWh |
| `predictions_v2.csv` | `model_advanced.ipynb` ensemble | ~101 EUR/MWh | ~68 EUR/MWh |

Both files cover Mon 11 May 02:00 CEST → Tue 12 May 01:00 CEST (24 rows, 1-hour frequency) and pass all sanity checks: correct timestamps, no NaNs, monotone quantiles (p025 ≤ p50 ≤ p975).

The v2 predictions are systematically higher — particularly for ES — because the full D-1/D-7 price vectors directly condition on the elevated May 4–10 price levels, whereas the sparse lags in `model.ipynb` average more broadly over history.

---

## Methodology notes

**Why hour-specific models?**
Day-ahead electricity prices have strongly heterogeneous intraday behaviour. A model trained on all hours simultaneously must learn 24 structurally different patterns simultaneously. Hour-specific models each specialise: the midday model learns solar suppression dynamics; the 07:00 model learns the morning demand ramp; the 20:00 model learns the evening peak. This consistently outperforms global models in the EPF literature (Uniejewski et al. 2019, Lago et al. 2021).

**Why quantile regression at q=0.45 rather than q=0.50?**
The competition's pinball loss asymmetry (overestimation penalised 1.22×) means the optimal point forecast sits below the conditional median. Training at q=0.45 directly minimises the evaluation loss.

**Why a bridge step for May 10?**
To predict May 11, we need `price_lag24h` = May 10 prices. These are not yet observed, so we first predict May 10 using actual May 9 data, then inject those predictions as pseudo-observed prices before computing May 11 features. Without this step, all lag features for May 11 are NaN, causing a cascade failure across all rolling statistics.

**References**
- Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021). *Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark.* Applied Energy.
- Uniejewski, B., Weron, R., Ziel, F. (2019). *Variance stabilizing transformations for electricity spot price forecasting.* IEEE Transactions on Power Systems.
