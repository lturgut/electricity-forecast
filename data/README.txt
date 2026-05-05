Electricity Price Forecasting — Training Data
Team: larat · ETH Zurich
==============================================

All data is publicly available, freely downloadable, and licensed for
non-commercial use (research / educational purposes).

FILES
-----

de_lu_prices.csv
  Source : Energy-Charts (Fraunhofer ISE) — https://api.energy-charts.info/price?bzn=DE-LU
  Variables : timestamp (UTC), price (EUR/MWh, day-ahead auction)
  Period : 2022-01-01 → 2026-05-04
  Frequency : hourly
  Notes : Covers the German-Luxembourg bidding zone (EPEX SPOT DE-LU).
          Negative prices retained; values clipped to [-500, 3000] to remove
          obvious data artefacts from the 2022 crisis.

es_prices.csv
  Source : Energy-Charts (Fraunhofer ISE) — https://api.energy-charts.info/price?bzn=ES
  Variables : timestamp (UTC), price (EUR/MWh, day-ahead auction)
  Period : 2022-01-01 → 2026-05-04
  Frequency : hourly
  Notes : Spanish OMIE market. Prices are generally higher and less negative
          than DE-LU due to the isolated Iberian grid.

de_lu_generation.csv
  Source : Energy-Charts — https://api.energy-charts.info/public_power?country=de
  Variables : timestamp (UTC), columns per generation technology
              (wind_onshore, wind_offshore, solar, nuclear, hard_coal,
               lignite, gas, hydro, biomass, other, load, residual_load, ...)
  Period : 2022-01-01 → 2026-05-04
  Frequency : hourly
  Notes : Used to compute renewable_share = (wind_* + solar) / load.

es_generation.csv
  Source : Energy-Charts — https://api.energy-charts.info/public_power?country=es
  Variables : same schema as de_lu_generation.csv
  Period : 2022-01-01 → 2026-05-04
  Frequency : hourly
  Notes : Spain has significantly higher solar share and meaningful hydro;
          no nuclear after 2024 closure programme.

de_lu_weather_hist.csv
  Source : Open-Meteo Archive API — https://archive-api.open-meteo.com/v1/archive
  Location : 51.1657°N, 10.4515°E (geographic centre of Germany)
  Variables : temperature_2m (°C), wind_speed_10m (m/s),
              shortwave_radiation (W/m²)
  Period : 2022-01-01 → 2026-05-04
  Frequency : hourly UTC
  License : Open-Meteo data is CC BY 4.0

es_weather_hist.csv
  Source : Open-Meteo Archive API
  Location : 40.4168°N, -3.7038°W (Madrid — geographic centre of Spain)
  Variables : same as de_lu_weather_hist.csv
  Period : 2022-01-01 → 2026-05-04
  Frequency : hourly UTC
  Notes : Spanish irradiance is materially higher than German throughout
          the year — key driver of the different model weights for
          solar_radiation between the two zones.

de_lu_weather_fcst.csv
  Source : Open-Meteo Forecast API — https://api.open-meteo.com/v1/forecast
  Location : same as de_lu_weather_hist.csv
  Variables : same schema (temperature_2m, wind_speed_10m, shortwave_radiation)
  Period : 16-day rolling forecast from run date (2026-05-05)
  Frequency : hourly UTC
  Notes : Used as input features for the short-term model predictions.

es_weather_fcst.csv
  Source : Open-Meteo Forecast API
  Location : same as es_weather_hist.csv
  Variables : same schema
  Period : 16-day rolling forecast from run date (2026-05-05)
  Frequency : hourly UTC

DATA FLOW
---------

1. Historical prices (2022–May 4 2026)  ← energy-charts.info
2. Historical generation (2022–May 4 2026) ← energy-charts.info
3. Historical weather (2022–May 4 2026) ← open-meteo archive
4. Weather forecast (May 5–21 2026)     ← open-meteo forecast API

For longer horizons (> 14 days), the model uses only the STL seasonal
decomposition of historical prices — no weather forecast is used.

REPRODUCTION
------------

Run all data-fetch cells in larat_model.ipynb. If these CSV files are
already present in ./data/, the notebook skips re-downloading and loads
from cache. This ensures fully reproducible runs.
