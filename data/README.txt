Electricity Price Forecasting — Training Data
==============================================

All data is publicly available and freely downloadable.

FILES
-----

de_lu_prices.csv
  Source    : Energy-Charts (Fraunhofer ISE)
              https://api.energy-charts.info/price?bzn=DE-LU
  Variables : timestamp (UTC), price (EUR/MWh, day-ahead auction)
  Period    : 2024-01-01 to 2026-05-10
  Frequency : hourly
  Notes     : German-Luxembourg bidding zone (EPEX SPOT DE-LU).
              Negative prices retained; clipped to [-500, 3000] EUR/MWh.

es_prices.csv
  Source    : Energy-Charts (Fraunhofer ISE)
              https://api.energy-charts.info/price?bzn=ES
  Variables : timestamp (UTC), price (EUR/MWh, day-ahead auction)
  Period    : 2024-01-01 to 2026-05-10
  Frequency : hourly
  Notes     : Spanish OMIE market. Generally higher and less negative than
              DE-LU due to the isolated Iberian grid.

de_lu_generation.csv
  Source    : Energy-Charts
              https://api.energy-charts.info/public_power?country=de
  Variables : timestamp (UTC), one column per generation technology
              (wind_onshore, wind_offshore, solar, nuclear, hard_coal,
               lignite, fossil_gas, hydro, biomass, load, ...)
  Period    : 2024-01-01 to 2026-05-10
  Frequency : hourly
  Notes     : Used to compute wind_mw, solar_mw, load_mw, fossil_gas_mw,
              renewable_share, and solar_surplus_ratio features.

es_generation.csv
  Source    : Energy-Charts
              https://api.energy-charts.info/public_power?country=es
  Variables : same schema as de_lu_generation.csv
  Period    : 2024-01-01 to 2026-05-10
  Frequency : hourly
  Notes     : Spain has higher solar share; meaningful hydro component.

de_lu_weather_hist.csv
  Source    : Open-Meteo Archive API
              https://archive-api.open-meteo.com/v1/archive
  Location  : 51.1657 N, 10.4515 E (geographic centre of Germany)
  Variables : temperature (degC), wind_speed (m/s), solar_radiation (W/m2)
  Period    : 2024-01-01 to 2026-05-10
  Frequency : hourly UTC
  License   : CC BY 4.0

es_weather_hist.csv
  Source    : Open-Meteo Archive API
  Location  : 40.4168 N, -3.7038 W (Madrid)
  Variables : same as de_lu_weather_hist.csv
  Period    : 2024-01-01 to 2026-05-10
  Frequency : hourly UTC

de_lu_weather_fcst.csv
  Source    : Open-Meteo Forecast API
              https://api.open-meteo.com/v1/forecast
  Location  : same as de_lu_weather_hist.csv
  Variables : same schema (temperature, wind_speed, solar_radiation)
  Period    : 16-day rolling forecast from run date (2026-05-09)
  Frequency : hourly UTC
  Notes     : Used as eval-window input features (forecast solar radiation
              scaled to adjust solar_mw for May 11 predictions).

es_weather_fcst.csv
  Source    : Open-Meteo Forecast API
  Location  : same as es_weather_hist.csv
  Variables : same schema
  Period    : 16-day rolling forecast from run date (2026-05-09)
  Frequency : hourly UTC

gas_prices_ng.csv
  Source    : Yahoo Finance — Henry Hub Natural Gas Futures (NG=F)
              via yfinance Python library
  Variables : timestamp (UTC), gas_price (USD/MMBtu)
  Period    : 2024-01-01 to 2026-05-08 (last trading day available)
  Frequency : daily, forward-filled to hourly
  Notes     : Used as gas_price_lag1d and gas_price_lag7d features.
              Henry Hub is a US benchmark but correlates with European
              gas sentiment and TTF directionally.

co2_prices_krbn.csv
  Source    : Yahoo Finance — KraneShares Global Carbon Strategy ETF (KRBN)
              via yfinance Python library
  Variables : timestamp (UTC), co2_price (USD, ETF closing price)
  Period    : 2024-01-01 to 2026-05-08 (last trading day available)
  Frequency : daily, forward-filled to hourly
  Notes     : Used as co2_price_lag1d and co2_price_lag7d features.
              KRBN tracks California Carbon Allowances and EU ETS credits,
              serving as a proxy for carbon cost pass-through to electricity.

DATA FLOW
---------

1. Historical prices (2024-01 to 2026-05-10)      <- energy-charts.info
2. Historical generation (2024-01 to 2026-05-10)  <- energy-charts.info
3. Historical weather (2024-01 to 2026-05-10)     <- open-meteo archive
4. Weather forecast (16-day rolling from May 9)   <- open-meteo forecast
5. Gas prices (daily, 2024-01 to 2026-05-08)      <- Yahoo Finance NG=F
6. CO2 prices (daily, 2024-01 to 2026-05-08)      <- Yahoo Finance KRBN

REPRODUCTION
------------

Run all cells in model.ipynb. If these CSV files are already present in
./data/, the notebook skips re-downloading and loads from cache.
To force a refresh, delete the relevant CSV and re-run.
