"""
Generates predictions.csv for the evaluation window:
  From: 2026-05-11 00:00 UTC (02:00 CEST)
  To:   2026-05-11 23:00 UTC (2026-05-12 01:00 CEST) — inclusive
  24 hourly slots, 7 columns.
"""
import os, warnings
import requests
import numpy as np
import pandas as pd
import lightgbm as lgb
import holidays as hol
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────
ZONES = {
    'DE-LU': {'ec_bzn': 'DE-LU', 'ec_country': 'de', 'lat': 51.1657, 'lon': 10.4515},
    'ES':    {'ec_bzn': 'ES',    'ec_country': 'es', 'lat': 40.4168, 'lon': -3.7038},
}
CROSS_ZONE = {'DE-LU': 'ES', 'ES': 'DE-LU'}

TRAIN_START = '2024-01-01'
TRAIN_END   = '2026-05-10'   # extended so May 9-10 prices feed into lag features
CUTOFF_DAYS = 14
QUANTILES   = [0.025, 0.45, 0.975]
DATA_DIR    = './data/'

# New evaluation window: 24 hourly slots on May 11 (UTC)
EVAL_TIMESTAMPS  = pd.date_range(
    start='2026-05-11 00:00', periods=24, freq='1h', tz='UTC'
)
TRAIN_START_TS = pd.Timestamp(TRAIN_START, tz='UTC')

DE_HOLIDAYS = hol.Germany(years=range(2022, 2028))
ES_HOLIDAYS = hol.Spain(years=range(2022, 2028))

os.makedirs(DATA_DIR, exist_ok=True)
print(f'Evaluation window: {EVAL_TIMESTAMPS[0]} -> {EVAL_TIMESTAMPS[-1]} UTC')
print(f'Slots: {len(EVAL_TIMESTAMPS)}')

# ── Data loading ──────────────────────────────────────────────────────────────
EC_BASE = 'https://api.energy-charts.info'

def ec_prices(bzn, start, end, cache_path):
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    r = requests.get(f'{EC_BASE}/price', params={'bzn': bzn, 'start': start, 'end': end}, timeout=120)
    r.raise_for_status()
    d = r.json()
    ts = pd.to_datetime(d['unix_seconds'], unit='s', utc=True)
    df = pd.DataFrame({'price': d['price']}, index=ts)
    df.index.name = 'timestamp'
    df.to_csv(cache_path)
    return df

def ec_generation(country, start, end, cache_path):
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    r = requests.get(f'{EC_BASE}/public_power', params={'country': country, 'start': start, 'end': end}, timeout=180)
    r.raise_for_status()
    d = r.json()
    ts = pd.to_datetime(d['unix_seconds'], unit='s', utc=True)
    df = pd.DataFrame(index=ts)
    df.index.name = 'timestamp'
    for series in d.get('production_types', []):
        col = series['name'].lower().replace(' ', '_').replace('-', '_')
        df[col] = series.get('data', [None] * len(ts))
    df.to_csv(cache_path)
    return df

def om_weather(lat, lon, start, end, cache_path, forecast=False):
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    variables = 'temperature_2m,wind_speed_10m,shortwave_radiation'
    if forecast:
        url    = 'https://api.open-meteo.com/v1/forecast'
        params = {'latitude': lat, 'longitude': lon, 'hourly': variables,
                  'timezone': 'UTC', 'forecast_days': 16}
    else:
        # Open-Meteo archive cuts off at "today"; clip to avoid 400.
        # Forecast API call (forecast=True) covers any future window.
        archive_max = pd.Timestamp.utcnow().strftime('%Y-%m-%d')
        end_clipped = min(end, archive_max)
        url    = 'https://archive-api.open-meteo.com/v1/archive'
        params = {'latitude': lat, 'longitude': lon, 'start_date': start,
                  'end_date': end_clipped, 'hourly': variables, 'timezone': 'UTC'}
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    d = r.json()['hourly']
    df = pd.DataFrame({
        'temperature':     d['temperature_2m'],
        'wind_speed':      d['wind_speed_10m'],
        'solar_radiation': d['shortwave_radiation'],
    }, index=pd.to_datetime(d['time'], utc=True))
    df.index.name = 'timestamp'
    df.to_csv(cache_path)
    return df

def fetch_co2_prices(cache_path):
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    try:
        import yfinance as yf
        end = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        raw_yf = yf.Ticker('KRBN').history(start=TRAIN_START, end=end)
        daily = raw_yf[['Close']].rename(columns={'Close': 'co2_price'})
        daily.index = (daily.index.tz_convert('UTC') if daily.index.tz
                       else pd.to_datetime(daily.index).tz_localize('UTC'))
    except Exception as e:
        print(f'  WARNING: CO2 price fetch failed ({e}), using constant')
        daily = pd.DataFrame({'co2_price': 25.0},
                             index=pd.date_range(TRAIN_START, periods=1500, freq='D', tz='UTC'))
    hourly = daily.resample('1h').asfreq()
    hourly['co2_price'] = hourly['co2_price'].ffill().bfill()
    hourly.index.name = 'timestamp'
    hourly.to_csv(cache_path)
    return hourly

def fetch_gas_prices(cache_path):
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    try:
        import yfinance as yf
        end = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        raw_yf = yf.Ticker('NG=F').history(start=TRAIN_START, end=end)
        daily = raw_yf[['Close']].rename(columns={'Close': 'gas_price'})
        daily.index = (daily.index.tz_convert('UTC') if daily.index.tz
                       else pd.to_datetime(daily.index).tz_localize('UTC'))
    except Exception as e:
        print(f'  WARNING: gas price fetch failed ({e}), using constant')
        daily = pd.DataFrame({'gas_price': 2.5},
                             index=pd.date_range(TRAIN_START, periods=1500, freq='D', tz='UTC'))
    hourly = daily.resample('1h').asfreq()
    hourly['gas_price'] = hourly['gas_price'].ffill().bfill()
    hourly.index.name = 'timestamp'
    hourly.to_csv(cache_path)
    return hourly

print('\nLoading data...')
raw = {}
for zone, cfg in ZONES.items():
    slug = zone.lower().replace('-', '_')
    print(f'  {zone}...')
    raw[zone] = {
        'prices':       ec_prices(cfg['ec_bzn'], TRAIN_START, TRAIN_END,
                                  f'{DATA_DIR}{slug}_prices.csv'),
        'gen':          ec_generation(cfg['ec_country'], TRAIN_START, TRAIN_END,
                                      f'{DATA_DIR}{slug}_generation.csv'),
        'weather_hist': om_weather(cfg['lat'], cfg['lon'], TRAIN_START, TRAIN_END,
                                   f'{DATA_DIR}{slug}_weather_hist.csv'),
        'weather_fcst': om_weather(cfg['lat'], cfg['lon'], None, None,
                                   f'{DATA_DIR}{slug}_weather_fcst.csv', forecast=True),
    }
    print(f'    prices: {len(raw[zone]["prices"])} rows | '
          f'gen: {len(raw[zone]["gen"])} rows | '
          f'wx_hist: {len(raw[zone]["weather_hist"])} rows')

gas_prices = fetch_gas_prices(f'{DATA_DIR}gas_prices_ng.csv')
print(f'  gas_prices: {len(gas_prices)} rows (last: {gas_prices.index[-1].date()})')
co2_prices = fetch_co2_prices(f'{DATA_DIR}co2_prices_krbn.csv')
print(f'  co2_prices: {len(co2_prices)} rows (last: {co2_prices.index[-1].date()})')

# ── Preprocessing ─────────────────────────────────────────────────────────────
def extract_gen_features(gen_df):
    cols = gen_df.columns.tolist()
    wind_cols  = [c for c in cols if 'wind' in c]
    solar_cols = [c for c in cols if 'solar' in c or 'photovoltaic' in c]
    total_cols = [c for c in cols if 'total' in c or 'load' in c]
    gas_cols   = [c for c in cols if 'fossil_gas' in c]
    renew = gen_df[wind_cols + solar_cols].clip(lower=0).sum(axis=1)
    total = gen_df[total_cols].clip(lower=0).sum(axis=1) if total_cols else renew * 2
    out = pd.DataFrame(index=gen_df.index)
    out['renewable_share'] = (renew / total.replace(0, np.nan)).ffill().clip(0, 1)
    out['wind_mw']         = gen_df[wind_cols].clip(lower=0).sum(axis=1)
    out['solar_mw']        = gen_df[solar_cols].clip(lower=0).sum(axis=1)
    out['load_mw']         = gen_df[total_cols].clip(lower=0).sum(axis=1) if total_cols else pd.Series(0.0, index=gen_df.index)
    out['fossil_gas_mw']   = gen_df[gas_cols].clip(lower=0).sum(axis=1) if gas_cols else pd.Series(0.0, index=gen_df.index)
    return out

def merge_zone(zone):
    prices  = raw[zone]['prices'].copy()
    prices  = prices[prices.index >= TRAIN_START_TS]
    gen     = raw[zone]['gen'].copy()
    gen     = gen[gen.index >= TRAIN_START_TS]
    wx_hist = raw[zone]['weather_hist'].copy()
    wx_fcst = raw[zone]['weather_fcst'].copy()
    wx = pd.concat([wx_hist, wx_fcst]).sort_index()
    wx = wx[~wx.index.duplicated(keep='last')]
    gf = extract_gen_features(gen)
    df = prices.join(wx, how='outer').join(gf, how='outer')
    df = df.resample('1h').asfreq()
    df = df.ffill(limit=3)
    df['price'] = df['price'].clip(lower=-500, upper=3000)
    df = df.fillna(df.median(numeric_only=True))
    print(f'  {zone}: {len(df)} rows, {df.isna().sum().sum()} NaNs remaining')
    return df

print('\nPreprocessing...')
merged = {z: merge_zone(z) for z in ZONES}

# ── Feature engineering ───────────────────────────────────────────────────────
FEATURE_COLS = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_weekend', 'is_summer',
    'price_lag24h', 'price_lag48h', 'price_lag72h', 'price_lag120h', 'price_lag168h',
    'price_momentum',
    'price_roll_mean_24h', 'price_roll_std_24h',
    'price_roll_mean_48h', 'price_roll_std_48h',
    'price_roll_mean_72h', 'price_roll_std_72h',
    'price_roll_mean_168h', 'price_roll_std_168h',
    'price_roll_min_24h', 'price_roll_min_48h', 'price_roll_p10_24h',
    'temperature', 'wind_speed', 'solar_radiation',
    'renewable_share', 'wind_mw', 'solar_mw', 'load_mw', 'fossil_gas_mw',
    'is_holiday_de', 'is_holiday_es',
    'gas_price_lag1d', 'gas_price_lag7d',
    'co2_price_lag1d', 'co2_price_lag7d',
    'solar_rad_delta',
    'solar_surplus_ratio',
    'cross_price_lag24h', 'cross_price_lag48h',
]

def build_features(df, zone):
    idx = df.index
    X = pd.DataFrame(index=idx)
    X['hour_sin']   = np.sin(2 * np.pi * idx.hour / 24)
    X['hour_cos']   = np.cos(2 * np.pi * idx.hour / 24)
    X['dow_sin']    = np.sin(2 * np.pi * idx.dayofweek / 7)
    X['dow_cos']    = np.cos(2 * np.pi * idx.dayofweek / 7)
    X['month_sin']  = np.sin(2 * np.pi * idx.month / 12)
    X['month_cos']  = np.cos(2 * np.pi * idx.month / 12)
    X['is_weekend'] = (idx.dayofweek >= 5).astype(int)
    X['is_summer']  = ((idx.month >= 6) & (idx.month <= 8)).astype(int)
    for lag in [24, 48, 72, 120, 168]:
        X[f'price_lag{lag}h'] = df['price'].shift(lag)
    X['price_momentum'] = df['price'].shift(24) - df['price'].shift(168)
    ps = df['price'].shift(1)
    for w in [24, 48, 72, 168]:
        X[f'price_roll_mean_{w}h'] = ps.rolling(w).mean()
        X[f'price_roll_std_{w}h']  = ps.rolling(w).std()
    X['price_roll_min_24h']  = ps.rolling(24, min_periods=1).min()
    X['price_roll_min_48h']  = ps.rolling(48, min_periods=1).min()
    X['price_roll_p10_24h']  = ps.rolling(24, min_periods=1).quantile(0.1)
    X['temperature']     = df.get('temperature',     pd.Series(15.0, index=idx))
    X['wind_speed']      = df.get('wind_speed',      pd.Series(4.0,  index=idx))
    X['solar_radiation'] = df.get('solar_radiation', pd.Series(0.0,  index=idx))
    X['renewable_share'] = df.get('renewable_share', pd.Series(0.3,  index=idx))
    X['wind_mw']         = df.get('wind_mw',         pd.Series(500.0, index=idx))
    X['solar_mw']        = df.get('solar_mw',        pd.Series(0.0,   index=idx))
    X['load_mw']         = df.get('load_mw',         pd.Series(3000.0, index=idx))
    X['fossil_gas_mw']   = df.get('fossil_gas_mw',   pd.Series(500.0, index=idx))
    X['is_holiday_de'] = [int(ts.date() in DE_HOLIDAYS) for ts in idx]
    X['is_holiday_es'] = [int(ts.date() in ES_HOLIDAYS) for ts in idx]
    g = gas_prices['gas_price'].reindex(idx, method='ffill').bfill()
    X['gas_price_lag1d'] = g.shift(24)
    X['gas_price_lag7d']  = g.shift(168)
    c = co2_prices['co2_price'].reindex(idx, method='ffill').bfill()
    X['co2_price_lag1d'] = c.shift(24)
    X['co2_price_lag7d']  = c.shift(168)
    X['solar_rad_delta'] = df.get('solar_radiation', pd.Series(0.0, index=idx)).diff()
    X['solar_surplus_ratio'] = (
        df.get('solar_mw', pd.Series(0.0, index=idx)) /
        df.get('load_mw', pd.Series(3000.0, index=idx)).clip(lower=1.0)
    )
    other_p = merged[CROSS_ZONE[zone]]['price'].reindex(idx, method='ffill').bfill()
    X['cross_price_lag24h'] = other_p.shift(24)
    X['cross_price_lag48h'] = other_p.shift(48)
    return X[FEATURE_COLS]

print('\nBuilding features...')
features = {z: build_features(merged[z], z) for z in ZONES}
for z, X in features.items():
    print(f'  {z}: {X.shape}, NaN rows = {X.isna().any(axis=1).sum()}')

# ── Model training ────────────────────────────────────────────────────────────
WEIGHT_HALF_LIFE_DAYS = 180   # data 6 months old gets half the weight of today

def pinball(y_true, y_pred, q):
    r = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.where(r >= 0, q * r, (q - 1) * r)))

def sample_weights(index):
    """Exponential decay: weight = exp(-ln2 / half_life * days_ago), normalised to mean=1."""
    days_ago = np.array((index.max() - index).total_seconds() / 86_400)
    w = np.exp(-np.log(2) / WEIGHT_HALF_LIFE_DAYS * days_ago)
    return w / w.mean()

LGBM_BASE = dict(
    n_estimators=800, learning_rate=0.04, num_leaves=63,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.5, n_jobs=-1, verbose=-1,
)

def train_quantile_models(zone):
    X = features[zone]
    y = merged[zone]['price']
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]
    split = len(X) - 8 * 7 * 24
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]
    w_tr  = sample_weights(X_tr.index)
    w_all = sample_weights(X.index)
    final_models = {}
    for q in QUANTILES:
        params = {**LGBM_BASE, 'objective': 'quantile', 'alpha': q}
        m_probe = lgb.LGBMRegressor(**params)
        m_probe.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        best_iter = m_probe.best_iteration_
        val_pb = pinball(y_val, m_probe.predict(X_val), q)
        print(f'  {zone} q={q:.3f} -> val pinball={val_pb:.4f}  best_iter={best_iter}')
        m_final = lgb.LGBMRegressor(**{**params, 'n_estimators': best_iter})
        m_final.fit(X, y, sample_weight=w_all)
        final_models[q] = m_final
    return final_models

st_models = {}  # populated in ensemble loop

# ── Long-term model ───────────────────────────────────────────────────────────
class LongTermForecaster:
    def __init__(self, series):
        self.series = series.dropna()
        self._fit()

    def _fit(self):
        stl = STL(self.series, period=168, robust=True)
        res = stl.fit()
        trend = res.trend
        window_h = 28 * 24
        recent = trend.iloc[-window_h:]
        hours = np.arange(len(recent))
        self.trend_slope      = np.polyfit(hours, recent.values, 1)[0]
        self.trend_anchor_val  = trend.iloc[-1]
        self.trend_anchor_time = trend.index[-1]
        seasonal = res.seasonal
        self.seasonal_profile = (
            seasonal.groupby([seasonal.index.month, seasonal.index.hour]).mean()
        )
        resid = res.resid
        self.resid_q025 = np.percentile(resid, 2.5)
        self.resid_q975 = np.percentile(resid, 97.5)

    def predict(self, timestamps):
        rows = []
        for ts in timestamps:
            hours_ahead = (ts - self.trend_anchor_time).total_seconds() / 3600
            days_ahead  = hours_ahead / 24
            trend_val = self.trend_anchor_val + self.trend_slope * hours_ahead
            try:
                seasonal_val = self.seasonal_profile.loc[(ts.month, ts.hour)]
            except KeyError:
                seasonal_val = 0.0
            p50  = trend_val + seasonal_val
            scale = np.sqrt(max(days_ahead / CUTOFF_DAYS, 1.0))
            rows.append({'p025': p50 + self.resid_q025 * scale,
                         'p50':  p50,
                         'p975': p50 + self.resid_q975 * scale})
        return pd.DataFrame(rows, index=timestamps)

print('\nFitting long-term models...')
lt_models = {z: LongTermForecaster(merged[z]['price']) for z in ZONES}

# ── Eval feature construction ─────────────────────────────────────────────────
def build_eval_features(zone):
    df_hist = merged[zone].copy()
    wx_fcst = raw[zone]['weather_fcst']

    eval_df = pd.DataFrame(index=EVAL_TIMESTAMPS)
    eval_df.index.name = 'timestamp'
    eval_df['price'] = np.nan
    for col in ['temperature', 'wind_speed', 'solar_radiation']:
        eval_df[col] = wx_fcst[col].reindex(EVAL_TIMESTAMPS)

    keys = [(ts.month, ts.hour) for ts in EVAL_TIMESTAMPS]
    def smean(col):
        return df_hist[col].groupby([df_hist.index.month, df_hist.index.hour]).mean().reindex(keys).values

    solar_mw_base  = smean('solar_mw')
    wind_mw_base   = smean('wind_mw')
    load_mw_base   = smean('load_mw')

    # Scale solar_mw using actual forecast solar_radiation vs seasonal mean
    solar_rad_base = smean('solar_radiation')
    solar_scale = np.where(solar_rad_base > 10,
                           eval_df['solar_radiation'].values / solar_rad_base, 1.0)
    eval_df['solar_mw'] = solar_mw_base * np.clip(solar_scale, 0.0, 3.0)

    # Scale wind_mw using actual forecast wind_speed vs seasonal mean
    wind_speed_base = smean('wind_speed')
    wind_scale = np.where(wind_speed_base > 0.5,
                          eval_df['wind_speed'].values / wind_speed_base, 1.0)
    eval_df['wind_mw'] = wind_mw_base * np.clip(wind_scale, 0.0, 3.0)

    # Recompute renewable_share from adjusted wind + solar
    renew = eval_df['solar_mw'] + eval_df['wind_mw']
    eval_df['renewable_share'] = np.clip(renew / np.maximum(load_mw_base, 1.0), 0.0, 1.0)
    eval_df['load_mw']         = load_mw_base
    eval_df['fossil_gas_mw']   = smean('fossil_gas_mw')

    extended = pd.concat([df_hist, eval_df])
    extended = extended[~extended.index.duplicated(keep='last')].sort_index()
    return build_features(extended, zone).loc[EVAL_TIMESTAMPS]

print('\nBuilding eval features...')
eval_features = {z: build_eval_features(z) for z in ZONES}

# ── Regime-switching prediction ───────────────────────────────────────────────
def predict_window(timestamps, zone, feature_rows):
    now = pd.Timestamp.utcnow()
    results = []
    for i, ts in enumerate(timestamps):
        horizon_days = (ts - now).total_seconds() / 86_400
        if horizon_days <= CUTOFF_DAYS:
            x = feature_rows.iloc[i:i+1]
            # Fill any NaN features with column medians from training
            x = x.fillna(features[zone].median())
            preds = {q: float(st_models[zone][q].predict(x)[0]) for q in QUANTILES}
            row = {'p025': preds[0.025], 'p50': preds[0.45], 'p975': preds[0.975]}
        else:
            lt_row = lt_models[zone].predict([ts]).iloc[0]
            row = {'p025': lt_row['p025'], 'p50': lt_row['p50'], 'p975': lt_row['p975']}
        row['p025'], row['p50'], row['p975'] = sorted([row['p025'], row['p50'], row['p975']])
        results.append(row)
    return pd.DataFrame(results, index=timestamps)

N_ENSEMBLE = 5
print(f'\nRunning {N_ENSEMBLE}-model ensemble...')
all_preds = {z: [] for z in ZONES}
for run in range(N_ENSEMBLE):
    LGBM_BASE['random_state'] = run * 17
    for zone in ZONES:
        st_models[zone] = train_quantile_models(zone)
    for z in ZONES:
        all_preds[z].append(predict_window(EVAL_TIMESTAMPS, z, eval_features[z]))

eval_preds = {z: pd.concat(all_preds[z]).groupby(level=0).mean() for z in ZONES}

# ── Interval widening for high-solar hours ────────────────────────────────────
def widen_solar_intervals(pred_df, solar_series):
    """Expand PI proportionally to solar radiation (model underestimates uncertainty at solar peaks)."""
    result = pred_df.copy()
    sol = solar_series.reindex(result.index).fillna(0.0)
    factor = (1.0 + (sol - 150).clip(lower=0) / 400).clip(upper=2.0)
    lo_half = (result['p50'] - result['p025']).clip(lower=0)
    hi_half = (result['p975'] - result['p50']).clip(lower=0)
    result['p025'] = result['p50'] - lo_half * factor
    result['p975'] = result['p50'] + hi_half * factor
    return result

for z in ZONES:
    eval_preds[z] = widen_solar_intervals(eval_preds[z], eval_features[z]['solar_radiation'])

# ── Save predictions.csv ──────────────────────────────────────────────────────
def format_cest(ts_utc):
    """UTC -> CEST string (+02:00)"""
    cest = ts_utc + pd.Timedelta(hours=2)
    return cest.strftime('%Y-%m-%dT%H:%M:%S+02:00')

rows = []
for ts in EVAL_TIMESTAMPS:
    de = eval_preds['DE-LU'].loc[ts]
    es = eval_preds['ES'].loc[ts]
    rows.append({
        'timestamp':   format_cest(ts),
        'DE-LU p025':  round(float(de['p025']), 2),
        'DE-LU p50':   round(float(de['p50']),  2),
        'DE-LU p975':  round(float(de['p975']), 2),
        'ES p025':     round(float(es['p025']), 2),
        'ES p50':      round(float(es['p50']),  2),
        'ES p975':     round(float(es['p975']), 2),
    })

out_df = pd.DataFrame(rows)
out_df.to_csv('predictions.csv', index=False)
print(f'\nSaved {len(out_df)} rows to predictions.csv')
print(out_df.to_string(index=False))

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert len(out_df) == 24
assert list(out_df.columns) == ['timestamp', 'DE-LU p025', 'DE-LU p50', 'DE-LU p975',
                                 'ES p025', 'ES p50', 'ES p975']
assert out_df.isna().sum().sum() == 0
ts_series = pd.to_datetime(out_df['timestamp'])
assert (ts_series.diff().iloc[1:] == pd.Timedelta('1h')).all()
for zp in ['DE-LU', 'ES']:
    assert ((out_df[f'{zp} p025'] <= out_df[f'{zp} p50']) &
            (out_df[f'{zp} p50']  <= out_df[f'{zp} p975'])).all()
print('\nAll sanity checks passed.')
