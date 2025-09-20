import pandas as pd
import numpy as np
import talib
from scipy.stats import linregress

def calculate_slope(y):
    """
    Calculates the slope of a regression line for a given series.
    To be used with .apply() on a rolling window.
    """
    # The rolling window provides the y values. The x values are just a sequence.
    x = np.arange(len(y))
    # Filter out NaNs if they exist in the window
    valid_mask = ~np.isnan(y)
    y_valid = y[valid_mask]
    x_valid = x[valid_mask]
    if len(y_valid) < 2:  # Need at least 2 points to define a line
        return np.nan
    slope, _, _, _, _ = linregress(x_valid, y_valid)
    return slope

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the candle DataFrame with a comprehensive set of features for the hybrid model.
    - Basic indicators for the Confluence Strategy.
    - Specialized features for the ML confirmation layer.
    """
    print("Generating a comprehensive feature set for the Hybrid Model...")

    df_feat = df.copy()
    if 'open_time' in df_feat.columns and not isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat['open_time'] = pd.to_datetime(df_feat['open_time'])
        df_feat.set_index('open_time', inplace=True, drop=False)

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_feat.columns for col in required_cols):
        raise ValueError(f"Input DataFrame is missing one of the required columns: {required_cols}")

    # Use .values for talib functions for performance
    open_p, high, low, close, volume = df_feat['open'].values, df_feat['high'].values, df_feat['low'].values, df_feat['close'].values, df_feat['volume'].astype(float).values

    # --- Task 1.1: Basic Indicators (using strategy parameters) ---
    # These are the raw signals for the rule-based part of the strategy.
    ema_fast_period = 12
    ema_slow_period = 50
    rsi_period = 14
    obv_sma_period = 20
    atr_period = 14
    atr_sma_period = 20
    adx_period = 14

    df_feat['EMA_fast'] = talib.EMA(close, timeperiod=ema_fast_period)
    df_feat['EMA_slow'] = talib.EMA(close, timeperiod=ema_slow_period)
    df_feat['RSI'] = talib.RSI(close, timeperiod=rsi_period)
    df_feat['OBV'] = talib.OBV(close, volume)
    df_feat['OBV_SMA'] = talib.SMA(df_feat['OBV'], timeperiod=obv_sma_period)
    df_feat['ATR'] = talib.ATR(high, low, close, timeperiod=atr_period)
    df_feat['ATR_SMA'] = talib.SMA(df_feat['ATR'], timeperiod=atr_sma_period)
    df_feat['ADX'] = talib.ADX(high, low, close, timeperiod=adx_period)
    df_feat['PDI'] = talib.PLUS_DI(high, low, close, timeperiod=adx_period) # +DI
    df_feat['MDI'] = talib.MINUS_DI(high, low, close, timeperiod=adx_period) # -DI

    # --- Task 1.2: Specialized Features for ML ---
    # These features are designed to be more stationary and provide richer context for the ML model.
    print("  - Generating specialized ML features...")
    df_feat['ema_spread_normalized'] = (df_feat['EMA_fast'] - df_feat['EMA_slow']) / df_feat['close']
    df_feat['rsi_roc'] = df_feat['RSI'].diff() # Equivalent to RSI(t) - RSI(t-1)
    df_feat['adx_roc'] = df_feat['ADX'].diff()
    df_feat['di_spread'] = df_feat['PDI'] - df_feat['MDI']
    df_feat['atr_ratio'] = df_feat['ATR'] / talib.SMA(df_feat['ATR'], timeperiod=100)

    print("  - Calculating OBV slope (this may take a moment)...")
    # Use the helper function on a rolling window of OBV values
    df_feat['obv_slope'] = df_feat['OBV'].rolling(window=20).apply(calculate_slope, raw=True)

    df_feat['price_dist_ema'] = (df_feat['close'] - df_feat['EMA_slow']) / df_feat['EMA_slow']

    # --- NEW FEATURES ---
    print("  - Generating new features: VWAP, Time Features, Realized Volatility...")

    # VWAP (resets daily)
    df_feat['typical_price_vol'] = ((df_feat['high'] + df_feat['low'] + df_feat['close']) / 3) * df_feat['volume']
    df_feat['cum_vol'] = df_feat.groupby(df_feat.index.date)['volume'].cumsum()
    df_feat['cum_typical_price_vol'] = df_feat.groupby(df_feat.index.date)['typical_price_vol'].cumsum()
    df_feat['VWAP'] = df_feat['cum_typical_price_vol'] / df_feat['cum_vol']
    df_feat.drop(['typical_price_vol', 'cum_vol', 'cum_typical_price_vol'], axis=1, inplace=True)

    # Cyclical Time Features
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat.index.hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat.index.hour / 24)
    df_feat['dayofweek_sin'] = np.sin(2 * np.pi * df_feat.index.dayofweek / 7)
    df_feat['dayofweek_cos'] = np.cos(2 * np.pi * df_feat.index.dayofweek / 7)

    # Realized Volatility (30-period)
    df_feat['log_return'] = np.log(df_feat['close'] / df_feat['close'].shift(1))
    df_feat['realized_volatility'] = df_feat['log_return'].rolling(window=30).std() * np.sqrt(365) # Annualized
    df_feat.drop('log_return', axis=1, inplace=True)
    # --- END NEW FEATURES ---

    # --- PRIORITY 3 FEATURES (User Suggested) ---
    print("  - Generating user-suggested features (Priority 3)...")

    # 1. Market Structure Features
    ema_long_period = 200
    df_feat['EMA_long'] = talib.EMA(close, timeperiod=ema_long_period)
    df_feat['price_dist_ema_long'] = (df_feat['close'] - df_feat['EMA_long']) / df_feat['EMA_long']

    rolling_window = 200
    df_feat['high_rolling'] = df_feat['high'].rolling(window=rolling_window).max()
    df_feat['low_rolling'] = df_feat['low'].rolling(window=rolling_window).min()
    df_feat['time_since_high'] = df_feat.groupby((df_feat['high'] != df_feat['high_rolling']).cumsum()).cumcount()
    df_feat['time_since_low'] = df_feat.groupby((df_feat['low'] != df_feat['low_rolling']).cumsum()).cumcount()
    df_feat.drop(['high_rolling', 'low_rolling'], axis=1, inplace=True)

    # 2. Order Flow Proxy Features
    # Added a small epsilon to avoid division by zero in flat candles (high == low)
    epsilon = 1e-10
    df_feat['buying_pressure'] = (df_feat['high'] - df_feat['close']) / (df_feat['high'] - df_feat['low'] + epsilon)
    df_feat['selling_pressure'] = (df_feat['close'] - df_feat['low']) / (df_feat['high'] - df_feat['low'] + epsilon)
    df_feat['volume_weighted_candle'] = df_feat['volume'] * (df_feat['close'] - df_feat['open'])

    # 3. Volatility Regime Features
    # The user suggested ATR(14)/ATR(100). This is already implemented as 'atr_ratio'.
    # We will ensure it is calculated correctly.
    atr_short_period = 14
    atr_long_period = 100
    atr_short = talib.ATR(high, low, close, timeperiod=atr_short_period)
    atr_long = talib.ATR(high, low, close, timeperiod=atr_long_period)
    df_feat['atr_ratio_14_100'] = atr_short / (atr_long + epsilon)

    # New ATR ratio features as requested
    atr_5 = talib.ATR(high, low, close, timeperiod=5)
    atr_20 = talib.ATR(high, low, close, timeperiod=20)
    atr_200 = talib.ATR(high, low, close, timeperiod=200)
    df_feat['atr_ratio_5_20'] = atr_5 / (atr_20 + epsilon)
    df_feat['atr_ratio_20_200'] = atr_20 / (atr_200 + epsilon)
    # --- END PRIORITY 3 FEATURES ---


    # --- Task 1.3: Lagged Features ---
    print("  - Generating lagged features...")
    specialized_features = [
        'ema_spread_normalized', 'rsi_roc', 'adx_roc', 'di_spread',
        'atr_ratio', 'obv_slope', 'price_dist_ema',
        'VWAP', 'realized_volatility',
        # Adding the new Priority 3 features to the lag list
        'price_dist_ema_long', 'time_since_high', 'time_since_low',
        'buying_pressure', 'selling_pressure', 'volume_weighted_candle',
        'atr_ratio_14_100', 'atr_ratio_5_20', 'atr_ratio_20_200'
    ]

    for feature in specialized_features:
        for lag in [1, 2]:
            df_feat[f'{feature}_lag_{lag}'] = df_feat[feature].shift(lag)

    # --- Final DataFrame ---
    # Keep all generated columns. Downstream processes will select what they need.
    all_cols = df_feat.columns.tolist()

    # Ensure original OHLCV and open_time are at the beginning
    ordered_cols = required_cols + ['open_time']
    feature_cols = [col for col in all_cols if col not in ordered_cols]
    final_cols = ordered_cols + sorted(feature_cols) # Sort features alphabetically for consistency

    df_final = df_feat[[col for col in final_cols if col in df_feat.columns]].copy()

    # Drop rows with NaN values resulting from indicator calculations
    df_final.dropna(inplace=True)

    # Reset index to turn 'open_time' back into a column for the calling script
    df_final = df_final.reset_index(drop=True)

    print(f"Feature generation complete. Final shape: {df_final.shape}")
    return df_final
