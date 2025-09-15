import pandas as pd
import numpy as np
import talib
from scipy.stats import linregress

def calculate_slope(y):
    """
    Calculates the slope of a regression line for a given series.
    To be used with .apply() on a rolling window.
    """
    x = np.arange(len(y))
    valid_mask = ~np.isnan(y)
    y_valid = y[valid_mask]
    x_valid = x[valid_mask]
    if len(y_valid) < 2:
        return np.nan
    slope, _, _, _, _ = linregress(x_valid, y_valid)
    return slope

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the candle DataFrame with a comprehensive set of features for the hybrid model.
    """
    print("Generating a comprehensive feature set for the Hybrid Model...")

    df_feat = df.copy()
    if 'open_time' in df_feat.columns and not isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat.set_index('open_time', inplace=True, drop=False)

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_feat.columns for col in required_cols):
        raise ValueError(f"Input DataFrame is missing one of the required columns: {required_cols}")

    open_p, high, low, close, volume = df_feat['open'].values, df_feat['high'].values, df_feat['low'].values, df_feat['close'].values, df_feat['volume'].astype(float).values

    # --- Basic Indicators ---
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
    df_feat['PDI'] = talib.PLUS_DI(high, low, close, timeperiod=adx_period)
    df_feat['MDI'] = talib.MINUS_DI(high, low, close, timeperiod=adx_period)

    # --- Final DataFrame ---
    all_cols = df_feat.columns.tolist()
    ordered_cols = required_cols + ['open_time']
    feature_cols = [col for col in all_cols if col not in ordered_cols]
    final_cols = ordered_cols + sorted(feature_cols)
    df_final = df_feat[[col for col in final_cols if col in df_feat.columns]].copy()

    # Drop rows with NaN values resulting from indicator calculations
    df_final.dropna(inplace=True)

    df_final = df_final.reset_index(drop=True)

    print(f"Feature generation complete. Final shape: {df_final.shape}")
    return df_final
