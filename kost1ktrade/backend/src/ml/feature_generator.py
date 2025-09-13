import pandas as pd
import numpy as np
import talib

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the candle DataFrame with a focused set of indicators for the main ML pipeline:
    - EMA(200) for trend
    - RSI(14) for momentum
    - OBV for volume confirmation
    - ATRr_14 for volatility (and labeling)
    - ADX(14) for trend strength / regime filter
    """
    print("Generating a focused set of features (Core Quartet + ADX)...")

    df_feat = df.copy()
    if 'open_time' in df_feat.columns and not isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat.set_index('open_time', inplace=True, drop=False)

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_feat.columns for col in required_cols):
        raise ValueError(f"Input DataFrame is missing one of the required columns: {required_cols}")

    high, low, close, volume = df_feat['high'].values, df_feat['low'].values, df_feat['close'].values, df_feat['volume'].values

    # --- Generate Core Indicators ---
    df_feat['EMA_200'] = talib.EMA(close, timeperiod=200)
    df_feat['RSI_14'] = talib.RSI(close, timeperiod=14)
    df_feat['OBV'] = talib.OBV(close, volume)
    df_feat['ATRr_14'] = talib.ATR(high, low, close, timeperiod=14)
    df_feat['ADX_14'] = talib.ADX(high, low, close, timeperiod=14)

    # --- Final DataFrame ---
    # We keep the raw OHLCV data as it's needed by downstream processes like labeling.
    final_cols = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'EMA_200', 'RSI_14', 'OBV', 'ATRr_14', 'ADX_14'
    ]
    # Re-order and select final columns to ensure consistency
    df_final = df_feat[[col for col in final_cols if col in df_feat.columns]]

    df_final.dropna(inplace=True)

    # Reset index to turn 'open_time' back into a column for the calling script
    df_final = df_final.reset_index(drop=True)

    print(f"Feature generation complete. Final shape: {df_final.shape}")
    return df_final
