import pandas as pd
import numpy as np
import talib

def create_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the candle DataFrame with the 'core quartet' of features for the
    hierarchical strategy.
    """
    print("Generating 'core quartet' of features...")

    df_feat = df.copy()
    # Ensure we have a DatetimeIndex for time-based features
    if 'open_time' in df_feat.columns and not isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat.set_index('open_time', inplace=True, drop=False)

    # Ensure the required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_feat.columns for col in required_cols):
        raise ValueError(f"Input DataFrame is missing one of the required columns: {required_cols}")

    high, low, close, volume = df_feat['high'].values, df_feat['low'].values, df_feat['close'].values, df_feat['volume'].values

    # --- Core Quartet Indicators ---
    df_feat['EMA_200'] = talib.EMA(close, timeperiod=200)
    df_feat['RSI_14'] = talib.RSI(close, timeperiod=14)
    df_feat['OBV'] = talib.OBV(close, volume)
    df_feat['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)

    # Keep original OHLCV and timestamp column for the strategy logic
    # The 'open_time' column is preserved by the set_index(drop=False) call
    final_cols = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'EMA_200', 'RSI_14', 'OBV', 'ATR_14'
    ]
    df_final = df_feat[final_cols]

    # Drop rows with NaNs created by the indicators (e.g., the first 200 for EMA200)
    df_final.dropna(inplace=True)

    print(f"Core feature generation complete. New shape: {df_final.shape}")

    return df_final

if __name__ == '__main__':
    # Example Usage
    data = {
        'open_time': pd.to_datetime(pd.date_range(start='2023-01-01', periods=300, freq='H')),
        'open': np.random.uniform(100, 102, 300),
        'high': np.random.uniform(102, 104, 300),
        'low': np.random.uniform(98, 100, 300),
        'close': np.random.uniform(100, 102, 300),
        'volume': np.random.uniform(1000, 2000, 300)
    }
    sample_df = pd.DataFrame(data)

    core_features_df = create_core_features(sample_df.copy())
    print("\n--- Core Features Created ---")
    print(core_features_df.head())
    print("\nColumns:", core_features_df.columns.tolist())
