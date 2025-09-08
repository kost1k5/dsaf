import pandas as pd
import numpy as np
import talib

def _calculate_ao(high, low):
    """Calculates Awesome Oscillator."""
    median_price = (high + low) / 2
    ao = talib.SMA(median_price, timeperiod=5) - talib.SMA(median_price, timeperiod=34)
    return ao

def _calculate_kc(high, low, close, timeperiod=20, atr_period=10, multiplier=2):
    """Calculates Keltner Channels."""
    kc_middle = talib.EMA(close, timeperiod=timeperiod)
    atr = talib.ATR(high, low, close, timeperiod=atr_period)
    kc_upper = kc_middle + (atr * multiplier)
    kc_lower = kc_middle - (atr * multiplier)
    return kc_upper, kc_middle, kc_lower

def _calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Helper to calculate daily VWAP efficiently."""
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: VWAP calculation requires a DatetimeIndex. Skipping.")
        return pd.Series(index=df.index, dtype='float64')

    # Use index.date for grouping if index is datetime
    grouped = df.groupby(df.index.date)

    cum_vol = grouped['volume'].transform('cumsum')
    cum_vol_price = (df['close'] * df['volume']).groupby(df.index.date).transform('cumsum')

    vwap = (cum_vol_price / cum_vol).fillna(0)
    return vwap

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the candle DataFrame with a variety of technical indicator features
    using the TA-Lib library.
    """
    print("Generating a rich set of TA features (using TA-Lib)...")

    # Create a copy to avoid modifying the original DataFrame
    df_feat = df.copy()

    # Prepare numpy arrays for TA-Lib
    open_p, high, low, close, volume = df_feat['open'].values, df_feat['high'].values, df_feat['low'].values, df_feat['close'].values, df_feat['volume'].values

    # Momentum Indicators
    df_feat['RSI_14'] = talib.RSI(close)
    macd, macdsignal, macdhist = talib.MACD(close)
    df_feat['MACD_12_26_9'] = macd
    df_feat['MACDs_12_26_9'] = macdsignal
    df_feat['MACDh_12_26_9'] = macdhist
    ppo, pposignal, ppohist = talib.PPO(close)
    df_feat['PPO_12_26_9'] = ppo
    df_feat['PPOs_12_26_9'] = pposignal
    df_feat['PPOh_12_26_9'] = ppohist
    df_feat['ROC_10'] = talib.ROC(close)
    slowk, slowd = talib.STOCH(high, low, close)
    df_feat['STOCHk_14_3_3'] = slowk
    df_feat['STOCHd_14_3_3'] = slowd
    df_feat['AO'] = _calculate_ao(high, low)

    # Trend Indicators
    df_feat['ADX_14'] = talib.ADX(high, low, close)
    aroondown, aroonup = talib.AROON(high, low)
    df_feat['AROOND_14'] = aroondown
    df_feat['AROONU_14'] = aroonup
    df_feat['AROONOSC_14'] = talib.AROONOSC(high, low)
    df_feat['PSAR'] = talib.SAR(high, low)
    df_feat['SMA_20'] = talib.SMA(close, timeperiod=20)
    df_feat['SMA_50'] = talib.SMA(close, timeperiod=50)
    df_feat['SMA_200'] = talib.SMA(close, timeperiod=200)

    # Volatility Indicators
    # The 'r' in 'ATRr' from pandas-ta means 'raw'. talib.ATR is raw by default.
    df_feat['ATRr_14'] = talib.ATR(high, low, close, timeperiod=14)
    upper, middle, lower = talib.BBANDS(close)
    df_feat['BBU_20_2.0'] = upper
    df_feat['BBM_20_2.0'] = middle
    df_feat['BBL_20_2.0'] = lower
    kc_upper, kc_middle, kc_lower = _calculate_kc(high, low, close)
    df_feat['KCUe_20_2'] = kc_upper
    df_feat['KCMe_20_2'] = kc_middle
    df_feat['KCLe_20_2'] = kc_lower

    # Volume Indicators
    df_feat['OBV'] = talib.OBV(close, volume)
    # Using MFI as a substitute for CMF
    df_feat['CMF_20'] = talib.MFI(high, low, close, volume, timeperiod=20)
    # Calculate VWAP, assuming the DataFrame index is a DatetimeIndex
    if isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat['VWAP_D'] = _calculate_vwap(df_feat)
    else:
        print("Warning: DataFrame index is not DatetimeIndex, cannot calculate VWAP.")
        df_feat['VWAP_D'] = np.nan

    # --- Feature Stationarity Transformations ---
    print("Transforming features to be stationary...")
    close_price = df_feat['close']

    # Normalize price-based indicators
    for col in df_feat.columns:
        # SMAs, PSAR, VWAP, and the moving average lines of BBands/KC
        if col.startswith(('SMA_', 'PSAR', 'BBM_', 'KCMe_', 'VWAP_')) and not col.endswith(('_pct', '_normalized')):
             if col in df_feat.columns:
                df_feat[f'{col}_normalized'] = (close_price / df_feat[col]) - 1
                df_feat.drop(columns=[col], inplace=True)

    # Normalize MACD lines (histogram is already stationary)
    for col in df_feat.columns:
        if col.startswith('MACD_') and not col.startswith('MACDh_') and not col.endswith('_normalized'):
            if col in df_feat.columns:
                df_feat[f'{col}_normalized'] = df_feat[col] / close_price
                df_feat.drop(columns=[col], inplace=True)

    # Transform OBV from cumulative to period-over-period change
    obv_col = next((col for col in df_feat.columns if col.startswith('OBV')), None)
    if obv_col:
        # Use the original column name for the new feature name for clarity
        df_feat[f'{obv_col}_pct_change'] = df_feat[obv_col].pct_change(periods=14) # Using a 14-period change
        df_feat.drop(columns=[obv_col], inplace=True)

    # Add custom time-based and return features
    # Ensure 'open_time' is a datetime object before accessing dt properties
    if 'open_time' in df_feat.columns and pd.api.types.is_datetime64_any_dtype(df_feat['open_time']):
        df_feat['hour'] = df_feat['open_time'].dt.hour
        df_feat['day_of_week'] = df_feat['open_time'].dt.dayofweek

    for n in [1, 2, 4, 8, 16]:
        df_feat[f'return_{n}h'] = df_feat['close'].pct_change(n)

    # Note: We do not drop NaNs here anymore. This will be handled in the main training
    # script after labels are also generated, ensuring we only drop rows that are
    # truly unusable for training.

    print(f"Feature generation complete. New shape: {df_feat.shape}")

    return df_feat

# --- CORRECTED FUNCTION ---
# Updated default atr_multiplier from 0.1 to a more standard 1.5
def create_labels(df: pd.DataFrame, look_forward_periods: int = 4, atr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Creates a binary target variable (label) for the classification model,
    filtering out noisy, sideways movements.

    The label is determined by comparing future returns to a dynamic threshold
    based on the Average True Range (ATR).

    - Label 1 (Up): If the future return is significantly positive.
    - Label 0 (Down): If the future return is significantly negative.
    - Sideways movements are dropped from the dataset.
    """
    # Ensure the ATR column from feature generation exists
    # We look for 'ATRr' (ATR raw) as generated by pandas_ta by default.
    atr_col = next((col for col in df.columns if 'ATRr' in col), None)
    if not atr_col:
        raise ValueError("ATR column (ATRr_*) not found in DataFrame. Please ensure it's generated in `create_features`.")

    # Calculate future returns (percentage change)
    df['future_return'] = df['close'].pct_change(look_forward_periods).shift(-look_forward_periods)

    # --- CRITICAL FIX: Calculate ATR percentage correctly ---
    # The ATR from pandas_ta is in raw price units, not percentage.
    # We must normalize it by the current price to compare it with returns.
    df['atr_pct'] = df[atr_col] / df['close']

    # Define dynamic thresholds based on volatility
    # Threshold = Multiplier * Normalized ATR
    df['up_threshold'] = atr_multiplier * df['atr_pct']
    df['down_threshold'] = -atr_multiplier * df['atr_pct']
    # --------------------------------------------------------

    # Assign labels based on thresholds
    df['target'] = np.nan
    df.loc[df['future_return'] > df['up_threshold'], 'target'] = 1  # 1 for Up
    df.loc[df['future_return'] < df['down_threshold'], 'target'] = -1 # -1 for Down

    # Diagnostic output to understand the labeling process
    print("\n--- Labeling Diagnostics ---")
    print(f"ATR Multiplier: {atr_multiplier}")
    print(f"Look Forward Periods: {look_forward_periods}")

    # Display the average threshold being used
    avg_threshold = df['up_threshold'].mean()
    print(f"Average movement threshold required: {avg_threshold*100:.4f}%")

    # Calculate statistics on labeling
    # Rows considered for labeling (dropping NaNs introduced by feature engineering and future returns)
    analysis_df = df.dropna(subset=['future_return', 'atr_pct'])
    total_rows = len(analysis_df)
    labeled_rows = analysis_df['target'].notna().sum()
    labeled_percentage = (labeled_rows / total_rows) * 100 if total_rows > 0 else 0

    print(f"\nTotal rows considered: {total_rows}")
    print(f"Rows labeled (significant movement): {labeled_rows}")
    print(f"Percentage of data kept: {labeled_percentage:.2f}%")

    if labeled_rows == 0:
        print("\nWARNING: No labels were generated. This might happen if the atr_multiplier is too high, ")
        print("or if the market was exceptionally stable during the period. ")
        print("Showing summary statistics of Future Returns:")
        print(df['future_return'].describe())
        print("Showing sample data with thresholds:")
        print(df[['close', atr_col, 'atr_pct', 'future_return', 'up_threshold']].tail(10))

    # Drop rows that don't meet the significance threshold (sideways movement)
    labeled_df = df.dropna(subset=['target']).copy()

    if not labeled_df.empty:
        labeled_df['target'] = labeled_df['target'].astype(int)
        print("Label Distribution:")
        print(labeled_df['target'].value_counts(normalize=True))

    print("--- End Labeling Diagnostics ---\n")

    # Clean up temporary columns
    # Use errors='ignore' in case the df is empty and columns don't exist
    labeled_df.drop(columns=['future_return', 'up_threshold', 'down_threshold', 'atr_pct'], inplace=True, errors='ignore')

    return labeled_df

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

    featured_df = create_features(sample_df.copy())
    print("\n--- Features Created (pandas_ta) ---")
    print(featured_df.head())
    print("\nColumns:", featured_df.columns.tolist())

    labeled_df = create_labels(featured_df.copy())
    print("\n--- Labels Created ---")
    if not labeled_df.empty:
        print(labeled_df[['open_time', 'close', 'target']].head())
