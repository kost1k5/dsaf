import pandas as pd
import numpy as np
import pandas_ta as ta

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the candle DataFrame with a variety of technical indicator features
    using the pandas_ta library.
    """
    print("Generating a rich set of TA features...")

    # Create a copy to avoid modifying the original DataFrame
    df_feat = df.copy()

    # Use pandas_ta to calculate a wide range of indicators
    # This list covers Momentum, Trend, Volatility, and Volume indicators.
    custom_strategy = ta.Strategy(
        name="RichFeatureSet",
        description="A comprehensive set of indicators for ML",
        ta=[
            # Momentum
            {"kind": "rsi"},
            {"kind": "macd"},
            {"kind": "ppo"},
            {"kind": "roc"},
            {"kind": "stoch"},
            {"kind": "ao"},
            # Trend
            {"kind": "adx"},
            {"kind": "aroon"},
            {"kind": "psar"},
            {"kind": "sma", "length": 20},
            {"kind": "sma", "length": 50},
            {"kind": "sma", "length": 200},
            # Volatility
            {"kind": "atr"},
            {"kind": "bbands"},
            {"kind": "kc"},
            # Volume
            {"kind": "obv"},
            {"kind": "cmf"},
            {"kind": "vwap"},
        ]
    )

    # Apply the study to the DataFrame
    df_feat.ta.study(custom_strategy)

    # --- Feature Stationarity Transformations ---
    print("Transforming features to be stationary...")
    close_price = df_feat['close']

    # Normalize price-based indicators
    for col in df_feat.columns:
        # SMAs, PSAR, VWAP, and the moving average lines of BBands/KC
        if col.startswith(('SMA_', 'PSAR', 'BBM_', 'KCM_', 'VWAP_')) and not col.endswith(('_pct', '_normalized')):
            df_feat[f'{col}_normalized'] = (close_price / df_feat[col]) - 1
            df_feat.drop(columns=[col], inplace=True)

    # Normalize MACD lines (histogram is already stationary)
    for col in df_feat.columns:
        if col.startswith('MACD_') and not col.startswith('MACDh_') and not col.endswith('_normalized'):
            df_feat[f'{col}_normalized'] = df_feat[col] / close_price
            df_feat.drop(columns=[col], inplace=True)

    # Transform OBV from cumulative to period-over-period change
    obv_col = next((col for col in df_feat.columns if col.startswith('OBV')), None)
    if obv_col:
        df_feat[f'{obv_col}_pct_change'] = df_feat[obv_col].pct_change(periods=14) # Using a 14-period change
        df_feat.drop(columns=[obv_col], inplace=True)

    # Add custom time-based and return features
    df_feat['hour'] = df_feat['open_time'].dt.hour
    df_feat['day_of_week'] = df_feat['open_time'].dt.dayofweek

    for n in [1, 2, 4, 8, 16]:
        df_feat[f'return_{n}h'] = df_feat['close'].pct_change(n)

    # Note: We do not drop NaNs here anymore. This will be handled in the main training
    # script after labels are also generated, ensuring we only drop rows that are
    # truly unusable for training.

    print(f"Feature generation complete. New shape: {df_feat.shape}")

    return df_feat

def create_labels(df: pd.DataFrame, look_forward_periods: int = 4, atr_multiplier: float = 0.5) -> pd.DataFrame:
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
    atr_col = next((col for col in df.columns if 'ATRr' in col), None)
    if not atr_col:
        raise ValueError("ATR column not found in DataFrame. Please ensure it's generated in `create_features`.")

    # Calculate future returns
    df['future_return'] = df['close'].pct_change(look_forward_periods).shift(-look_forward_periods)

    # Define dynamic thresholds based on volatility
    df['up_threshold'] = atr_multiplier * df[atr_col] / 100 # ATR is in %, so divide by 100
    df['down_threshold'] = -atr_multiplier * df[atr_col] / 100

    # Assign labels based on thresholds
    df['target'] = np.nan
    df.loc[df['future_return'] > df['up_threshold'], 'target'] = 1  # 1 for Up
    df.loc[df['future_return'] < df['down_threshold'], 'target'] = -1 # -1 for Down

    # Drop rows that don't meet the significance threshold (sideways movement)
    labeled_df = df.dropna(subset=['target']).copy()
    labeled_df['target'] = labeled_df['target'].astype(int)

    # Clean up temporary columns
    labeled_df.drop(columns=['future_return', 'up_threshold', 'down_threshold'], inplace=True)

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
    print(labeled_df[['open_time', 'close', 'target']].head())
