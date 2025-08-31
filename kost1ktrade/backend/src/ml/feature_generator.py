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
        ]
    )

    # Apply the study to the DataFrame
    df_feat.ta.study(custom_strategy)

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

def create_labels(df: pd.DataFrame, look_forward_periods: int = 4, threshold: float = 0.005) -> pd.DataFrame:
    """
    Creates the target variable (label) for the classification model.
    Label '1' (UP) if the price increases by the threshold within the look_forward period.
    Label '-1' (DOWN) if the price decreases by the threshold.
    Label '0' (SIDEWAYS) otherwise.
    """
    # Make sure to use the original close prices for label generation
    df['future_return'] = df['close'].pct_change(look_forward_periods).shift(-look_forward_periods)

    df['target'] = 0
    df.loc[df['future_return'] > threshold, 'target'] = 1
    df.loc[df['future_return'] < -threshold, 'target'] = -1

    df = df.drop(columns=['future_return'])
    df = df.dropna(subset=['target'])

    return df

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
