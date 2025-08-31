import pandas as pd
import numpy as np

# --- Manual Indicator Implementations ---

def manual_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI manually."""
    delta = close.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def manual_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD, Signal Line, and Histogram manually."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the candle DataFrame with a variety of technical indicator features.
    """
    # Calculate indicators manually
    df['rsi'] = manual_rsi(df['close'])
    df['macd'], df['macd_signal'], df['macd_hist'] = manual_macd(df['close'])

    # Add some custom features
    df['hour'] = df['open_time'].dt.hour
    df['day_of_week'] = df['open_time'].dt.dayofweek

    # Calculate returns and volatility
    for n in [1, 2, 4, 8, 16]:
        df[f'return_{n}h'] = df['close'].pct_change(n)

    df['volatility_4h'] = df['close'].rolling(window=4).std()

    df = df.dropna()

    return df

def create_labels(df: pd.DataFrame, look_forward_periods: int = 4, threshold: float = 0.005) -> pd.DataFrame:
    """
    Creates the target variable (label) for the classification model.
    Label '1' (UP) if the price increases by the threshold within the look_forward period.
    Label '-1' (DOWN) if the price decreases by the threshold.
    Label '0' (SIDEWAYS) otherwise.
    """
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
        'open_time': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='H')),
        'open': np.random.uniform(100, 102, 100),
        'high': np.random.uniform(102, 104, 100),
        'low': np.random.uniform(98, 100, 100),
        'close': np.random.uniform(100, 102, 100),
        'volume': np.random.uniform(1000, 2000, 100)
    }
    sample_df = pd.DataFrame(data)

    featured_df = create_features(sample_df.copy())
    print("--- Features Created (Manual) ---")
    print(featured_df.head())

    labeled_df = create_labels(featured_df.copy())
    print("\n--- Labels Created ---")
    print(labeled_df[['close', 'target']].head())
