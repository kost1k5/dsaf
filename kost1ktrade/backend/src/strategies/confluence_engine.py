import pandas as pd
import numpy as np

def generate_signals(dataframe: pd.DataFrame, strategy_type: str = 'advanced') -> pd.DataFrame:
    """
    Generates trading signals based on the Confluence Strategy.
    This version is simplified to generate more signals for ML training.
    """
    df = dataframe.copy()

    # --- Strategy Parameters ---
    rsi_entry_level = 50

    # --- Verify Required Columns ---
    required_cols = ['EMA_fast', 'EMA_slow', 'RSI']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Input DataFrame is missing required indicator columns: {missing}")

    # --- Define Core Conditions ---

    # Trend Conditions
    long_trend_condition = df['EMA_fast'] > df['EMA_slow']
    short_trend_condition = df['EMA_fast'] < df['EMA_slow']

    # RSI Trigger Conditions
    rsi_state_long = df['RSI'] > rsi_entry_level
    rsi_state_short = df['RSI'] < rsi_entry_level

    # --- Combine Conditions for the Simplified Strategy ---
    final_long_signal = long_trend_condition & rsi_state_long
    final_short_signal = short_trend_condition & rsi_state_short

    # --- Create Signal Column ---
    conditions = [
        final_long_signal,
        final_short_signal
    ]
    choices = [1, -1]
    df['signal'] = np.select(conditions, choices, default=0)

    return df
