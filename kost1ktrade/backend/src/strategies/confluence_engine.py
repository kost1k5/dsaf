import pandas as pd
import numpy as np

def generate_signals(dataframe: pd.DataFrame, strategy_type: str = 'advanced') -> pd.DataFrame:
    """
    Generates trading signals based on the Confluence Strategy.

    This function takes a DataFrame with pre-calculated indicators and adds a 'signal'
    column with three possible values:
    -  1: Long signal
    - -1: Short signal
    -  0: No signal

    Args:
        dataframe (pd.DataFrame): DataFrame containing OHLCV data and required indicators
                                  (EMAs, RSI, OBV, ATR, ADX, PDI, MDI).
        strategy_type (str): The type of strategy to apply.
                             'advanced' (default): Strategy 3 with ADX and DMI filters.
                             'basic': Strategy 2 without ADX and DMI filters.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'signal' column.
    """
    df = dataframe.copy()

    # --- Strategy Parameters ---
    # These are based on the .env configuration from Task 0
    adx_trend_threshold = 25
    rsi_entry_level = 50

    # --- Verify Required Columns ---
    required_cols = [
        'EMA_fast', 'EMA_slow', 'RSI', 'OBV', 'OBV_SMA',
        'ATR', 'ATR_SMA', 'ADX', 'PDI', 'MDI'
    ]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Input DataFrame is missing required indicator columns: {missing}")

    # --- Define Core Conditions (used in both strategies) ---

    # Trend Conditions
    long_trend_condition = df['EMA_fast'] > df['EMA_slow']
    short_trend_condition = df['EMA_fast'] < df['EMA_slow']

    # RSI Trigger Conditions (State-based logic)
    rsi_state_long = df['RSI'] > rsi_entry_level
    rsi_state_short = df['RSI'] < rsi_entry_level

    # Volume Confirmation
    volume_confirmation_long = df['OBV'] > df['OBV_SMA']
    volume_confirmation_short = df['OBV'] < df['OBV_SMA']

    # Volatility Filter (ATR must be elevated for a signal)
    volatility_filter = df['ATR'] > df['ATR_SMA']

    # --- Combine Conditions for Basic Strategy (Strategy 2) ---
    long_signal_basic = (
        long_trend_condition &
        rsi_state_long &
        volume_confirmation_long &
        volatility_filter
    )

    short_signal_basic = (
        short_trend_condition &
        rsi_state_short &
        volume_confirmation_short &
        volatility_filter
    )

    # --- Add Advanced Strategy Filters (Strategy 3) ---
    if strategy_type == 'advanced':
        # Regime Filter
        trend_regime_filter = df['ADX'] > adx_trend_threshold

        # DMI Confirmation
        dmi_confirmation_long = df['PDI'] > df['MDI']
        dmi_confirmation_short = df['PDI'] < df['MDI']

        # Combine all advanced conditions
        long_signal_advanced = long_signal_basic & trend_regime_filter & dmi_confirmation_long
        short_signal_advanced = short_signal_basic & trend_regime_filter & dmi_confirmation_short

        final_long_signal = long_signal_advanced
        final_short_signal = short_signal_advanced

    elif strategy_type == 'basic':
        final_long_signal = long_signal_basic
        final_short_signal = short_signal_basic

    else:
        raise ValueError(f"Unknown strategy_type: '{strategy_type}'. Use 'basic' or 'advanced'.")

    # --- Create Signal Column ---
    # Use np.select for clarity and performance
    conditions = [
        final_long_signal,
        final_short_signal
    ]
    choices = [1, -1]
    df['signal'] = np.select(conditions, choices, default=0)

    return df
