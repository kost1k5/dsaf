import pandas as pd
import numpy as np

def apply_triple_barrier(close_prices: pd.Series, atr: pd.Series, tp_atr_mult: float, sl_atr_mult: float, time_limit_periods: int):
    """
    Applies the Triple-Barrier Method for labeling financial data.
    This method creates labels based on which of three barriers is hit first:
    1. Upper barrier (take-profit)
    2. Lower barrier (stop-loss)
    3. Vertical barrier (time limit)

    :param close_prices: Series of close prices.
    :param atr: Series of Average True Range values, used for dynamic barrier sizing.
    :param tp_atr_mult: Multiplier for the ATR to set the take-profit barrier.
    :param sl_atr_mult: Multiplier for the ATR to set the stop-loss barrier.
    :param time_limit_periods: Max number of periods to wait for a barrier to be hit.
    :return: A Series containing the labels (1 for upper barrier, 0 for lower/vertical barrier).
    """
    print("Applying Triple-Barrier Method...")
    labels = pd.Series(np.nan, index=close_prices.index)

    for i in range(len(close_prices) - time_limit_periods):
        entry_price = close_prices.iloc[i]
        current_atr = atr.iloc[i]

        # Define barriers for the current event
        upper_barrier = entry_price + (current_atr * tp_atr_mult)
        lower_barrier = entry_price - (current_atr * sl_atr_mult)

        # Define the window of future prices to check
        future_prices = close_prices.iloc[i+1 : i+1+time_limit_periods]

        # Find the first time the upper or lower barrier is hit
        hit_upper_times = future_prices[future_prices >= upper_barrier].index
        hit_lower_times = future_prices[future_prices <= lower_barrier].index

        first_hit_upper = hit_upper_times[0] if not hit_upper_times.empty else None
        first_hit_lower = hit_lower_times[0] if not hit_lower_times.empty else None

        if first_hit_upper and first_hit_lower:
            # If both are hit, choose the one that happened first
            if first_hit_upper < first_hit_lower:
                labels.iloc[i] = 1 # Profit-take hit first
            else:
                labels.iloc[i] = 0 # Stop-loss hit first
        elif first_hit_upper:
            labels.iloc[i] = 1 # Profit-take hit
        elif first_hit_lower:
            labels.iloc[i] = 0 # Stop-loss hit
        else:
            # Vertical barrier hit (time limit)
            # Label based on the price at the end of the window
            end_price = future_prices.iloc[-1]
            if end_price > entry_price:
                labels.iloc[i] = 1
            else:
                labels.iloc[i] = 0

    print("Labeling complete.")
    return labels.dropna()
