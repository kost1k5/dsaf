import pandas as pd
import numpy as np

def apply_triple_barrier(close_prices: pd.Series, atr: pd.Series, tp_atr_mult: float, sl_atr_mult: float, time_limit_periods: int):
    """
    Applies the Triple-Barrier Method for labeling financial data for a 3-class model.
    This method creates labels based on which of three barriers is hit first.

    :param close_prices: Series of close prices.
    :param atr: Series of Average True Range values, used for dynamic barrier sizing.
    :param tp_atr_mult: Multiplier for the ATR to set the take-profit barrier.
    :param sl_atr_mult: Multiplier for the ATR to set the stop-loss barrier.
    :param time_limit_periods: Max number of periods to wait for a barrier to be hit.
    :return: A DataFrame with 'label' and 'event_end_time' columns.
    """
    print("Applying 3-Class Triple-Barrier Method...")
    labels = pd.Series(np.nan, index=close_prices.index)
    event_end_times = pd.Series(pd.NaT, index=close_prices.index)

    for i in range(len(close_prices) - time_limit_periods):
        entry_price = close_prices.iloc[i]
        current_atr = atr.iloc[i]

        # Define barriers
        upper_barrier = entry_price + (current_atr * tp_atr_mult)
        lower_barrier = entry_price - (current_atr * sl_atr_mult)

        # Define the window of future prices to check
        window_end_time = close_prices.index[i + time_limit_periods]
        future_prices = close_prices.iloc[i+1 : i+1+time_limit_periods]

        # Find barrier hits
        hit_upper_times = future_prices[future_prices >= upper_barrier].index
        hit_lower_times = future_prices[future_prices <= lower_barrier].index

        first_hit_upper = hit_upper_times[0] if not hit_upper_times.empty else None
        first_hit_lower = hit_lower_times[0] if not hit_lower_times.empty else None

        # Determine which barrier was hit first
        if first_hit_upper and first_hit_lower:
            if first_hit_upper < first_hit_lower:
                labels.iloc[i] = 1  # Buy
                event_end_times.iloc[i] = first_hit_upper
            else:
                labels.iloc[i] = -1 # Sell
                event_end_times.iloc[i] = first_hit_lower
        elif first_hit_upper:
            labels.iloc[i] = 1  # Buy
            event_end_times.iloc[i] = first_hit_upper
        elif first_hit_lower:
            labels.iloc[i] = -1 # Sell
            event_end_times.iloc[i] = first_hit_lower
        else:
            labels.iloc[i] = 0 # Hold
            event_end_times.iloc[i] = window_end_time

    # Combine into a single DataFrame
    result_df = pd.DataFrame({
        'label': labels,
        'event_end_time': event_end_times
    })

    print("Labeling complete.")
    return result_df.dropna()
