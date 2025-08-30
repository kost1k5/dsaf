import pandas as pd
import pandas_ta as ta
from .base import BaseStrategy

class AwesomeOscillatorStrategy(BaseStrategy):
    """
    A strategy based on the Awesome Oscillator (AO).
    """
    def __init__(self,
                 fast_period: int = 5,
                 slow_period: int = 34):
        self.fast_period = fast_period
        self.slow_period = slow_period
        print(f"AwesomeOscillatorStrategy initialized with fast={fast_period}, slow={slow_period}")

    def generate_signals(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Awesome Oscillator and generates signals on zero-line crossover.
        - BUY signal when AO crosses above zero.
        - SELL signal when AO crosses below zero.
        """
        if not all(col in candles_df.columns for col in ['high', 'low']):
            raise ValueError("Candles DataFrame must contain 'high' and 'low' columns.")

        df = candles_df.copy()

        # Calculate Awesome Oscillator using pandas-ta
        # It uses the median price, so it needs high and low
        ao = df.ta.ao(fast=self.fast_period, slow=self.slow_period)
        df['ao'] = ao

        df['signal'] = 'HOLD'

        # Find where the crossover happened in the previous step
        previous_ao = df['ao'].shift(1)

        # A BUY signal is generated when AO crosses ABOVE zero
        buy_conditions = (df['ao'] > 0) & (previous_ao <= 0)

        # A SELL signal is generated when AO crosses BELOW zero
        sell_conditions = (df['ao'] < 0) & (previous_ao >= 0)

        df.loc[buy_conditions, 'signal'] = 'BUY'
        df.loc[sell_conditions, 'signal'] = 'SELL'

        return df
