import pandas as pd
from .base import BaseStrategy

class MacdStrategy(BaseStrategy):
    """
    A strategy based on the Moving Average Convergence Divergence (MACD).
    """
    def __init__(self,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        print(f"MacdStrategy initialized with periods: fast={fast_period}, slow={slow_period}, signal={signal_period}")

    def generate_signals(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates MACD and generates BUY/SELL signals on crossover.
        - BUY signal when the MACD line crosses above the signal line.
        - SELL signal when the MACD line crosses below the signal line.
        """
        if 'close' not in candles_df.columns:
            raise ValueError("Candles DataFrame must contain a 'close' column.")

        df = candles_df.copy()

        # Calculate Fast and Slow EMAs
        ema_fast = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow_period, adjust=False).mean()

        # Calculate MACD line
        df['macd'] = ema_fast - ema_slow

        # Calculate Signal line
        df['signal_line'] = df['macd'].ewm(span=self.signal_period, adjust=False).mean()

        # Calculate Histogram
        df['histogram'] = df['macd'] - df['signal_line']

        # Generate signals based on crossover
        df['signal'] = 'HOLD'

        # Find where the crossover happened in the previous step
        previous_macd = df['macd'].shift(1)
        previous_signal_line = df['signal_line'].shift(1)

        # A BUY signal is generated when the MACD crosses ABOVE the signal line
        buy_conditions = (df['macd'] > df['signal_line']) & (previous_macd <= previous_signal_line)

        # A SELL signal is generated when the MACD crosses BELOW the signal line
        sell_conditions = (df['macd'] < df['signal_line']) & (previous_macd >= previous_signal_line)

        df.loc[buy_conditions, 'signal'] = 'BUY'
        df.loc[sell_conditions, 'signal'] = 'SELL'

        return df
