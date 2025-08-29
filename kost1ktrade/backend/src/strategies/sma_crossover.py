import pandas as pd
from .base import BaseStrategy

class SmaCrossoverStrategy(BaseStrategy):
    """
    A strategy based on the crossover of two Simple Moving Averages (SMAs).
    """
    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates SMAs and generates BUY/SELL signals.
        - BUY signal when the short SMA crosses above the long SMA.
        - SELL signal when the short SMA crosses below the long SMA.
        """
        if 'close' not in candles_df.columns:
            raise ValueError("Candles DataFrame must contain a 'close' column.")

        # Create a copy to avoid modifying the original DataFrame
        df = candles_df.copy()

        # Calculate SMAs
        df['short_sma'] = df['close'].rolling(window=self.short_window, min_periods=1).mean()
        df['long_sma'] = df['close'].rolling(window=self.long_window, min_periods=1).mean()

        # Initialize signal column
        df['signal'] = 'HOLD'

        # Determine the crossover points
        # The position is 1 if short_sma > long_sma, 0 otherwise.
        # We use .shift(1) to get the previous period's position to detect the crossover event.
        position = pd.Series(df['short_sma'] > df['long_sma']).astype(int)

        # A BUY signal is when the position changes from 0 to 1
        df.loc[position.diff() == 1, 'signal'] = 'BUY'

        # A SELL signal is when the position changes from 1 to 0
        df.loc[position.diff() == -1, 'signal'] = 'SELL'

        return df

# Example Usage
if __name__ == '__main__':
    # Create some sample data
    data = {
        'close': [100, 102, 105, 103, 108, 110, 115, 112, 109, 105, 100, 98, 95, 99, 103, 108, 112, 118, 122, 120]
    }
    sample_df = pd.DataFrame(data)

    # Initialize and run the strategy
    strategy = SmaCrossoverStrategy(short_window=5, long_window=10)
    result_df = strategy.generate_signals(sample_df)

    print("SMA Crossover Strategy Results:")
    print(result_df[['close', 'short_sma', 'long_sma', 'signal']])

    print("\nSignals generated:")
    print(result_df[result_df['signal'] != 'HOLD'])
