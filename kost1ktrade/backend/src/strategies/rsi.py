import pandas as pd
from .base import BaseStrategy
from src.core.config import settings

class RsiStrategy(BaseStrategy):
    """
    A strategy based on the Relative Strength Index (RSI).
    """
    def __init__(self,
                 rsi_period: int = None,
                 oversold_threshold: int = 30,
                 overbought_threshold: int = 70):
        self.rsi_period = rsi_period or settings.INDICATORS.RSI_PERIOD
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        print(f"RsiStrategy initialized with period={self.rsi_period}, thresholds={self.oversold_threshold}/{self.overbought_threshold}")

    def generate_signals(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates RSI and generates BUY/SELL signals.
        - BUY signal when RSI is below the oversold threshold.
        - SELL signal when RSI is above the overbought threshold.
        Note: This is a simple implementation. A more robust one would check for crossovers.
        """
        if 'close' not in candles_df.columns:
            raise ValueError("Candles DataFrame must contain a 'close' column.")

        df = candles_df.copy()

        # Calculate RSI
        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0).ewm(alpha=1/self.rsi_period, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/self.rsi_period, adjust=False).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Generate signals
        df['signal'] = 'HOLD'
        df.loc[df['rsi'] < self.oversold_threshold, 'signal'] = 'BUY'
        df.loc[df['rsi'] > self.overbought_threshold, 'signal'] = 'SELL'

        # To generate signals only on crossover, we can do this:
        # position = pd.Series(np.where(df['rsi'] > self.overbought_threshold, -1, np.where(df['rsi'] < self.oversold_threshold, 1, 0)))
        # df['signal'] = np.where(position.diff() > 0, 'BUY', np.where(position.diff() < 0, 'SELL', 'HOLD'))

        return df

# Example Usage
if __name__ == '__main__':
    # Create some sample data with a clear trend
    data = {'close': [100, 105, 110, 115, 120, 115, 110, 105, 100, 95, 90, 85, 80, 85, 90, 95, 100]}
    sample_df = pd.DataFrame(data)

    # Initialize and run the strategy
    # Using a shorter period for demonstration on small dataset
    strategy = RsiStrategy(rsi_period=5)
    result_df = strategy.generate_signals(sample_df)

    print("\nRSI Strategy Results:")
    pd.set_option('display.max_rows', None)
    print(result_df[['close', 'rsi', 'signal']])

    print("\nSignals generated:")
    print(result_df[result_df['signal'] != 'HOLD'])
