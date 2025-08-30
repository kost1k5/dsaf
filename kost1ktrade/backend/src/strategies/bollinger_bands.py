import pandas as pd
from .base import BaseStrategy
from src.core.config import settings

class BollingerBandsStrategy(BaseStrategy):
    """
    A strategy based on Bollinger Bands.
    """
    def __init__(self,
                 bb_period: int = None,
                 bb_std_dev: float = None):
        self.bb_period = bb_period or settings.INDICATORS.BB_PERIOD
        self.bb_std_dev = bb_std_dev or settings.INDICATORS.BB_STD_DEV
        print(f"BollingerBandsStrategy initialized with period={self.bb_period}, std_dev={self.bb_std_dev}")

    def generate_signals(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Bollinger Bands and generates BUY/SELL signals.
        - BUY signal when the close price is below the lower band.
        - SELL signal when the close price is above the upper band.
        """
        if 'close' not in candles_df.columns:
            raise ValueError("Candles DataFrame must contain a 'close' column.")

        df = candles_df.copy()

        # Calculate Bollinger Bands
        df['middle_band'] = df['close'].rolling(window=self.bb_period).mean()
        df['std_dev'] = df['close'].rolling(window=self.bb_period).std()
        df['upper_band'] = df['middle_band'] + (df['std_dev'] * self.bb_std_dev)
        df['lower_band'] = df['middle_band'] - (df['std_dev'] * self.bb_std_dev)

        # Generate signals
        df['signal'] = 'HOLD'
        df.loc[df['close'] < df['lower_band'], 'signal'] = 'BUY'
        df.loc[df['close'] > df['upper_band'], 'signal'] = 'SELL'

        return df

# Example Usage
if __name__ == '__main__':
    # Create some sample data with volatility
    data = {'close': [100, 102, 105, 103, 98, 95, 99, 104, 110, 115, 122, 118, 113, 108, 105, 102, 100]}
    sample_df = pd.DataFrame(data)

    # Initialize and run the strategy
    strategy = BollingerBandsStrategy(bb_period=5, bb_std_dev=2.0)
    result_df = strategy.generate_signals(sample_df)

    print("\nBollinger Bands Strategy Results:")
    pd.set_option('display.max_rows', None)
    print(result_df[['close', 'lower_band', 'middle_band', 'upper_band', 'signal']].round(2))

    print("\nSignals generated:")
    print(result_df[result_df['signal'] != 'HOLD'])
