import pandas as pd
import talib
from .base import BaseStrategy

class ParabolicSarStrategy(BaseStrategy):
    """
    A strategy based on the Parabolic Stop and Reverse (SAR).
    """
    def __init__(self,
                 acceleration: float = 0.02,
                 maximum: float = 0.2):
        self.acceleration = acceleration
        self.maximum = maximum
        print(f"ParabolicSarStrategy initialized with acceleration={acceleration}, maximum={maximum}")

    def generate_signals(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Parabolic SAR and generates signals on trend flips.
        - BUY signal when the trend flips from bearish to bullish.
        - SELL signal when the trend flips from bullish to bearish.
        """
        if not all(col in candles_df.columns for col in ['high', 'low', 'close']):
            raise ValueError("Candles DataFrame must contain 'high', 'low', and 'close' columns.")

        df = candles_df.copy()

        # Calculate Parabolic SAR using TA-Lib
        df['psar'] = talib.SAR(df['high'], df['low'], acceleration=self.acceleration, maximum=self.maximum)

        # Determine trend direction based on price vs. SAR
        # Bullish trend when price is above SAR, Bearish when below
        df['trend'] = 1  # Default to bullish
        df.loc[df['close'] < df['psar'], 'trend'] = -1

        # A flip occurs when the trend changes from the previous period
        df['prev_trend'] = df['trend'].shift(1)

        is_bullish_flip = (df['trend'] == 1) & (df['prev_trend'] == -1)
        is_bearish_flip = (df['trend'] == -1) & (df['prev_trend'] == 1)

        df['signal'] = 'HOLD'
        df.loc[is_bullish_flip, 'signal'] = 'BUY'
        df.loc[is_bearish_flip, 'signal'] = 'SELL'

        # Clean up temporary columns
        df.drop(columns=['psar', 'trend', 'prev_trend'], inplace=True)

        return df
