import pandas as pd
import pandas_ta as ta
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

        # Calculate Parabolic SAR using pandas-ta
        psar = df.ta.psar(af=self.acceleration, max_af=self.maximum)

        # pandas-ta returns a DataFrame with PSARl (long), PSARs (short), AF, and RV columns
        # We need to determine the active SAR value and the trend direction
        df['psar'] = psar['PSARl_0.02_0.2'].fillna(psar['PSARs_0.02_0.2'])

        # A flip occurs when the SAR value switches from being above the low to below the high, or vice-versa
        # A simpler way to detect flips is to see when the long/short SAR columns switch from NaN
        is_bullish_flip = ~psar['PSARl_0.02_0.2'].isna() & psar['PSARl_0.02_0.2'].shift(1).isna()
        is_bearish_flip = ~psar['PSARs_0.02_0.2'].isna() & psar['PSARs_0.02_0.2'].shift(1).isna()

        df['signal'] = 'HOLD'
        df.loc[is_bullish_flip, 'signal'] = 'BUY'
        df.loc[is_bearish_flip, 'signal'] = 'SELL'

        return df
