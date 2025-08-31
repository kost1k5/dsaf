import pandas as pd
import pandas_ta as ta
from .base import BaseStrategy

class KeltnerChannelsStrategy(BaseStrategy):
    """
    A strategy based on Keltner Channels.
    """
    def __init__(self,
                 length: int = 20,
                 multiplier: float = 2.0,
                 atr_length: int = 14):
        self.length = length
        self.multiplier = multiplier
        self.atr_length = atr_length
        print(f"KeltnerChannelsStrategy initialized with length={length}, multiplier={multiplier}, atr_length={atr_length}")

    def generate_signals(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Keltner Channels and generates signals on channel breakouts.
        - BUY signal when price closes above the upper channel.
        - SELL signal when price closes below the lower channel.
        """
        if not all(col in candles_df.columns for col in ['high', 'low', 'close']):
            raise ValueError("Candles DataFrame must contain 'high', 'low', and 'close' columns.")

        df = candles_df.copy()

        # Calculate Keltner Channels using pandas-ta
        kc_df = df.ta.kc(length=self.length, scalar=self.multiplier, atr_length=self.atr_length)
        df = pd.concat([df, kc_df], axis=1)

        # Find the column names dynamically
        try:
            upper_band_col = next(col for col in df.columns if col.startswith('KCUe'))
            lower_band_col = next(col for col in df.columns if col.startswith('KCL'))
        except StopIteration:
            raise ValueError("Could not find Keltner Channel columns in the DataFrame after calculation.")

        df['signal'] = 'HOLD'

        # Find where the breakout/breakdown happened
        previous_close = df['close'].shift(1)
        previous_upper_band = df[upper_band_col].shift(1)
        previous_lower_band = df[lower_band_col].shift(1)

        # A BUY signal is generated on a close above the upper channel
        buy_conditions = (df['close'] > df[upper_band_col]) & (previous_close <= previous_upper_band)

        # A SELL signal is generated on a close below the lower channel
        sell_conditions = (df['close'] < df[lower_band_col]) & (previous_close >= previous_lower_band)

        df.loc[buy_conditions, 'signal'] = 'BUY'
        df.loc[sell_conditions, 'signal'] = 'SELL'

        return df
