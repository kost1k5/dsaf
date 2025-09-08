import pandas as pd
import talib
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

        # Calculate Keltner Channels manually using TA-Lib
        basis = talib.EMA(df['close'], timeperiod=self.length)
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_length)

        upper_band = basis + (atr * self.multiplier)
        lower_band = basis - (atr * self.multiplier)

        df['signal'] = 'HOLD'

        # Find where the breakout/breakdown happened
        previous_close = df['close'].shift(1)
        previous_upper_band = upper_band.shift(1)
        previous_lower_band = lower_band.shift(1)

        # A BUY signal is generated on a close above the upper channel
        buy_conditions = (df['close'] > upper_band) & (previous_close <= previous_upper_band)

        # A SELL signal is generated on a close below the lower channel
        sell_conditions = (df['close'] < lower_band) & (previous_close >= previous_lower_band)

        df.loc[buy_conditions, 'signal'] = 'BUY'
        df.loc[sell_conditions, 'signal'] = 'SELL'

        return df
