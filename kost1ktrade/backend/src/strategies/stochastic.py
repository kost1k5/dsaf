import pandas as pd
import talib
from .base import BaseStrategy

class StochasticStrategy(BaseStrategy):
    """
    A strategy based on the Stochastic Oscillator.
    """
    def __init__(self,
                 k_period: int = 14,
                 d_period: int = 3,
                 oversold_threshold: int = 20,
                 overbought_threshold: int = 80):
        self.k_period = k_period
        self.d_period = d_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        print(f"StochasticStrategy initialized with k={k_period}, d={d_period}, thresholds={oversold_threshold}/{overbought_threshold}")

    def generate_signals(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Stochastic Oscillator and generates BUY/SELL signals.
        - BUY signal on %K crossing above the oversold threshold.
        - SELL signal on %K crossing below the overbought threshold.
        """
        if not all(col in candles_df.columns for col in ['high', 'low', 'close']):
            raise ValueError("Candles DataFrame must contain 'high', 'low', and 'close' columns.")

        df = candles_df.copy()

        # Calculate Stochastic Oscillator using TA-Lib
        # The '3' in the original pandas-ta column name STOCHk_14_3_3 refers to the smoothing period for %K,
        # which corresponds to `slowk_period` in TA-Lib's STOCH function.
        slowk, slowd = talib.STOCH(
            df['high'],
            df['low'],
            df['close'],
            fastk_period=self.k_period,
            slowk_period=3,
            slowk_matype=0, # SMA
            slowd_period=self.d_period,
            slowd_matype=0  # SMA
        )
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd

        df['signal'] = 'HOLD'

        # Find where the crossover happened in the previous step
        previous_k = df['stoch_k'].shift(1)

        # A BUY signal is generated when %K crosses ABOVE the oversold level
        buy_conditions = (df['stoch_k'] > self.oversold_threshold) & (previous_k <= self.oversold_threshold)

        # A SELL signal is generated when %K crosses BELOW the overbought level
        sell_conditions = (df['stoch_k'] < self.overbought_threshold) & (previous_k >= self.overbought_threshold)

        df.loc[buy_conditions, 'signal'] = 'BUY'
        df.loc[sell_conditions, 'signal'] = 'SELL'

        # Clean up temporary columns
        df.drop(columns=['stoch_k', 'stoch_d'], inplace=True)

        return df
