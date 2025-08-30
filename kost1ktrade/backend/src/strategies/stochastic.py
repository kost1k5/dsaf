import pandas as pd
import pandas_ta as ta
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

        # Calculate Stochastic Oscillator using pandas-ta
        stoch = df.ta.stoch(k=self.k_period, d=self.d_period)
        df['stoch_k'] = stoch[f'STOCHk_{self.k_period}_{self.d_period}_3']
        df['stoch_d'] = stoch[f'STOCHd_{self.k_period}_{self.d_period}_3']

        df['signal'] = 'HOLD'

        # Find where the crossover happened in the previous step
        previous_k = df['stoch_k'].shift(1)

        # A BUY signal is generated when %K crosses ABOVE the oversold level
        buy_conditions = (df['stoch_k'] > self.oversold_threshold) & (previous_k <= self.oversold_threshold)

        # A SELL signal is generated when %K crosses BELOW the overbought level
        sell_conditions = (df['stoch_k'] < self.overbought_threshold) & (previous_k >= self.overbought_threshold)

        df.loc[buy_conditions, 'signal'] = 'BUY'
        df.loc[sell_conditions, 'signal'] = 'SELL'

        return df
