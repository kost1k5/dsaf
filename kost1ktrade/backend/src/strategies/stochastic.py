import pandas as pd
import talib
from .base import BaseStrategy

class StochasticStrategy(BaseStrategy):
    """
    A strategy based on the Stochastic Oscillator, with candlestick confirmation.
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
        - BUY signal on %K crossing above the oversold threshold, confirmed by a bullish candlestick pattern.
        - SELL signal on %K crossing below the overbought threshold.
        """
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in candles_df.columns for col in required_cols):
            raise ValueError(f"Candles DataFrame must contain {', '.join(required_cols)} columns.")

        df = candles_df.copy()

        # --- Indicator Calculations ---
        # 1. Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'],
            df['low'],
            df['close'],
            fastk_period=self.k_period,
            slowk_period=3,
            slowk_matype=0, # SMA
            slowd_period=self.d_period,
            slowd_matype=0  # SMA
        )

        # 2. Bullish Candlestick Patterns for Confirmation
        df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])

        # --- Signal Logic ---
        df['signal'] = 'HOLD'

        # Previous state for crossover detection
        previous_k = df['stoch_k'].shift(1)

        # Condition 1: Stochastic crosses above the oversold threshold
        stoch_buy_signal = (df['stoch_k'] > self.oversold_threshold) & (previous_k <= self.oversold_threshold)

        # Condition 2: A bullish candlestick pattern (Hammer or Bullish Engulfing) is present
        # talib returns 100 for bullish patterns, -100 for bearish, 0 for none.
        pattern_confirmation = (df['hammer'] > 0) | (df['engulfing'] > 0)

        # Final BUY condition
        buy_conditions = stoch_buy_signal & pattern_confirmation

        # A SELL signal is generated when %K crosses BELOW the overbought level (no pattern confirmation needed)
        sell_conditions = (df['stoch_k'] < self.overbought_threshold) & (previous_k >= self.overbought_threshold)

        df.loc[buy_conditions, 'signal'] = 'BUY'
        df.loc[sell_conditions, 'signal'] = 'SELL'

        # Clean up temporary columns
        df.drop(columns=['stoch_k', 'stoch_d', 'hammer', 'engulfing'], inplace=True)

        return df
