import pandas as pd
import pandas_ta as ta
from .base import BaseStrategy

class IchimokuStrategy(BaseStrategy):
    """
    A strategy based on the Ichimoku Cloud indicator.
    Signal logic: Crossover of Tenkan-sen and Kijun-sen, filtered by the cloud.
    """
    def __init__(self,
                 tenkan_period: int = 9,
                 kijun_period: int = 26,
                 senkou_b_period: int = 52):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        print(f"IchimokuStrategy initialized with tenkan={tenkan_period}, kijun={kijun_period}, senkou_b={senkou_b_period}")

    def generate_signals(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Ichimoku Cloud and generates signals.
        - BUY: Bullish TK cross (Tenkan > Kijun) while price is above the cloud.
        - SELL: Bearish TK cross (Tenkan < Kijun) while price is below the cloud.
        """
        if not all(col in candles_df.columns for col in ['high', 'low', 'close']):
            raise ValueError("Candles DataFrame must contain 'high', 'low', and 'close' columns.")

        df = candles_df.copy()

        # Calculate Ichimoku Cloud using pandas-ta
        ichimoku_df = df.ta.ichimoku(tenkan=self.tenkan_period, kijun=self.kijun_period, senkou=self.senkou_b_period)

        # Extract the relevant columns
        df['tenkan'] = ichimoku_df[f'ITS_{self.tenkan_period}']
        df['kijun'] = ichimoku_df[f'IKS_{self.kijun_period}']
        df['senkou_a'] = ichimoku_df[f'ISA_{self.tenkan_period}']
        df['senkou_b'] = ichimoku_df[f'ISB_{self.kijun_period}']

        df['signal'] = 'HOLD'

        # Previous state for crossover detection
        previous_tenkan = df['tenkan'].shift(1)
        previous_kijun = df['kijun'].shift(1)

        # Conditions for bullish crossover (TK Cross)
        tk_cross_up = (df['tenkan'] > df['kijun']) & (previous_tenkan <= previous_kijun)

        # Conditions for bearish crossover
        tk_cross_down = (df['tenkan'] < df['kijun']) & (previous_tenkan >= previous_kijun)

        # Cloud conditions
        price_above_cloud = (df['close'] > df['senkou_a']) & (df['close'] > df['senkou_b'])
        price_below_cloud = (df['close'] < df['senkou_a']) & (df['close'] < df['senkou_b'])

        # Final signal logic
        buy_conditions = tk_cross_up & price_above_cloud
        sell_conditions = tk_cross_down & price_below_cloud

        df.loc[buy_conditions, 'signal'] = 'BUY'
        df.loc[sell_conditions, 'signal'] = 'SELL'

        return df
