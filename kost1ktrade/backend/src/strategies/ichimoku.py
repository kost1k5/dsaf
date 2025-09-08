import pandas as pd
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

    def _calculate_ichimoku(self, df: pd.DataFrame):
        """
        Calculates Ichimoku Cloud components manually.
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=self.tenkan_period).max()
        tenkan_low = low.rolling(window=self.tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=self.kijun_period).max()
        kijun_low = low.rolling(window=self.kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A)
        # Shifted forward by kijun_period
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(self.kijun_period)

        # Senkou Span B (Leading Span B)
        senkou_b_high = high.rolling(window=self.senkou_b_period).max()
        senkou_b_low = low.rolling(window=self.senkou_b_period).min()
        # Shifted forward by kijun_period
        senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(self.kijun_period)

        # The original code expects specific column names from pandas-ta
        # We will replicate them for compatibility with the signal logic.
        df[f'ITS_{self.tenkan_period}'] = tenkan_sen
        df[f'IKS_{self.kijun_period}'] = kijun_sen
        df[f'ISA_{self.tenkan_period}'] = senkou_a
        df[f'ISB_{self.kijun_period}'] = senkou_b # Note: pandas-ta names this with kijun_period

        return df

    def generate_signals(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Ichimoku Cloud and generates signals.
        - BUY: Bullish TK cross (Tenkan > Kijun) while price is above the cloud.
        - SELL: Bearish TK cross (Tenkan < Kijun) while price is below the cloud.
        """
        if not all(col in candles_df.columns for col in ['high', 'low', 'close']):
            raise ValueError("Candles DataFrame must contain 'high', 'low', and 'close' columns.")

        df = candles_df.copy()

        # Calculate Ichimoku components manually
        df = self._calculate_ichimoku(df)

        # Define column names based on the names we created
        tenkan_col = f'ITS_{self.tenkan_period}'
        kijun_col = f'IKS_{self.kijun_period}'
        senkou_a_col = f'ISA_{self.tenkan_period}'
        senkou_b_col = f'ISB_{self.kijun_period}'

        # Ensure all required columns exist before proceeding
        required_cols = [tenkan_col, kijun_col, senkou_a_col, senkou_b_col]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Ichimoku calculation failed. Missing columns: {[c for c in required_cols if c not in df.columns]}")

        df['signal'] = 'HOLD'

        # Previous state for crossover detection
        previous_tenkan = df[tenkan_col].shift(1)
        previous_kijun = df[kijun_col].shift(1)

        # Conditions for bullish crossover (TK Cross)
        tk_cross_up = (df[tenkan_col] > df[kijun_col]) & (previous_tenkan <= previous_kijun)

        # Conditions for bearish crossover
        tk_cross_down = (df[tenkan_col] < df[kijun_col]) & (previous_tenkan >= previous_kijun)

        # Cloud conditions
        price_above_cloud = (df['close'] > df[senkou_a_col]) & (df['close'] > df[senkou_b_col])
        price_below_cloud = (df['close'] < df[senkou_a_col]) & (df['close'] < df[senkou_b_col])

        # Final signal logic
        buy_conditions = tk_cross_up & price_above_cloud
        sell_conditions = tk_cross_down & price_below_cloud

        df.loc[buy_conditions, 'signal'] = 'BUY'
        df.loc[sell_conditions, 'signal'] = 'SELL'

        # Clean up temporary columns
        df.drop(columns=required_cols, inplace=True)

        return df
