import pandas as pd
from .base import BaseStrategy

class PairsTradingStrategy(BaseStrategy):
    """
    A strategy for statistical arbitrage based on pairs trading.

    This strategy assumes the input DataFrame to `generate_signals` is specially
    prepared to contain the close prices of two assets, named 'close_asset1'
    and 'close_asset2'.
    """

    def __init__(self, window: int = 20, z_threshold: float = 2.0, close_threshold: float = 0.1):
        """
        Initializes the stateless PairsTradingStrategy.

        Args:
            window (int): The rolling window to calculate the Z-score.
            z_threshold (float): The Z-score value to trigger entry signals.
            close_threshold (float): The Z-score value to trigger exit signals (crossing zero).
        """
        super().__init__()
        self.window = window
        self.z_threshold = z_threshold
        self.close_threshold = close_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates stateless trading signals based on the Z-score of the price ratio spread.
        The controller is responsible for managing the position state.

        Args:
            data (pd.DataFrame): A DataFrame containing 'close_asset1' and 'close_asset2'.

        Returns:
            pd.DataFrame: The data with a 'signal' column added.
                          Signals can be 'BUY_PAIR', 'SELL_PAIR', 'CLOSE_PAIR', 'HOLD'.
        """
        if 'close_asset1' not in data.columns or 'close_asset2' not in data.columns:
            raise ValueError("Input DataFrame must contain 'close_asset1' and 'close_asset2' columns.")

        df = data.copy()

        # 1. Calculate the spread (using price ratio)
        df['spread'] = df['close_asset1'] / df['close_asset2']

        # 2. Calculate moving average and standard deviation of the spread
        df['spread_ma'] = df['spread'].rolling(window=self.window).mean()
        df['spread_std'] = df['spread'].rolling(window=self.window).std()

        # 3. Calculate the Z-score
        df['z_score'] = (df['spread'] - df['spread_ma']) / df['spread_std']

        # 4. Generate stateless signals
        conditions = [
            (df['z_score'] > self.z_threshold),
            (df['z_score'] < -self.z_threshold),
            (abs(df['z_score']) < self.close_threshold)
        ]
        choices = ['SELL_PAIR', 'BUY_PAIR', 'CLOSE_PAIR']

        df['signal'] = np.select(conditions, choices, default='HOLD')

        return df
