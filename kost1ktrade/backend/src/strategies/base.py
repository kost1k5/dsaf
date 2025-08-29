from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """

    @abstractmethod
    def generate_signals(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals based on historical candle data.

        :param candles_df: A pandas DataFrame with candle data.
                           It must contain 'open_time', 'open', 'high', 'low', 'close', 'volume' columns.
        :return: A pandas DataFrame with an added 'signal' column.
                 The signal can be 'BUY', 'SELL', or 'HOLD'.
        """
        pass
