import pandas as pd
from .base import BaseStrategy
from typing import List

class HybridStrategy(BaseStrategy):
    """
    A base class for strategies that combine signals from multiple sub-strategies.
    """
    def __init__(self, sub_strategies: List[BaseStrategy], **kwargs):
        """
        Initializes the HybridStrategy.

        Args:
            sub_strategies (List[BaseStrategy]): A list of instantiated strategy objects.
        """
        super().__init__(**kwargs)
        if not sub_strategies:
            raise ValueError("HybridStrategy requires at least one sub-strategy.")
        self.sub_strategies = sub_strategies

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates signals based on the consensus of all sub-strategies.

        A 'BUY' signal is generated only if all sub-strategies signal 'BUY'.
        A 'SELL' signal is generated only if all sub-strategies signal 'SELL'.
        Otherwise, the signal is 'HOLD'.

        Args:
            data (pd.DataFrame): The input data with OHLCV columns.

        Returns:
            pd.DataFrame: The data with a 'signal' column added.
        """
        # Create a copy to avoid modifying the original DataFrame
        df = data.copy()

        # --- (FIX) Bug Fix for Signal Overwriting ---
        # Generate signals for each sub-strategy independently and store them
        all_signals = {}
        for i, strategy in enumerate(self.sub_strategies):
            # Pass a copy of the data to each strategy to avoid side-effects
            strategy_df = strategy.generate_signals(data.copy())
            all_signals[f'signal_{i}'] = strategy_df['signal']

        # Combine all signal series into one DataFrame
        signals_df = pd.DataFrame(all_signals, index=df.index)

        # Determine the consensus signal
        def get_consensus(row):
            # Get all signals for the current row, drop NaNs
            signals = row.dropna().tolist()
            if not signals:
                return 'HOLD'

            first_signal = signals[0]
            # Check if all signals are the same and not 'HOLD'
            if first_signal != 'HOLD' and all(s == first_signal for s in signals):
                return first_signal
            return 'HOLD'

        # Apply the consensus logic row-wise
        df['signal'] = signals_df.apply(get_consensus, axis=1)

        return df
