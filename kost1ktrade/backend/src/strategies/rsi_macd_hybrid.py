from .hybrid_base import HybridStrategy
from .rsi import RsiStrategy
from .macd import MacdStrategy
from typing import Dict, Any

class RsiMacdHybridStrategy(HybridStrategy):
    """
    A hybrid strategy that requires a consensus between RSI and MACD signals.

    - A 'BUY' signal is generated only if RSI is oversold AND MACD has a bullish crossover.
    - A 'SELL' signal is generated only if RSI is overbought AND MACD has a bearish crossover.
    """
    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Initializes the hybrid strategy by creating instances of the sub-strategies.

        This strategy accepts all parameters for both RsiStrategy and MacdStrategy.
        """
        # Create instances of the sub-strategies
        rsi_strategy = RsiStrategy(
            rsi_period=kwargs.get("rsi_period", 14),
            oversold_threshold=kwargs.get("oversold_threshold", 30),
            overbought_threshold=kwargs.get("overbought_threshold", 70)
        )

        macd_strategy = MacdStrategy(
            fast_period=kwargs.get("fast_period", 12),
            slow_period=kwargs.get("slow_period", 26),
            signal_period=kwargs.get("signal_period", 9)
        )

        # Pass the list of strategies to the parent class
        super().__init__(sub_strategies=[rsi_strategy, macd_strategy], **kwargs)

    def get_name(self) -> str:
        return "RsiMacdHybrid"
