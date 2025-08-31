import threading
from typing import Optional, TYPE_CHECKING, Dict

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from src.trading.engine import TradingEngine
    from src.strategies.base import BaseStrategy

class BotState:
    """
    A simple singleton-like class to hold the global state of the bot.
    """
    def __init__(self):
        # State for the signal-based bot
        self.signal_bot_mode: str = "stopped"  # 'stopped', 'real', 'demo'
        self.signal_bot_engine: Optional['TradingEngine'] = None
        self.signal_bot_strategy: Optional['BaseStrategy'] = None
        self.signal_bot_strategy_name: Optional[str] = None
        self.signal_bot_symbol: Optional[str] = None
        self.signal_bot_thread: Optional[threading.Thread] = None
        self.signal_bot_stop_event: threading.Event = threading.Event()

        # State for the grid bot
        self.grid_bot_mode: str = "stopped"
        self.grid_bot_engine: Optional['TradingEngine'] = None
        self.grid_bot_thread: Optional[threading.Thread] = None
        self.grid_bot_stop_event: threading.Event = threading.Event()
        self.grid_bot_config: Dict[str, Any] = {
            "symbol": "ETH/USDT",
            "grid_range_low": 1800,
            "grid_range_high": 2200,
            "num_grids": 10,
            "amount_per_grid": 50, # This is in quote currency (e.g., USDT)
        }

        # State for the master controller
        self.master_bot_mode: str = "stopped"
        self.master_bot_target_mode: str = "demo" # 'demo' or 'real'
        self.master_bot_stop_event: threading.Event = threading.Event()
        self.market_state: Optional[str] = None
        self.adx_value: Optional[float] = None

        # State for strategy activation
        # Initialize all known strategies as active by default.
        self.active_strategies: Dict[str, bool] = {
            "rsi": True,
            "sma_crossover": True,
            "macd": True,
            "stochastic": True,
            "awesome_oscillator": True,
            "parabolic_sar": True,
            "keltner_channels": True,
            "ichimoku": True,
            "bollinger_bands": True,
        }

# Global instance of the bot state
bot_state = BotState()
