import threading
from typing import Optional, TYPE_CHECKING

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
        self.signal_bot_thread: Optional[threading.Thread] = None
        self.signal_bot_stop_event: threading.Event = threading.Event()

        # State for the grid bot
        self.grid_bot_mode: str = "stopped"
        self.grid_bot_engine: Optional['TradingEngine'] = None
        self.grid_bot_thread: Optional[threading.Thread] = None
        self.grid_bot_stop_event: threading.Event = threading.Event()

# Global instance of the bot state
bot_state = BotState()
