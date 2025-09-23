import threading
import json
from typing import Optional, TYPE_CHECKING, Dict, Any

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from src.trading.engine import TradingEngine
    from src.strategies.base import BaseStrategy

STRATEGY_PARAMS_FILE = 'strategy_params.json'

class BotState:
    """
    A thread-safe singleton class to hold the application's state.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BotState, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """
        Initializer, will only run once for the singleton instance.
        """
        with self._lock:
            if self._initialized:
                return

            # State for the signal-based bot
            self.signal_bot_mode: str = "stopped"
            self.signal_bot_engine: Optional['TradingEngine'] = None
            self.signal_bot_strategy: Optional['BaseStrategy'] = None
            self.signal_bot_strategy_name: Optional[str] = None
            self.signal_bot_symbol: Optional[str] = None
            self.signal_bot_thread: Optional[threading.Thread] = None
            self.signal_bot_stop_event: threading.Event = threading.Event()

            # State for the master controller
            self.master_bot_mode: str = "stopped"
            self.master_bot_target_mode: str = "demo"
            self.master_bot_stop_event: threading.Event = threading.Event()
            self.market_state: Optional[str] = None
            self.adx_value: Optional[float] = None

            # State for strategy activation
            self.active_strategies: Dict[str, bool] = self._load_strategies()

            # State for position tracking (Priority 1 Fix)
            self.is_in_position: bool = False
            self.active_position_symbol: Optional[str] = None

            self._initialized = True

    def _load_strategies(self) -> Dict[str, bool]:
        """
        Loads strategy names from the params file and initializes them as inactive.
        """
        try:
            with open(STRATEGY_PARAMS_FILE, 'r') as f:
                params = json.load(f)
                return {name: False for name in params.keys()}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load '{STRATEGY_PARAMS_FILE}': {e}.")
            return {}

    # --- Thread-Safe Getters and Setters ---

    def set_signal_bot_state(self, mode: str, engine: Optional['TradingEngine'], strategy: Optional['BaseStrategy'], name: Optional[str], symbol: Optional[str], thread: Optional[threading.Thread]):
        with self._lock:
            self.signal_bot_mode = mode
            self.signal_bot_engine = engine
            self.signal_bot_strategy = strategy
            self.signal_bot_strategy_name = name
            self.signal_bot_symbol = symbol
            self.signal_bot_thread = thread

    def get_signal_bot_thread(self) -> Optional[threading.Thread]:
        with self._lock:
            return self.signal_bot_thread

    def get_signal_bot_stop_event(self) -> threading.Event:
        # Event objects are thread-safe themselves, no lock needed to return
        return self.signal_bot_stop_event

    def set_master_bot_mode(self, mode: str):
        with self._lock:
            self.master_bot_mode = mode

    def get_master_bot_mode(self) -> str:
        with self._lock:
            return self.master_bot_mode

    def get_master_bot_stop_event(self) -> threading.Event:
        return self.master_bot_stop_event

    def set_market_state(self, state: str, adx: float):
        with self._lock:
            self.market_state = state
            self.adx_value = adx

    def get_market_state(self) -> (Optional[str], Optional[float]):
        with self._lock:
            return self.market_state, self.adx_value

    def set_active_strategies(self, active_dict: Dict[str, bool]):
        with self._lock:
            self.active_strategies = active_dict

    def get_active_strategies(self) -> Dict[str, bool]:
        with self._lock:
            return self.active_strategies.copy()

    def set_position_state(self, in_position: bool, symbol: Optional[str]):
        """Sets the state of the current trading position."""
        with self._lock:
            self.is_in_position = in_position
            self.active_position_symbol = symbol if in_position else None

    def get_position_state(self) -> (bool, Optional[str]):
        """Gets the state of the current trading position."""
        with self._lock:
            return self.is_in_position, self.active_position_symbol


# Global instance of the bot state
bot_state = BotState()
