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
    A singleton-like class to hold the global state of the bot.
    It now dynamically loads available strategies from a JSON file.
    """
    def _load_strategies(self) -> Dict[str, bool]:
        """
        Loads strategy names from the params file and initializes them as inactive.
        """
        try:
            # Note: The strategy_params.json is in the `backend` directory,
            # one level up from `src/core`.
            with open(STRATEGY_PARAMS_FILE, 'r') as f:
                params = json.load(f)
                # Initialize all found strategies to False (inactive)
                return {name: False for name in params.keys()}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load '{STRATEGY_PARAMS_FILE}': {e}. No strategies will be available for activation.")
            return {}

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
        self.grid_bots_mode: str = "stopped"  # A single mode ('real' or 'demo') for all grid bots
        self.grid_bots_engine: Optional['TradingEngine'] = None
        self.grid_bot_threads: Dict[str, threading.Thread] = {}
        self.grid_bot_stop_events: Dict[str, threading.Event] = {}
        # Holds the configuration for each active grid bot, keyed by symbol
        self.grid_bot_configs: Dict[str, Dict[str, Any]] = {}

        # State for the master controller
        self.master_bot_mode: str = "stopped"
        self.master_bot_target_mode: str = "demo" # 'demo' or 'real'
        self.master_bot_stop_event: threading.Event = threading.Event()
        self.market_state: Optional[str] = None
        self.adx_value: Optional[float] = None

        # State for strategy activation
        # Dynamically load strategies from the JSON file.
        self.active_strategies: Dict[str, bool] = self._load_strategies()

# Global instance of the bot state
bot_state = BotState()
