from typing import Optional
from src.trading.engine import TradingEngine

class BotState:
    """
    A simple singleton-like class to hold the global state of the bot.
    """
    def __init__(self):
        self.mode: str = "stopped"  # Can be 'stopped', 'real', or 'demo'
        self.trading_engine: Optional[TradingEngine] = None

# Global instance of the bot state
bot_state = BotState()
