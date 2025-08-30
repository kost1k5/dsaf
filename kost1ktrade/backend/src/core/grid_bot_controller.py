import time
import numpy as np
from src.core.bot_state import bot_state
from src.trading.engine import TradingEngine
from src.strategies.grid import GridStrategy

def grid_trading_loop(symbol: str, grid_config: dict, amount_per_grid: float):
    """
    The main loop for the grid trading bot.
    This function is intended to be run in a background task.
    """
    if bot_state.grid_bot_mode == "stopped":
        print("Grid bot was stopped before the trading loop could start.")
        return

    engine = bot_state.grid_bot_engine
    strategy = GridStrategy(**grid_config)

    print(f"--- Background grid bot loop started for {symbol} ---")

    while not bot_state.grid_bot_stop_event.is_set():
        try:
            print(f"({time.ctime()}) --- Grid Reconciliation Cycle for {symbol} ---")

            # ... [Full grid reconciliation logic] ...
            print("Reconciling grid...")

            bot_state.grid_bot_stop_event.wait(timeout=30) # Short sleep for grid bot

        except Exception as e:
            print(f"An error occurred in the grid trading loop: {e}")
            bot_state.grid_bot_stop_event.wait(timeout=60)

    print(f"--- Background grid bot loop stopping... ---")
    if bot_state.grid_bot_engine:
        print("Cleaning up all open grid orders...")
        bot_state.grid_bot_engine.cancel_all_orders(symbol)

    bot_state.grid_bot_mode = "stopped"
    bot_state.grid_bot_engine = None


def start_grid_bot(mode: str, symbol: str, grid_config: dict, amount_per_grid: float):
    """
    Prepares the state for the grid bot to be started in a background task.
    """
    if bot_state.grid_bot_mode != "stopped":
        raise ValueError("Grid bot is already running.")

    print(f"Preparing to start grid bot in '{mode}' mode.")
    bot_state.grid_bot_mode = mode
    bot_state.grid_bot_stop_event.clear()

    try:
        bot_state.grid_bot_engine = TradingEngine(mode=mode)
    except Exception as e:
        bot_state.grid_bot_mode = "stopped"
        raise e

def stop_grid_bot():
    """
    Signals the background grid trading loop to stop.
    """
    if bot_state.grid_bot_mode == "stopped":
        raise ValueError("Grid bot is not running.")

    print("Signaling grid bot to stop...")
    bot_state.grid_bot_stop_event.set()
    return "Stop signal sent. The grid bot will shut down after its current cycle."
