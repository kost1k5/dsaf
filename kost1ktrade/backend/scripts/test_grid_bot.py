import time
import sys
import os
import threading

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.grid_bot_controller import start_grid_bot, stop_grid_bot, stop_all_grid_bots
from src.core.bot_state import bot_state

def run_test():
    print("--- Multi-Grid Bot Controller Test ---")

    # Configurations for two different bots
    configs = {
        "BTC/USDT": {
            "symbol": "BTC/USDT",
            "grid_range_low": 60000.0,
            "grid_range_high": 70000.0,
            "num_grids": 10,
            "amount_per_grid": 10, # in quote currency (USDT)
        },
        "SOL/USDT": {
            "symbol": "SOL/USDT",
            "grid_range_low": 150.0,
            "grid_range_high": 200.0,
            "num_grids": 10,
            "amount_per_grid": 5, # in quote currency (USDT)
        }
    }

    try:
        # --- Start Bots ---
        print("\n--- Attempting to start multiple grid bots in 'demo' mode ---")
        for symbol, config in configs.items():
            try:
                print(f"Starting bot for {symbol}...")
                start_grid_bot(symbol=symbol, mode='demo', config=config)
            except ValueError as e:
                print(f"Caught expected error on startup for {symbol}: {e}")

        # Verify they are running
        print(f"\nCurrent running bots: {list(bot_state.grid_bot_states.keys())}")
        assert len(bot_state.grid_bot_threads) == 2, "Expected 2 bots to be running."
        print("All bots started successfully.")

        # Let the bots run for a few cycles
        print("\nBots running. Waiting for 15 seconds...")
        time.sleep(15)

    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- Stop Bots ---
        print("\n--- Attempting to stop all grid bots ---")
        try:
            stop_all_grid_bots()
            # Wait a moment for threads to clean up
            time.sleep(5)
            print(f"Bots stopped. Current running bots: {list(bot_state.grid_bot_states.keys())}")
            assert len(bot_state.grid_bot_threads) == 0, "Expected all bots to be stopped."
            print("All bots stopped successfully.")
        except Exception as e:
            print(f"An error occurred during stop_all_grid_bots: {e}")


    print("\n--- Test Complete ---")

if __name__ == '__main__':
    # This script is a direct test of the controller logic, not the API.
    # It requires a running environment where ccxt can initialize the trading engine.
    # Ensure you have network connectivity.
    run_test()
