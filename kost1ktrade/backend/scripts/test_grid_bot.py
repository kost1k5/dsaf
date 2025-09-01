import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.grid_bot_controller import start_grid_bot, stop_grid_bot

if __name__ == '__main__':
    print("--- Grid Bot Controller Test ---")

    grid_config = {
        "grid_range_low": 50000.0,
        "grid_range_high": 60000.0,
        "num_grids": 10,
    }

    try:
        print("Attempting to start grid bot in 'demo' mode...")
        start_grid_bot(
            mode='demo',
            symbol='BTC/USDT',
            grid_config=grid_config,
            amount_per_grid=0.001
        )

        # Let the bot run for a few cycles
        print("\nBot running. Waiting for 10 seconds before stopping...")
        time.sleep(10)

    except Exception as e:
        print(f"\nCaught expected error on startup: {e}")

    finally:
        print("\nAttempting to stop grid bot...")
        try:
            stop_grid_bot()
        except ValueError as e:
            print(f"Stop command failed as expected: {e}")

    print("\n--- Test Complete ---")
