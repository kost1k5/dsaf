import ccxt
import sys
import os

# Add the project root to the python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_okx_symbols():
    """
    Connects to OKX via ccxt and prints symbols related to BTC and USDT
    to help find the correct format.
    """
    try:
        # We need to add the parent directory to the path to find the `src` module
        # This is a common pattern for scripts inside a package

        exchange = ccxt.okx()
        markets = exchange.load_markets()

        print("--- Finding OKX Symbol Formats ---")

        # OKX uses perpetual swaps like BTC-USDT-SWAP or spot markets like BTC/USDT
        # Let's find the spot market symbol
        spot_symbols = [s for s in markets if s.endswith('/USDT') and 'BTC' in s]

        if spot_symbols:
            print("\nFound potential SPOT matches for BTC/USDT:")
            for symbol in spot_symbols:
                print(symbol)
        else:
            print("\nNo SPOT matches found for BTC/USDT.")

        swap_symbols = [s for s in markets if 'BTC' in s and 'USDT-SWAP' in s]

        if swap_symbols:
            print("\nFound potential SWAP matches for BTC-USDT-SWAP:")
            for symbol in swap_symbols:
                print(symbol)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_okx_symbols()
