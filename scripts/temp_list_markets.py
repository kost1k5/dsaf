import ccxt

def get_ccxt_okx_spot_symbols():
    """
    Fetches the list of all available SPOT instruments from OKX using the ccxt library.
    """
    try:
        exchange = ccxt.okx()
        markets = exchange.load_markets()

        # We are interested in spot markets that are active
        spot_symbols = [
            symbol for symbol, market in markets.items()
            if market.get('spot') and market.get('active')
        ]
        return spot_symbols

    except Exception as e:
        print(f"An error occurred while fetching data from ccxt: {e}")
        return []

if __name__ == "__main__":
    print("Fetching available SPOT symbols from OKX using ccxt...")
    live_symbols = get_ccxt_okx_spot_symbols()

    if live_symbols:
        print(f"Found {len(live_symbols)} live SPOT symbols.")

        # Check for the specific symbols from the config
        config_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "DOGE/USDT"]
        print("\nChecking for symbols from config (using '/' separator as ccxt does):")

        found_symbols = []
        for symbol in config_symbols:
            if symbol in live_symbols:
                print(f" - Found '{symbol}'")
                found_symbols.append(symbol)
            else:
                print(f" - Did NOT find '{symbol}'")

        if found_symbols:
            print("\nTo fix the config, use one of the found symbols.")
            # CCXT uses '/', but the app was changed to use '-'. Let's provide both.
            hyphenated_symbols = [s.replace('/', '-') for s in found_symbols]
            print(f"Suggestion for 'SYMBOLS_RAW' in config.py (using hyphens):")
            print(f'SYMBOLS_RAW: str = "{",".join(hyphenated_symbols)}"')
        else:
            print("\nCould not find any of the desired symbols. Here are some examples of what was found:")
            for symbol in live_symbols[:15]:
                print(symbol)

    else:
        print("Could not retrieve a list of symbols using ccxt.")
