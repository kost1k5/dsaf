import time
import pandas as pd
from src.data_collector.collector import DataCollector
from src.trading.engine import TradingEngine
from src.strategies.sma_crossover import SmaCrossoverStrategy

def main_loop():
    """The main loop for the trading bot."""
    print("--- Kost1kTrade Bot Starting ---")

    # Initialize components
    try:
        # It's better to use the same exchange for data and trading
        exchange_id = 'okx'
        collector = DataCollector(exchange_id=exchange_id)
        # The engine will be initialized later, based on the mode
        # engine = TradingEngine(mode='demo', exchange_id=exchange_id)
        print("Data collector initialized.")
    except (ValueError, ConnectionError) as e:
        print(f"FATAL: Failed to initialize components: {e}")
        return

    strategy = SmaCrossoverStrategy(short_window=40, long_window=100)
    symbol = 'BTC/USDT'
    base_currency = 'BTC'
    quote_currency = 'USDT'
    timeframe = '1h' # The bot will run once per hour
    sleep_duration_seconds = 3600 # 1 hour

    while True:
        try:
            print(f"\n--- New Cycle: {time.ctime()} ---")

            # 1. Fetch latest data
            print(f"Fetching latest {strategy.long_window} candles for {symbol}...")
            candles_list = collector.fetch_candles(symbol, timeframe, limit=strategy.long_window)

            if not candles_list or len(candles_list) < strategy.long_window:
                raise ValueError(f"Could not fetch enough candle data ({len(candles_list)}/{strategy.long_window})")

            candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])

            # 2. Generate signal
            print("Analyzing data and generating signal...")
            result_df = strategy.generate_signals(candles_df)
            latest_signal = result_df.iloc[-1]['signal']
            print(f"Strategy: {strategy.__class__.__name__} | Signal: {latest_signal}")

            # 3. Execute trade
            # NOTE: This is a simplified logic. A real bot would need more sophisticated
            # position and risk management.
            if latest_signal == 'BUY':
                balance = engine.get_balance(quote_currency)
                if balance and balance['free'] > 10: # Min order size check
                    print(f"BUY signal detected. Checking {quote_currency} balance...")
                    amount_to_buy = balance['free'] / candles_df.iloc[-1]['close']
                    print(f"Attempting to place MARKET BUY order for ~{amount_to_buy:.6f} {base_currency}.")
                    engine.create_order(symbol, 'market', 'buy', amount_to_buy)
                else:
                    print(f"BUY signal detected, but not enough {quote_currency} to trade or balance check failed.")

            elif latest_signal == 'SELL':
                balance = engine.get_balance(base_currency)
                if balance and balance['free'] > 0.0001: # Min order size check
                    print(f"SELL signal detected. Checking {base_currency} balance...")
                    print(f"Attempting to place MARKET SELL order for {balance['free']:.6f} {base_currency}.")
                    engine.create_order(symbol, 'market', 'sell', balance['free'])
                else:
                    print(f"SELL signal detected, but no {base_currency} position to sell or balance check failed.")

            # Wait for the next cycle
            print(f"Cycle complete. Sleeping for {sleep_duration_seconds / 60:.0f} minutes...")
            time.sleep(sleep_duration_seconds)

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            print("Restarting loop after 1 minute...")
            time.sleep(60)

if __name__ == "__main__":
    main_loop()
