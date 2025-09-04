import os
import pandas as pd
from datetime import datetime, timedelta
import time
import argparse
import sys

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure these import paths are correct for your project structure
try:
    # Assuming these are the correct import paths based on the previous script
    from src.data_collector.collector import DataCollector
    from src.data_collector.macro_collector import MacroDataCollector
    from src.data_collector.sentiment_collector import SentimentCollector
except ImportError as e:
    print(f"Error importing modules: {e}. Please check your 'src' structure and import paths.")
    sys.exit(1)

def main(days_history: int):
    """
    Main orchestration script to collect all required data sources.
    """
    # --- Configuration ---
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    CRYPTO_ASSETS = ['BTC', 'ETH', 'SOL', 'LINK']
    TIMEFRAMES = ['1h', '4h', '1d']

    # Define date range
    # Using UTC for consistency with exchange times
    end_date = datetime.utcnow() 
    start_date = end_date - timedelta(days=days_history)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Convert timestamps to milliseconds for exchange API calls
    since_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    print(f"Starting data collection...")
    print(f" - All data will be fetched for {days_history} days (from {start_date_str} to {end_date_str}).")

    # --- Initialize Collectors ---
    # CHANGE: Switched from OKX (previous default) to Bybit for better historical data depth
    print("\nInitializing Bybit Data Collector...")
    data_collector = DataCollector(exchange_id='bybit') 
    
    macro_collector = MacroDataCollector()
    sentiment_collector = SentimentCollector()

    # --- 1. Collect Asset-Agnostic Data ---
    # (Macro/Sentiment collection logic remains the same)
    print("\n--- Collecting Macroeconomic Data ---")
    macro_df = macro_collector.fetch_data(start_date=start_date_str, end_date=end_date_str)
    if not macro_df.empty:
        macro_df.to_csv(os.path.join(OUTPUT_DIR, 'macro_data.csv'))
        print(f"Saved macro_data.csv to {OUTPUT_DIR}")

    print("\n--- Collecting Fear & Greed Index ---")
    fng_df = sentiment_collector.fetch_fear_greed_data(limit=0)
    if not fng_df.empty:
        # Ensure index is datetime for filtering
        if not isinstance(fng_df.index, pd.DatetimeIndex):
             if 'timestamp' in fng_df.columns:
                 fng_df = fng_df.set_index(pd.to_datetime(fng_df['timestamp']))

        fng_df_filtered = fng_df.loc[start_date_str:end_date_str].copy()
        if not fng_df_filtered.empty:
            fng_df_filtered.to_csv(os.path.join(OUTPUT_DIR, 'fng_data.csv'))
            print(f"Saved fng_data.csv to {OUTPUT_DIR}")
        else:
            print("Warning: No F&G data found within the specified date range after fetching.")

    print("\n--- Collecting RSS News ---")
    news_items = sentiment_collector.fetch_rss_news()
    if news_items:
        news_df = pd.DataFrame(news_items)
        news_df.to_csv(os.path.join(OUTPUT_DIR, 'news_headlines.csv'), index=False)
        print(f"Saved news_headlines.csv to {OUTPUT_DIR}")

    # --- 2. Collect Crypto-Specific Data ---
    for asset in CRYPTO_ASSETS:
        print(f"\n{'='*20} Collecting data for {asset} {'='*20}")
        # Use standard symbol for Perpetual Swaps
        symbol = f"{asset}/USDT:USDT" 

        # --- OHLCV Data ---
        for tf in TIMEFRAMES:
            print(f"\n--- Collecting {asset} OHLCV ({tf}) ---")
            try:
                # CHANGE: Using the generic data_collector instance
                ohlcv_data = data_collector.fetch_candles_in_range(symbol, tf, since_ms, end_ms)
                if ohlcv_data:
                    ohlcv_df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
                    ohlcv_df.to_csv(os.path.join(OUTPUT_DIR, f'{asset}_ohlcv_{tf}.csv'), index=False)
                    print(f"Saved {asset}_ohlcv_{tf}.csv to {OUTPUT_DIR}")
            except Exception as e:
                print(f"Could not fetch OHLCV for {asset} ({tf}). Error: {e}")
            time.sleep(1) # Rate limiting precaution

        # --- Open Interest Data ---
        print(f"\n--- Collecting {asset} Open Interest (1h) ---")
        try:
            # CHANGE: Using the generic data_collector instance (since/end parameters are already correct)
            oi_data = data_collector.fetch_paginated_history_backwards(
                data_collector.fetch_open_interest_history,
                symbol=symbol,
                timeframe='1h',
                since=since_ms,
                end=end_ms
            )
            if oi_data:
                oi_df = pd.DataFrame(oi_data)
                # Ensure timestamp format is correct
                if 'timestamp' in oi_df.columns and pd.api.types.is_numeric_dtype(oi_df['timestamp']):
                    oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
                oi_df.to_csv(os.path.join(OUTPUT_DIR, f'{asset}_open_interest_1h.csv'), index=False)
                print(f"Saved {asset}_open_interest_1h.csv to {OUTPUT_DIR}")
        except Exception as e:
            print(f"Could not fetch Open Interest for {asset}. Error: {e}")
        time.sleep(1)

        # --- Funding Rate Data ---
        print(f"\n--- Collecting {asset} Funding Rates ---")
        try:
            # CHANGE: Using the generic data_collector instance
            fr_data = data_collector.fetch_paginated_history_backwards(
                data_collector.fetch_funding_rate_history,
                symbol=symbol,
                since=since_ms,
                end=end_ms
            )
            if fr_data:
                fr_df = pd.DataFrame(fr_data)
                if 'timestamp' in fr_df.columns and pd.api.types.is_numeric_dtype(fr_df['timestamp']):
                    fr_df['timestamp'] = pd.to_datetime(fr_df['timestamp'], unit='ms')
                fr_df.to_csv(os.path.join(OUTPUT_DIR, f'{asset}_funding_rates.csv'), index=False)
                print(f"Saved {asset}_funding_rates.csv to {OUTPUT_DIR}")
        except Exception as e:
            print(f"Could not fetch Funding Rates for {asset}. Error: {e}")
        time.sleep(1)

    print("\n--- Data collection complete! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Collection Orchestrator")
    parser.add_argument(
        "--days",
        type=int,
        default=1095, # 3 years
        help="Number of past days of history to collect."
    )
    args = parser.parse_args()

    main(days_history=args.days)