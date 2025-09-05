import os
import pandas as pd
from datetime import datetime, timedelta
import time
import argparse
import sys
import csv

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
    # Using OKX as the primary exchange as requested by the user.
    print("\nInitializing OKX Data Collector...")
    data_collector = DataCollector(exchange_id='okx')
    
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
        # NOTE: Historical Open Interest data is not available via the OKX API for the required range.
        # This feature has been removed as per user's final request.
        print(f"\n--- Skipping Open Interest for {asset} (Not Available) ---")


        # --- Funding Rate Data (with custom forward pagination) ---
        print(f"\n--- Collecting {asset} Funding Rates ---")
        all_fr_data = []
        current_since = since_ms
        timeframe_duration_ms = 1 * 60 * 60 * 1000 # Assume 1h for safety, though FR is often 8h

        while current_since < end_ms:
            try:
                print(f"  Fetching funding rate chunk since {datetime.fromtimestamp(current_since/1000)}...")
                fr_chunk = data_collector.exchange.fetch_funding_rate_history(
                    symbol=symbol,
                    since=current_since,
                    limit=100 # OKX limit is 100
                )

                if not fr_chunk:
                    print("  No more Funding Rate data returned, stopping pagination.")
                    break

                # Sort and filter out duplicates
                fr_chunk_sorted = sorted(fr_chunk, key=lambda x: x['timestamp'])
                last_timestamp_in_all_data = all_fr_data[-1]['timestamp'] if all_fr_data else 0
                new_data = [d for d in fr_chunk_sorted if d['timestamp'] > last_timestamp_in_all_data]

                if not new_data:
                    print("  No new data in this chunk (all records were duplicates). Advancing time.")
                    current_since = fr_chunk_sorted[-1]['timestamp'] + timeframe_duration_ms
                    continue

                all_fr_data.extend(new_data)

                last_ts_in_chunk = new_data[-1]['timestamp']
                current_since = last_ts_in_chunk + 1 # Increment by 1ms to avoid fetching the same record

                first_ts_str = datetime.fromtimestamp(new_data[0]['timestamp']/1000)
                last_ts_str = datetime.fromtimestamp(new_data[-1]['timestamp']/1000)
                print(f"  Fetched {len(new_data)} new FR points. Total: {len(all_fr_data)}. Chunk range: {first_ts_str} to {last_ts_str}")

            except Exception as e:
                print(f"  An error occurred while fetching Funding Rate chunk for {asset}: {e}")
                break

            time.sleep(data_collector.exchange.rateLimit / 1000)

        if all_fr_data:
            fr_df = pd.DataFrame(all_fr_data)
            fr_df['timestamp'] = pd.to_datetime(fr_df['timestamp'], unit='ms')
            fr_df.to_csv(os.path.join(OUTPUT_DIR, f'{asset}_funding_rates.csv'), index=False)
            print(f"Saved {asset}_funding_rates.csv with {len(fr_df)} records to {OUTPUT_DIR}")
        else:
            print(f"No Funding Rate data was collected for {asset}.")
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