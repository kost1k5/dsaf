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

        # --- Open Interest Data (Append-Only Logic) ---
        print(f"\n--- Collecting {asset} Open Interest (1h) ---")
        oi_filepath = os.path.join(OUTPUT_DIR, f'{asset}_open_interest_1h.csv')
        try:
            # 1. Fetch the single most recent OI data point from the exchange.
            latest_point = data_collector.exchange.fetch_open_interest_history(symbol, '1h', limit=1)
            if not latest_point:
                print(f"  -> No OI data returned from exchange for {asset}.")
                continue

            # 2. Extract and format the new data.
            new_data_point = latest_point[0]
            new_timestamp = pd.to_datetime(new_data_point['timestamp'], unit='ms', utc=True)
            new_value = new_data_point.get('openInterestValue') or new_data_point.get('openInterest')

            if new_value is None:
                print(f"  -> OI value not found in response for {asset}.")
                continue

            # 3. Check for duplicates by reading the last line of the existing file.
            file_exists = os.path.exists(oi_filepath)
            if file_exists:
                with open(oi_filepath, 'r', encoding='utf-8') as f:
                    # Find the last line to get the last timestamp
                    try:
                        last_line = f.readlines()[-1]
                        last_timestamp_str = last_line.split(',')[0]
                        last_timestamp = pd.to_datetime(last_timestamp_str, utc=True)
                        if new_timestamp <= last_timestamp:
                            print(f"  -> Latest OI data for {asset} is already recorded. No changes made.")
                            continue
                    except (IndexError, pd.errors.ParserError):
                        # Handle case where file is empty or corrupt
                        file_exists = False # Treat as a new file

            # 4. If the data is new, append it to the CSV.
            print(f"  -> New OI data found for {asset} at {new_timestamp}. Appending to file.")
            with open(oi_filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header only if the file is brand new
                if not file_exists:
                    writer.writerow(['timestamp', 'openInterestValue'])

                # Write the new data row
                writer.writerow([new_timestamp.isoformat(), new_value])

        except Exception as e:
            print(f"  -> An error occurred while collecting latest Open Interest for {asset}: {e}")
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