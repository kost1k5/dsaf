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

        # --- Open Interest Data (with custom forward pagination) ---
        # NOTE: The generic `fetch_paginated_history_backwards` function from the collector
        # was found to be incompatible with Bybit's API for open interest, leading to empty data.
        # This custom forward-pagination loop is a more robust replacement. It iterates
        # from the start date to the end date, fetching data in chunks.
        print(f"\n--- Collecting {asset} Open Interest (1h) ---")
        all_oi_data = []
        current_since = since_ms
        oi_timeframe = '1h'
        # Calculate timeframe duration in milliseconds for advancing the timestamp
        timeframe_duration_ms = 1 * 60 * 60 * 1000 # 1h in ms

        while current_since < end_ms:
            try:
                print(f"  Fetching chunk since {datetime.fromtimestamp(current_since/1000)}...")
                oi_chunk = data_collector.exchange.fetch_open_interest_history(
                    symbol=symbol,
                    timeframe=oi_timeframe,
                    since=current_since,
                    limit=500 # Bybit allows up to 500
                )

                if not oi_chunk:
                    print("  No more Open Interest data returned, stopping.")
                    break

                # Filter out duplicates that might be returned by the API
                last_timestamp = all_oi_data[-1]['timestamp'] if all_oi_data else 0
                new_data = [d for d in oi_chunk if d['timestamp'] > last_timestamp]

                if not new_data:
                    print("  No new data in this chunk, advancing time to prevent loop.")
                    current_since += timeframe_duration_ms * 500 # Advance by the limit
                    continue

                all_oi_data.extend(new_data)

                # Update the 'since' parameter for the next iteration
                last_ts_in_chunk = new_data[-1]['timestamp']
                current_since = last_ts_in_chunk + timeframe_duration_ms # Start next chunk after the last one

                print(f"  Fetched {len(new_data)} new OI points. Total: {len(all_oi_data)}. Last timestamp: {datetime.fromtimestamp(last_ts_in_chunk/1000)}")

            except Exception as e:
                print(f"  An error occurred while fetching Open Interest chunk for {asset}: {e}")
                print("  Stopping OI collection for this asset due to error.")
                break

            time.sleep(data_collector.exchange.rateLimit / 1000) # Respect rate limits

        if all_oi_data:
            oi_df = pd.DataFrame(all_oi_data)
            # Filter one last time to ensure we are within the date range
            oi_df = oi_df[(oi_df['timestamp'] >= since_ms) & (oi_df['timestamp'] <= end_ms)]
            if 'timestamp' in oi_df.columns:
                oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms')

            # The 'info' dict from ccxt can be very large, let's just keep the useful columns
            # Standardized ccxt response for OI includes: 'symbol', 'timestamp', 'datetime', 'openInterestValue'
            # We select columns that are likely to exist and be useful
            cols_to_keep = ['symbol', 'timestamp', 'openInterestValue']
            oi_df_filtered = oi_df[[col for col in cols_to_keep if col in oi_df.columns]]

            oi_df_filtered.to_csv(os.path.join(OUTPUT_DIR, f'{asset}_open_interest_1h.csv'), index=False)
            print(f"Saved {asset}_open_interest_1h.csv with {len(oi_df_filtered)} records to {OUTPUT_DIR}")
        else:
            print(f"No Open Interest data was collected for {asset}.")
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