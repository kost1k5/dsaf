import os
import pandas as pd
from datetime import datetime, timedelta
import time
import argparse
import sys
from sqlalchemy.orm import Session

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.session import SessionLocal
from src.data_collector.collector import DataCollector
from src.data_collector.macro_collector import MacroDataCollector
from src.data_collector.sentiment_collector import SentimentCollector

def main(days_history: int):
    """
    Main orchestration script to collect all required data sources,
    performing incremental updates based on data already in the database.
    """
    db: Session = SessionLocal()
    summary = {
        "asset_agnostic": {},
        "crypto_specific": {}
    }
    try:
        # --- Configuration ---
        CRYPTO_ASSETS = ['BTC', 'ETH', 'SOL', 'LINK']
        TIMEFRAMES = ['1h', '4h', '1d']
        end_date = datetime.utcnow()

        # --- Initialize Collectors with DB Session ---
        data_collector = DataCollector(exchange_id='okx', db_session=db)
        macro_collector = MacroDataCollector(db_session=db)
        sentiment_collector = SentimentCollector(db_session=db)

        # --- 1. Collect Asset-Agnostic Data (Incremental) ---
        print("\n--- Collecting Macroeconomic Data ---")
        latest_macro_ts = macro_collector.get_latest_macro_timestamp()
        if latest_macro_ts:
            start_date_macro = latest_macro_ts + timedelta(days=1)
            print(f"Found existing macro data up to {latest_macro_ts.strftime('%Y-%m-%d')}.")
        else:
            start_date_macro = end_date - timedelta(days=days_history)
            print(f"No existing macro data found. Starting full history download from {start_date_macro.strftime('%Y-%m-%d')}.")

        if start_date_macro < end_date:
            print(f"Fetching macro data from {start_date_macro.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
            macro_df = macro_collector.fetch_data(start_date=start_date_macro.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            if not macro_df.empty:
                count = macro_collector.save_macro_data_to_db(macro_df)
                summary["asset_agnostic"]["Macro Data"] = count
                print(f"-> Saved {count} new macro data entries.")
            else:
                print("-> No new data returned from the source.")
        else:
            print("Macro data is already up to date.")

        print("\n--- Collecting Fear & Greed Index ---")
        fng_df = sentiment_collector.fetch_fear_greed_data(limit=0) # Fetch all available
        if not fng_df.empty:
            count = sentiment_collector.save_fng_data_to_db(fng_df)
            summary["asset_agnostic"]["Fear & Greed"] = count
            print(f"-> Saved {count} new F&G entries (database handles conflicts).")

        print("\n--- Collecting RSS News ---")
        news_items = sentiment_collector.fetch_rss_news()
        if news_items:
            count = sentiment_collector.save_news_to_db(news_items)
            summary["asset_agnostic"]["News Headlines"] = count
            print(f"-> Saved {count} new news items (database handles conflicts).")

        # --- 2. Collect Crypto-Specific Data (Incremental) ---
        for asset in CRYPTO_ASSETS:
            summary["crypto_specific"][asset] = {}
            print(f"\n{'='*20} Collecting data for {asset} {'='*20}")
            symbol = f"{asset}/USDT:USDT"

            # --- OHLCV Data ---
            for tf in TIMEFRAMES:
                print(f"\n--- Collecting {asset} OHLCV ({tf}) ---")
                latest_ohlcv_ms = data_collector.get_latest_candle_timestamp(symbol, tf)
                if latest_ohlcv_ms:
                    since_ms = latest_ohlcv_ms + (1000 * 60 * 60) # Start from the next hour
                    print(f"Found existing OHLCV data up to {datetime.fromtimestamp(latest_ohlcv_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}.")
                else:
                    since_ms = int((end_date - timedelta(days=days_history)).timestamp() * 1000)
                    print(f"No existing OHLCV data found. Starting full history download from {datetime.fromtimestamp(since_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}.")

                is_historical_fill = not latest_ohlcv_ms

                if since_ms < int(end_date.timestamp() * 1000):
                    print(f"Fetching OHLCV data from {datetime.fromtimestamp(since_ms/1000).strftime('%Y-%m-%d %H:%M:%S')} to now...")
                    ohlcv_data = data_collector.fetch_candles_in_range(symbol, tf, since_ms, int(end_date.timestamp() * 1000))

                    # (Fix) If a full history download fails, try a shorter period.
                    if not ohlcv_data and is_historical_fill and days_history > 365:
                        print(f"Warning: Full history download for {asset} ({tf}) with {days_history} days returned no data. Retrying with 365 days.")
                        new_since_ms = int((end_date - timedelta(days=365)).timestamp() * 1000)
                        ohlcv_data = data_collector.fetch_candles_in_range(symbol, tf, new_since_ms, int(end_date.timestamp() * 1000))

                    if ohlcv_data:
                        count = data_collector.save_candles_to_db(ohlcv_data, symbol, tf)
                        summary["crypto_specific"][asset][f"OHLCV ({tf})"] = count
                        print(f"-> Saved {count} new {tf} candles for {asset}.")
                    else:
                        # Add a more prominent warning if the historical fill fails completely.
                        if is_historical_fill:
                            print(f"CRITICAL WARNING: Full history download for {asset} ({tf}) returned no data, even after retries. Downstream processes will likely fail.")
                        else:
                            print("-> No new data returned from the exchange.")
                else:
                    print(f"OHLCV data for {asset} ({tf}) is already up to date.")
                time.sleep(1)

            # --- Funding Rate Data ---
            print(f"\n--- Collecting {asset} Funding Rates ---")
            latest_fr_ms = data_collector.get_latest_funding_rate_timestamp(symbol)
            if latest_fr_ms:
                since_fr_ms = latest_fr_ms + 1
                print(f"Found existing funding rate data up to {datetime.fromtimestamp(latest_fr_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}.")
            else:
                since_fr_ms = int((end_date - timedelta(days=days_history)).timestamp() * 1000)
                print(f"No existing funding rate data found. Starting full history download from {datetime.fromtimestamp(since_fr_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}.")

            if since_fr_ms < int(end_date.timestamp() * 1000):
                print(f"Fetching funding rate data from {datetime.fromtimestamp(since_fr_ms/1000).strftime('%Y-%m-%d %H:%M:%S')} to now...")
                all_fr_data = []
                current_since = since_fr_ms
                while current_since < int(end_date.timestamp() * 1000):
                    fr_chunk = data_collector.fetch_funding_rate_history(symbol=symbol, since=current_since, limit=100)
                    if not fr_chunk:
                        break # No more data from exchange

                    last_ts_in_all_data = all_fr_data[-1]['timestamp'] if all_fr_data else 0
                    new_data = [d for d in fr_chunk if d['timestamp'] > last_ts_in_all_data]
                    if not new_data:
                        break

                    all_fr_data.extend(new_data)
                    current_since = new_data[-1]['timestamp'] + 1
                    time.sleep(data_collector.exchange.rateLimit / 1000)

                if all_fr_data:
                    count = data_collector.save_funding_rates_to_db(all_fr_data, symbol)
                    summary["crypto_specific"][asset]["Funding Rates"] = count
                    print(f"-> Saved {count} new funding rate entries for {asset}.")
                else:
                    print("-> No new data returned from the exchange.")
            else:
                print(f"Funding rate data for {asset} is already up to date.")
            time.sleep(1)

        print("\n" + "="*50)
        print("=== DATA COLLECTION SUMMARY" + " "*25 + "===")
        print("="*50)
        for data_type, count in summary.get("asset_agnostic", {}).items():
            print(f"- {data_type:<20}: {count} records")
        for asset, details in summary.get("crypto_specific", {}).items():
            print(f"\n--- {asset} ---")
            for data_type, count in details.items():
                print(f"  - {data_type:<18}: {count} records")
        print("="*50 + "\n")

        print("\n--- Data collection complete! ---")
    finally:
        db.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Incremental Data Collection Orchestrator")
    parser.add_argument(
        "--days",
        type=int,
        default=1095, # 3 years, for the initial full load
        help="Number of past days of history to collect on the first run."
    )
    args = parser.parse_args()
    main(days_history=args.days)
