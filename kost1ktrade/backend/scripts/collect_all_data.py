import os
import pandas as pd
from datetime import datetime, timedelta, UTC
import time
import argparse
import sys
from sqlalchemy.orm import Session
from sqlalchemy import func

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.session import SessionLocal
from src.database.models import Candle
from src.data_collector.collector import DataCollector
from src.data_collector.macro_collector import MacroDataCollector
from src.data_collector.sentiment_collector import SentimentCollector
from src.data_collector.calendar_data import fetch_and_store_economic_calendar
from src.core.config import settings
from src.core.utils import parse_asset_from_symbol
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
        # Use settings from the central config file
        CRYPTO_ASSETS = settings.SYMBOLS
        # Collect data for all relevant timeframes for multi-timeframe analysis
        TIMEFRAMES = ['1h', '4h', '1d']
        end_date = datetime.now(UTC)

        # --- Initialize Collectors with DB Session ---
        data_collector = DataCollector(exchange_id='okx', db_session=db)
        macro_collector = MacroDataCollector(db_session=db)
        sentiment_collector = SentimentCollector(db_session=db)

        # --- 1. Collect Asset-Agnostic Data (Incremental) ---
        print("\n--- Collecting Macroeconomic Data ---")
        latest_macro_ts = macro_collector.get_latest_macro_timestamp()
        if latest_macro_ts:
            # Assume the timestamp from the DB is UTC, make it aware before comparison
            latest_macro_ts = latest_macro_ts.replace(tzinfo=UTC)
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

        print("\n--- Collecting Economic Calendar Events ---")
        # The new function uses the existing authenticated data_collector instance
        # to handle API keys and signing automatically.
        fetch_and_store_economic_calendar(data_collector=data_collector)


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
        for raw_symbol in settings.SYMBOLS:
            asset = parse_asset_from_symbol(raw_symbol)
            summary["crypto_specific"][asset] = {}
            print(f"\n{'='*20} Collecting data for {asset} (from {raw_symbol}) {'='*20}")
            # Construct the ccxt-compatible symbol for the swap market
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

                if since_ms < int(end_date.timestamp() * 1000):
                    print(f"Fetching OHLCV data from {datetime.fromtimestamp(since_ms/1000).strftime('%Y-%m-%d %H:%M:%S')} to now...")
                    ohlcv_data = data_collector.fetch_candles_in_range(symbol, tf, since_ms, int(end_date.timestamp() * 1000))
                    if ohlcv_data:
                        count = data_collector.save_candles_to_db(ohlcv_data, symbol, tf)
                        summary["crypto_specific"][asset][f"OHLCV ({tf})"] = count
                        print(f"-> Saved {count} new {tf} candles for {asset}.")
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

        print("\n" + "="*60)
        print("=== FINAL DATA COLLECTION SUMMARY" + " "*29 + "===")
        print("="*60)
        print("\n--- Asset-Agnostic Data ---")
        for data_type, count in summary.get("asset_agnostic", {}).items():
            print(f"  - {data_type:<20}: Fetched {count} records")

        print("\n--- Crypto-Specific Data ---")
        for asset, details in summary.get("crypto_specific", {}).items():
            print(f"\n* Asset: {asset}")
            # Query the DB for actual date ranges and counts
            start_date = db.query(func.min(Candle.open_time)).filter(Candle.symbol.like(f'{asset}%')).scalar()
            end_date = db.query(func.max(Candle.open_time)).filter(Candle.symbol.like(f'{asset}%')).scalar()

            if start_date and end_date:
                print(f"  - Date Range in DB  : {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            for data_type, count in details.items():
                print(f"  - {data_type:<20}: Fetched {count} new records")
        print("="*60 + "\n")

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
