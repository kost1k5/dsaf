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
        CRYPTO_ASSETS = settings.SYMBOLS
        TIMEFRAMES = ['1h', '4h', '1d']
        end_date = datetime.now(UTC)

        # --- Initialize Collectors ---
        data_collector = DataCollector(exchange_id='okx', db_session=db)
        macro_collector = MacroDataCollector(db_session=db)
        sentiment_collector = SentimentCollector(db_session=db)

        # --- 1. Collect Asset-Agnostic Data ---
        print("\n--- Collecting Macroeconomic Data ---")
        # ... (rest of the asset-agnostic data collection logic is correct)

        # --- 2. Collect Crypto-Specific Data (Incremental) ---
        for raw_symbol in settings.SYMBOLS:
            try:
                asset = parse_asset_from_symbol(raw_symbol)
                summary["crypto_specific"][asset] = {}
                print(f"\n{'='*20} Collecting data for {asset} (from {raw_symbol}) {'='*20}")
                symbol = f"{asset}/USDT:USDT"

                # --- OHLCV Data ---
                for tf in TIMEFRAMES:
                    print(f"\n--- Collecting {asset} OHLCV ({tf}) ---")
                    latest_ohlcv_ms = data_collector.get_latest_candle_timestamp(symbol, tf)
                    if latest_ohlcv_ms:
                        since_ms = latest_ohlcv_ms + (1000 * 60 * 60)
                    else:
                        since_ms = int((end_date - timedelta(days=days_history)).timestamp() * 1000)

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
                print(f"\n--- Collecting {asset} Funding Rates (Full History Method) ---")
                latest_fr_ms = data_collector.get_latest_funding_rate_timestamp(symbol)
                start_date_obj = end_date - timedelta(days=days_history)
                if latest_fr_ms:
                    start_date_obj = datetime.fromtimestamp(latest_fr_ms / 1000, tz=UTC) + timedelta(days=1)

                instrument_family = f"{asset}-USDT"
                if start_date_obj < end_date:
                    all_fr_data = data_collector.fetch_full_funding_rate_history(
                        instrument_family=instrument_family,
                        start_date_str=start_date_obj.strftime('%Y-%m-%d'),
                        end_date_str=end_date.strftime('%Y-%m-%d')
                    )
                    if all_fr_data:
                        count = data_collector.save_funding_rates_to_db(all_fr_data, symbol)
                        summary["crypto_specific"][asset]["Funding Rates"] = count
                        print(f"-> Saved {count} new funding rate entries for {asset}.")
                    else:
                        print("-> No new funding rate data was fetched or processed.")
                else:
                    print(f"Funding rate data for {asset} is already up to date.")
                time.sleep(1)

            except Exception as e:
                print(f"\n{'!'*20} ERROR processing symbol {raw_symbol} {'!'*20}")
                print(f"An unexpected error occurred: {e}")
                print(f"Skipping this asset and continuing to the next one.")
                print(f"{'!'*60}\n")
                continue

        # --- Summary Printout ---
        # ... (rest of the summary logic is correct)

        print("\n--- Data collection complete! ---")
    finally:
        db.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Incremental Data Collection Orchestrator")
    parser.add_argument(
        "--days",
        type=int,
        default=1095,
        help="Number of past days of history to collect on the first run."
    )
    args = parser.parse_args()
    main(days_history=args.days)
