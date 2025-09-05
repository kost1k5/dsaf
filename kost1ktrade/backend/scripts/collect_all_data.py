import os
import pandas as pd
from datetime import datetime, timedelta, UTC
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
    try:
        # --- Configuration ---
        CRYPTO_ASSETS = ['BTC', 'ETH', 'SOL', 'LINK']
        TIMEFRAMES = ['1h', '4h', '1d']
        end_date = datetime.now(UTC)

        # --- Initialize Collectors with DB Session ---
        data_collector = DataCollector(exchange_id='okx', db_session=db)
        macro_collector = MacroDataCollector(db_session=db)
        sentiment_collector = SentimentCollector(db_session=db)

        # --- 1. Collect Asset-Agnostic Data (Incremental) ---
        print("\n--- Collecting Macroeconomic Data ---")
        latest_macro_ts = macro_collector.get_latest_macro_timestamp()
        start_date_macro = (latest_macro_ts + timedelta(days=1)) if latest_macro_ts else (end_date - timedelta(days=days_history))
        if start_date_macro < end_date:
            macro_df = macro_collector.fetch_data(start_date=start_date_macro.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            if not macro_df.empty:
                macro_collector.save_macro_data_to_db(macro_df)
        else:
            print("Macro data is already up to date.")

        print("\n--- Collecting Fear & Greed Index ---")
        # F&G API is simple, fetching all and letting the DB handle conflicts is easiest.
        fng_df = sentiment_collector.fetch_fear_greed_data(limit=0)
        if not fng_df.empty:
            sentiment_collector.save_fng_data_to_db(fng_df)

        print("\n--- Collecting RSS News ---")
        # News is also simple to fetch all and let DB handle conflicts.
        news_items = sentiment_collector.fetch_rss_news()
        if news_items:
            sentiment_collector.save_news_to_db(news_items)

        # --- 2. Collect Crypto-Specific Data (Incremental) ---
        for asset in CRYPTO_ASSETS:
            print(f"\n{'='*20} Collecting data for {asset} {'='*20}")
            symbol = f"{asset}/USDT:USDT"

            # --- OHLCV Data ---
            for tf in TIMEFRAMES:
                print(f"\n--- Collecting {asset} OHLCV ({tf}) ---")
                latest_ohlcv_ms = data_collector.get_latest_candle_timestamp(symbol, tf)
                since_ms = latest_ohlcv_ms + (1000 * 60 * 60) if latest_ohlcv_ms else int((end_date - timedelta(days=days_history)).timestamp() * 1000)

                if since_ms < int(end_date.timestamp() * 1000):
                    ohlcv_data = data_collector.fetch_candles_in_range(symbol, tf, since_ms, int(end_date.timestamp() * 1000))
                    if ohlcv_data:
                        data_collector.save_candles_to_db(ohlcv_data, symbol, tf)
                else:
                    print(f"OHLCV data for {asset} ({tf}) is already up to date.")
                time.sleep(1)

            # --- Funding Rate Data ---
            print(f"\n--- Collecting {asset} Funding Rates ---")
            latest_fr_ms = data_collector.get_latest_funding_rate_timestamp(symbol)
            since_fr_ms = latest_fr_ms + 1 if latest_fr_ms else int((end_date - timedelta(days=days_history)).timestamp() * 1000)

            if since_fr_ms < int(end_date.timestamp() * 1000):
                all_fr_data = []
                current_since = since_fr_ms
                while current_since < int(end_date.timestamp() * 1000):
                    fr_chunk = data_collector.fetch_funding_rate_history(symbol=symbol, since=current_since, limit=100)
                    if not fr_chunk:
                        break

                    last_ts_in_all_data = all_fr_data[-1]['timestamp'] if all_fr_data else 0
                    new_data = [d for d in fr_chunk if d['timestamp'] > last_ts_in_all_data]

                    if not new_data:
                        break

                    all_fr_data.extend(new_data)
                    current_since = new_data[-1]['timestamp'] + 1
                    time.sleep(data_collector.exchange.rateLimit / 1000)

                if all_fr_data:
                    data_collector.save_funding_rates_to_db(all_fr_data, symbol)
            else:
                print(f"Funding rate data for {asset} is already up to date.")
            time.sleep(1)

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
