import os
import pandas as pd
import argparse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing.feature_generator import FeatureGenerator
from src.database.session import SessionLocal
from src.database.models import Candle, FundingRate, MacroData, FearGreedIndex, NewsHeadline
from sqlalchemy import func

def load_data_from_db(db: Session, asset: str, timeframe: str) -> dict:
    """
    Loads all necessary raw data from the database for a given asset.
    """
    print(f"Loading data for {asset} at {timeframe} timeframe from database...")
    data = {}

    # --- OHLCV Data ---
    print("  - Loading OHLCV data...")
    symbol = f"{asset}/USDT:USDT"
    data['ohlcv'] = pd.read_sql(db.query(Candle).filter(Candle.symbol == symbol, Candle.interval == timeframe).statement, db.bind, index_col='open_time', parse_dates=['open_time'])
    data['ohlcv_4h'] = pd.read_sql(db.query(Candle).filter(Candle.symbol == symbol, Candle.interval == '4h').statement, db.bind, index_col='open_time', parse_dates=['open_time'])
    data['ohlcv_1d'] = pd.read_sql(db.query(Candle).filter(Candle.symbol == symbol, Candle.interval == '1d').statement, db.bind, index_col='open_time', parse_dates=['open_time'])

    # Rename index to 'timestamp' to match previous structure
    for key in ['ohlcv', 'ohlcv_4h', 'ohlcv_1d']:
        if not data[key].empty:
            data[key].index.name = 'timestamp'


    # --- Funding Rate Data ---
    print("  - Loading Funding Rate data...")
    data['funding_rate'] = pd.read_sql(db.query(FundingRate).filter(FundingRate.symbol == symbol).statement, db.bind, index_col='funding_time', parse_dates=['funding_time'])
    if not data['funding_rate'].empty:
        data['funding_rate'].index.name = 'timestamp'


    # --- Asset-Agnostic Data ---
    print("  - Loading Macro, F&G, and News data...")
    data['macro'] = pd.read_sql(db.query(MacroData).statement, db.bind, index_col='date', parse_dates=['date'])
    data['fng'] = pd.read_sql(db.query(FearGreedIndex).statement, db.bind, index_col='timestamp', parse_dates=['timestamp'])
    data['news'] = pd.read_sql(db.query(NewsHeadline).statement, db.bind, index_col='published_at', parse_dates=['published_at'])
    # Rename columns to match previous structure
    if not data['macro'].empty: data['macro'].index.name = 'Date'
    if not data['fng'].empty: data['fng'] = data['fng'].rename(columns={'value': 'fng_value'})
    if not data['news'].empty: data['news'] = data['news'].rename(columns={'published_at': 'published'})

    # --- ETH Data for Correlation ---
    if asset != 'ETH':
        print("  - Loading ETH data for context...")
        eth_symbol = "ETH/USDT:USDT"
        data['eth_ohlcv'] = pd.read_sql(db.query(Candle).filter(Candle.symbol == eth_symbol, Candle.interval == timeframe).statement, db.bind, index_col='open_time', parse_dates=['open_time'])
        if not data['eth_ohlcv'].empty:
            data['eth_ohlcv'].index.name = 'timestamp'

    if data.get('ohlcv') is None or data.get('ohlcv').empty:
        raise FileNotFoundError(f"Critical data 'ohlcv' not found for asset {asset}. Cannot proceed.")

    print("Finished loading raw data from database.")
    return data

def get_latest_input_timestamp(db: Session, asset: str, timeframe: str) -> datetime:
    """Gets the most recent timestamp from all input data sources for caching purposes."""
    symbol = f"{asset}/USDT:USDT"
    timestamps = [
        db.query(func.max(Candle.open_time)).filter(Candle.symbol == symbol, Candle.interval == timeframe).scalar(),
        db.query(func.max(FundingRate.funding_time)).filter(FundingRate.symbol == symbol).scalar(),
        db.query(func.max(MacroData.date)).scalar(),
        db.query(func.max(FearGreedIndex.timestamp)).scalar(),
        db.query(func.max(NewsHeadline.published_at)).scalar()
    ]
    # Filter out None values and find the max
    valid_timestamps = [ts for ts in timestamps if ts is not None]
    return max(valid_timestamps) if valid_timestamps else None


def main(asset: str, timeframe: str):
    """
    Main orchestration script to generate features for a given asset.
    """
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, f'{asset}_{timeframe}_features.parquet')

    db: Session = SessionLocal()
    try:
        # --- Smart Caching Check ---
        latest_input_ts = get_latest_input_timestamp(db, asset, timeframe)
        if os.path.exists(output_path) and latest_input_ts:
            output_mod_time = datetime.fromtimestamp(os.path.getmtime(output_path), tz=timezone.utc)
            # Make latest_input_ts timezone-aware if it's not already
            if latest_input_ts.tzinfo is None:
                latest_input_ts = latest_input_ts.replace(tzinfo=timezone.utc)

            if output_mod_time > latest_input_ts:
                print(f"'{output_path}' is already up-to-date. Skipping feature generation.")
                return

        # 1. Load all data from DB
        raw_data = load_data_from_db(db, asset, timeframe)

        # 2. Instantiate Feature Generator
        feature_generator = FeatureGenerator(
            asset=asset,
            ohlcv_df=raw_data['ohlcv'],
            timeframe=timeframe,
            ohlcv_df_4h=raw_data.get('ohlcv_4h'),
            ohlcv_df_1d=raw_data.get('ohlcv_1d'),
            funding_rate_df=raw_data.get('funding_rate'),
            macro_df=raw_data.get('macro'),
            fng_df=raw_data.get('fng'),
            news_df=raw_data.get('news'),
            eth_ohlcv_df=raw_data.get('eth_ohlcv')
        )

        # 3. Run the pipeline
        features_df = feature_generator.generate_all_features()

        # 4. Save the processed data
        features_df.to_parquet(output_path)

        print(f"\nSuccessfully generated {len(features_df.columns)} features for {asset}.")
        print(f"Processed data saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure you have run the 'collect_all_data.py' script first.")
        return
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature Generation Orchestrator")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to process.")
    parser.add_argument("--timeframe", type=str, default="1h", help="The OHLCV timeframe to use.")
    args = parser.parse_args()
    main(asset=args.asset, timeframe=args.timeframe)
