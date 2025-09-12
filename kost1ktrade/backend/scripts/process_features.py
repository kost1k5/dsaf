import os
import pandas as pd
import argparse
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import sys

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.feature_generator import create_features
from src.database.session import SessionLocal
from src.database.models import Candle, FundingRate, MacroData, FearGreedIndex, NewsHeadline, EconomicCalendarEvent
from sqlalchemy import func

def load_data_from_db(db: Session, asset: str, timeframe: str) -> pd.DataFrame:
    """
    Loads all necessary raw data from the database for a given asset and
    merges them into a single comprehensive DataFrame.
    """
    print(f"Loading and merging data for {asset} ({timeframe}) from database...")
    symbol = f"{asset}/USDT:USDT"

    # --- Main OHLCV Data ---
    print("  - Loading primary OHLCV data...")
    main_df = pd.read_sql(
        db.query(Candle).filter(Candle.symbol == symbol, Candle.interval == timeframe).statement,
        db.bind, index_col='open_time'
    )
    if main_df.empty:
        raise FileNotFoundError(f"Critical data 'ohlcv' not found for asset {asset}. Cannot proceed.")

    # The 'open_time' index is loaded as an int (ms timestamp); convert it to datetime
    main_df.index = pd.to_datetime(main_df.index, unit='ms')
    main_df.index.name = 'timestamp'

    # --- Additional Timeframe Data ---
    # We can create these features using resampling now, simplifying the data loading
    # For now, we will rely on the main feature generator to handle this if needed

    # --- External & Macro Data ---
    print("  - Loading Funding Rate, Macro, F&G data...")
    funding_rate_df = pd.read_sql(
        db.query(FundingRate).filter(FundingRate.symbol == symbol).statement,
        db.bind, index_col='funding_time', parse_dates=['funding_time']
    )
    if not funding_rate_df.empty:
        funding_rate_df.index.name = 'timestamp'
        main_df = main_df.join(funding_rate_df[['funding_rate']], how='left')

    fng_df = pd.read_sql(db.query(FearGreedIndex).statement, db.bind, index_col='timestamp', parse_dates=['timestamp'])
    if not fng_df.empty:
        fng_df.index.name = 'timestamp'
        fng_df = fng_df.rename(columns={'value': 'fng_value'})
        main_df = pd.merge_asof(main_df.sort_index(), fng_df[['fng_value']].sort_index(), on='timestamp', direction='backward')

    macro_df = pd.read_sql(db.query(MacroData).statement, db.bind, index_col='date', parse_dates=['date'])
    if not macro_df.empty:
        macro_df.index.name = 'timestamp'
        main_df = pd.merge_asof(main_df.sort_index(), macro_df.sort_index(), left_index=True, right_index=True, direction='backward')

    # --- Clean up ---
    # Remove database ID columns that might have been loaded
    for col in ['id', 'id_x', 'id_y']:
        if col in main_df.columns:
            main_df.drop(columns=col, inplace=True)

    # Forward-fill data from sources that update less frequently (like F&G, Macro)
    main_df.ffill(inplace=True)

    print("Finished loading and merging raw data from database.")
    return main_df


def get_latest_input_timestamp(db: Session, asset: str, timeframe: str) -> datetime:
    """Gets the most recent timestamp from all input data sources for caching purposes."""
    symbol = f"{asset}/USDT:USDT"
    timestamps = [
        db.query(func.max(Candle.open_time)).filter(Candle.symbol == symbol, Candle.interval == timeframe).scalar(),
        db.query(func.max(FundingRate.funding_time)).filter(FundingRate.symbol == symbol).scalar(),
        db.query(func.max(MacroData.date)).scalar(),
        db.query(func.max(FearGreedIndex.timestamp)).scalar(),
        db.query(func.max(NewsHeadline.published_at)).scalar(),
        db.query(func.max(EconomicCalendarEvent.created_at)).scalar() # Use created_at for caching
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
            if latest_input_ts.tzinfo is None:
                latest_input_ts = latest_input_ts.replace(tzinfo=timezone.utc)
            if output_mod_time > latest_input_ts:
                print(f"'{output_path}' is already up-to-date. Skipping feature generation.")
                return

        # 1. Load and merge all data from DB
        merged_df = load_data_from_db(db, asset, timeframe)

        # --- CRITICAL DATA CHECK ---
        if merged_df.empty:
            print(f"CRITICAL ERROR: Input DataFrame for {asset} on timeframe {timeframe} is empty after loading. Halting execution.")
            sys.exit(1)

        # 2. Prepare DataFrame for feature generation
        # The new feature generator expects 'open_time' as a column
        df_for_features = merged_df.reset_index().rename(columns={'timestamp': 'open_time'})

        # 3. Generate features using the new function-based generator
        features_df = create_features(df_for_features)

        # Set the timestamp back as the index
        features_df.set_index('open_time', inplace=True)
        features_df.index.name = 'timestamp'

        # 4. Save the processed data
        print(f"[DEBUG] Columns in features_df before saving: {features_df.columns.tolist()}")
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
    parser.add_argument("--timeframe", type=str, default="4h", help="The OHLCV timeframe to use.")
    args = parser.parse_args()
    main(asset=args.asset, timeframe=args.timeframe)
