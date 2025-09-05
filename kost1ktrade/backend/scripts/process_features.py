import os
import pandas as pd
import argparse

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing.feature_generator import FeatureGenerator

def load_data(asset: str, timeframe: str, data_dir: str) -> dict:
    """
    Loads all necessary raw data files for a given asset.
    Handles missing files gracefully.
    Also loads ETH data for correlation features.
    """
    print(f"Loading data for {asset} at {timeframe} timeframe...")
    data = {}

    # Define paths
    paths = {
        'ohlcv': (os.path.join(data_dir, f'{asset}_ohlcv_{timeframe}.csv'), ['timestamp']),
        'ohlcv_4h': (os.path.join(data_dir, f'{asset}_ohlcv_4h.csv'), ['timestamp']),
        'ohlcv_1d': (os.path.join(data_dir, f'{asset}_ohlcv_1d.csv'), ['timestamp']),
        'funding_rate': (os.path.join(data_dir, f'{asset}_funding_rates.csv'), ['timestamp']),
        'macro': (os.path.join(data_dir, 'macro_data.csv'), ['Date']),
        'fng': (os.path.join(data_dir, 'fng_data.csv'), ['timestamp']),
        'news': (os.path.join(data_dir, 'news_headlines.csv'), ['published'])
    }

    # Also load ETH data if the main asset is not ETH
    if asset != 'ETH':
        paths['eth_ohlcv'] = (os.path.join(data_dir, f'ETH_ohlcv_{timeframe}.csv'), ['timestamp'])


    # Load dataframes
    for key, (path, parse_dates_cols) in paths.items():
        try:
            data[key] = pd.read_csv(path, parse_dates=parse_dates_cols)
            print(f"  - Loaded {key} data.")
        except FileNotFoundError:
            print(f"  - WARNING: File not found for {key} at {path}. Skipping.")
            data[key] = None

    if data.get('ohlcv') is None:
        raise FileNotFoundError(f"Critical data 'ohlcv' not found for asset {asset}. Cannot proceed.")

    print("Finished loading raw data.")
    return data

def main(asset: str, timeframe: str):
    """
    Main orchestration script to generate features for a given asset.
    """
    RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # 1. Load all data
    try:
        raw_data = load_data(asset, timeframe, RAW_DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure you have run the 'collect_all_data.py' script first.")
        return

    # 2. Instantiate Feature Generator
    feature_generator = FeatureGenerator(
        ohlcv_df=raw_data['ohlcv'],
        timeframe=timeframe,
        ohlcv_df_4h=raw_data.get('ohlcv_4h'),
        ohlcv_df_1d=raw_data.get('ohlcv_1d'),
        funding_rate_df=raw_data.get('funding_rate'),
        macro_df=raw_data.get('macro'),
        fng_df=raw_data.get('fng'),
        news_df=raw_data.get('news'),
        eth_ohlcv_df=raw_data.get('eth_ohlcv') # Pass ETH data for context
    )

    # 3. Run the pipeline
    features_df = feature_generator.generate_all_features()

    # 4. Save the processed data
    output_path = os.path.join(PROCESSED_DATA_DIR, f'{asset}_{timeframe}_features.parquet')
    features_df.to_parquet(output_path)

    print(f"\nSuccessfully generated {len(features_df.columns)} features for {asset}.")
    print(f"Processed data saved to: {output_path}")
    print("\n--- Feature DataFrame Info ---")
    features_df.info()
    print("\n--- Sample of Features (tail) ---")
    print(features_df.tail())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature Generation Orchestrator")
    parser.add_argument(
        "--asset",
        type=str,
        default="BTC",
        help="The crypto asset to process (e.g., BTC, ETH)."
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="The OHLCV timeframe to use as the base (e.g., 1h, 4h, 1d)."
    )
    args = parser.parse_args()

    main(asset=args.asset, timeframe=args.timeframe)
