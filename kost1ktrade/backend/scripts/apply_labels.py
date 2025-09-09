import os
import pandas as pd
import argparse

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing.labeling import apply_triple_barrier

def main(asset: str, timeframe: str, tp_mult: float, sl_mult: float, time_limit_h: int):
    """
    Main script to apply triple-barrier labels to the feature dataset.
    """
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    LABELED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled')
    os.makedirs(LABELED_DATA_DIR, exist_ok=True)

    # 1. Load the feature dataset
    features_path = os.path.join(PROCESSED_DATA_DIR, f'{asset}_{timeframe}_features.parquet')
    try:
        features_df = pd.read_parquet(features_path)
        print(f"Loaded feature data for {asset} from {features_path}")
        print(f"[DEBUG] Columns in features_df on load in apply_labels: {features_df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: Feature file not found at {features_path}. Please run 'process_features.py' first.")
        return

    # Check if required columns exist
    if 'close' not in features_df.columns or 'ATRr_14' not in features_df.columns:
        print("Error: 'close' or 'ATRr_14' column not found in the feature set. Cannot apply labels.")
        return

    # 2. Apply triple-barrier labeling
    # Convert time limit from hours to number of periods based on timeframe
    if 'h' in timeframe:
        periods_in_hour = 1
        time_limit_periods = int(time_limit_h * periods_in_hour)
    elif 'd' in timeframe:
        periods_in_day = 24
        time_limit_periods = int(time_limit_h / periods_in_day) if time_limit_h >= 24 else 1
    else: # default to hours for other timeframes like '15m' etc.
        # This is a simplification, a more robust solution would parse the timeframe string
        time_limit_periods = time_limit_h

    print(f"Applying labels with TP={tp_mult}*ATR, SL={sl_mult}*ATR, Hold={time_limit_periods} periods.")

    label_info = apply_triple_barrier(
        close_prices=features_df['close'],
        atr=features_df['ATRr_14'],
        tp_atr_mult=tp_mult,
        sl_atr_mult=sl_mult,
        time_limit_periods=time_limit_periods
    )

    # 3. Join labels to the feature set
    labeled_df = features_df.join(label_info)

    # Drop rows where a label could not be generated (typically the last few rows)
    labeled_df.dropna(subset=['label', 'event_end_time'], inplace=True)
    labeled_df['label'] = labeled_df['label'].astype(int)

    # 4. Save the labeled data
    output_path = os.path.join(LABELED_DATA_DIR, f'{asset}_{timeframe}_labeled.parquet')
    print(f"[DEBUG] Columns in labeled_df before saving in apply_labels: {labeled_df.columns.tolist()}")
    labeled_df.to_parquet(output_path)

    print(f"\nSuccessfully generated labels for {asset}.")
    print(f"Labeled data saved to: {output_path}")
    print("\n--- Labeled DataFrame Info ---")
    labeled_df.info()
    print("\n--- Label Distribution ---")
    print(labeled_df['label'].value_counts(normalize=True))
    print("\n--- Sample of Labeled Data (tail) ---")
    print(labeled_df.tail())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triple-Barrier Labeling Orchestrator")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to process.")
    parser.add_argument("--timeframe", type=str, default="1h", help="The OHLCV timeframe to use.")
    parser.add_argument("--tp", type=float, default=1.5, help="Take-profit ATR multiplier.")
    parser.add_argument("--sl", type=float, default=1.0, help="Stop-loss ATR multiplier.")
    parser.add_argument("--hold", type=int, default=12, help="Holding period in hours for the vertical barrier.")

    args = parser.parse_args()

    main(
        asset=args.asset,
        timeframe=args.timeframe,
        tp_mult=args.tp,
        sl_mult=args.sl,
        time_limit_h=args.hold
    )
