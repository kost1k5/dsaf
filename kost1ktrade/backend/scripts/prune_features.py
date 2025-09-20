import os
import json
import argparse
import pandas as pd

def prune_features(asset: str, timeframe: str, num_to_remove: int):
    """
    Prunes the least important features from a feature set based on
    a pre-computed feature importance list.

    Args:
        asset (str): The asset symbol (e.g., 'BTC').
        timeframe (str): The timeframe (e.g., '1h').
        num_to_remove (int): The number of least important features to remove.
    """
    print(f"--- Pruning {num_to_remove} features for {asset} ({timeframe}) ---")

    REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')

    # 1. Define file paths
    importance_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_feature_importances.json')
    original_features_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_selected_features.json')
    pruned_features_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_selected_features_pruned.json')

    # 2. Load the feature importance list
    try:
        importances = pd.read_json(importance_path, typ='series')
        importances = importances.sort_values(ascending=True) # Sort ascending to find least important
    except FileNotFoundError:
        print(f"ERROR: Feature importance file not found at {importance_path}")
        print("Please run the training script first to generate importances.")
        return
    except Exception as e:
        print(f"Error loading feature importances: {e}")
        return

    # 3. Load the original full feature list
    try:
        with open(original_features_path, 'r') as f:
            original_features = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Original selected features file not found at {original_features_path}")
        return

    # 4. Identify the features to prune
    features_to_prune = importances.head(num_to_remove).index.tolist()
    print(f"Identified the following {num_to_remove} least important features to prune:")
    for feature in features_to_prune:
        print(f"  - {feature} (Importance: {importances.get(feature, 'N/A')})")

    # 5. Create the new pruned feature list
    pruned_features = [f for f in original_features if f not in features_to_prune]

    # 6. Save the new feature list
    with open(pruned_features_path, 'w') as f:
        json.dump(pruned_features, f, indent=4)

    print(f"\nSuccessfully pruned {len(features_to_prune)} features.")
    print(f"Original feature count: {len(original_features)}")
    print(f"New feature count: {len(pruned_features)}")
    print(f"Pruned feature list saved to: {pruned_features_path}")
    print("\nNOTE: To use this new feature set, you may need to modify the training script to load this file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prune features based on importance.")
    parser.add_argument("--asset", required=True, type=str, help="The crypto asset to process.")
    parser.add_argument("--timeframe", required=True, type=str, help="The OHLCV timeframe to use.")
    parser.add_argument("--num_to_remove", required=True, type=int, help="The number of least important features to remove.")
    args = parser.parse_args()

    prune_features(asset=args.asset, timeframe=args.timeframe, num_to_remove=args.num_to_remove)
