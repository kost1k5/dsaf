import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def select_features(X: pd.DataFrame, y: pd.Series, shap_threshold=0.01, corr_threshold=0.75):
    print("Starting two-stage feature selection...")

    # ===================================================================
    # Preprocessing and Cleaning (Essential)
    # ===================================================================
    X_cleaned = X.copy()

    # A. Handle Sparse Columns (e.g., 'news_sentiment', 'oi_pct_change')
    min_required_data = 100
    X_cleaned = X_cleaned.dropna(axis=1, how='all')

    insufficient_data_cols = X_cleaned.columns[X_cleaned.isnull().sum() > (len(X_cleaned) - min_required_data)]
    if len(insufficient_data_cols) > 0:
        # print(f"Dropping columns with insufficient data: {list(insufficient_data_cols)}")
        X_cleaned = X_cleaned.drop(columns=insufficient_data_cols)

    # B. Impute remaining NaNs
    X_cleaned = X_cleaned.fillna(0)

    if X_cleaned.isnull().sum().sum() > 0:
        raise ValueError("NaN values remain in X after preprocessing.")

    # ===================================================================
    # Label Encoding (Essential for LGBM Multi-class)
    # ===================================================================
    # Converts labels (e.g., -1, 0, 1) to (0, 1, 2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ===================================================================
    # DEBUGGING BLOCK 1: Input Dimensions
    # ===================================================================
    print("\n--- DEBUGGING INFO START ---")
    print(f"[Debug 1] Shape of X_cleaned (Samples, Features): {X_cleaned.shape}")
    print(f"[Debug 1] Length of X_cleaned.columns: {len(X_cleaned.columns)}")
    # ===================================================================

    print("Stage 1: Calculating SHAP values for multi-class model...")

    # Train the model
    model = lgb.LGBMClassifier(objective='multiclass', n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_cleaned, y_encoded)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_cleaned)

    # ===================================================================
    # DEBUGGING BLOCK 2: SHAP Output Structure
    # ===================================================================
    print(f"[Debug 2] Type of shap_values: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"[Debug 2] Length of shap_values list (Classes): {len(shap_values)}")
        for i, sv in enumerate(shap_values):
            if hasattr(sv, 'shape'):
                 # Shapes should match X_cleaned (Samples, Features)
                 print(f"  [Debug 2] Shape of shap_values[{i}]: {sv.shape}")
            else:
                 print(f"  [Debug 2] shap_values[{i}] is not an array (Type: {type(sv)})")
    elif isinstance(shap_values, np.ndarray):
        print(f"[Debug 2] Shape of shap_values: {shap_values.shape}")
    # ===================================================================

    # ===================================================================
    # SHAP Aggregation Logic (With integrated debugging)
    # ===================================================================

    if isinstance(shap_values, list) and len(shap_values) > 1:
        # Multi-class case
        try:
            # 1. Calculate absolute values and stack
            # This requires all arrays in the list (Debug 2) to have the same shape
            stacked_abs_shap = np.stack([np.abs(sv) for sv in shap_values])

            # ===================================================================
            # DEBUGGING BLOCK 3: Aggregation Intermediate
            # ===================================================================
            # Shape expected: (n_classes, n_samples, n_features)
            print(f"[Debug 3] Shape of stacked_abs_shap: {stacked_abs_shap.shape}")
            # ===================================================================

        except ValueError as e:
            print(f"[Debug 3] ERROR: Failed to stack SHAP values: {e}. Check if shapes in Debug 2 are consistent.")
            # If stacking fails, we cannot proceed reliably.
            raise ValueError("Cannot aggregate SHAP values due to inconsistent dimensions across classes.")

        else:
             # 2. Calculate the mean across classes (axis=0) AND samples (axis=1)
             # Resulting shape expected: (n_features,)
            shap_sum = stacked_abs_shap.mean(axis=(0, 1))

    elif isinstance(shap_values, np.ndarray) or (isinstance(shap_values, list) and len(shap_values) <= 1):
        # Binary, regression, or edge case
        if isinstance(shap_values, list):
             if shap_values:
                 sv_data = shap_values[0]
             else:
                 raise ValueError("SHAP values list is empty.")
        else:
            sv_data = shap_values # It's an ndarray

        shap_sum = np.abs(sv_data).mean(axis=0)

    else:
        raise ValueError("Unexpected format for shap_values.")

    # Ensure the result is flat
    shap_sum = np.ravel(shap_sum)

    # ===================================================================
    # DEBUGGING BLOCK 4: Final Dimensions Before Crash
    # ===================================================================
    if hasattr(shap_sum, 'shape'):
        print(f"[Debug 4] Shape of final shap_sum: {shap_sum.shape}")
    else:
        print(f"[Debug 4] shap_sum is not an array (Type: {type(shap_sum)})")

    print(f"[Debug 4] Length of final shap_sum: {len(shap_sum)}")

    if len(X_cleaned.columns) != len(shap_sum):
        print("[Debug 4] ERROR: Mismatch detected! Feature count and shap_sum length are different.")
    else:
        print("[Debug 4] SUCCESS: Dimensions match.")
    print("--- DEBUGGING INFO END ---\n")
    # ===================================================================

    # Create importance DataFrame (Where the error occurs)
    importance_df = pd.DataFrame({'feature': X_cleaned.columns, 'shap_importance': shap_sum})

    # ... (The rest of the function logic)

    # Example continuation (ensure the function can return values if successful)
    # Normalize importance
    importance_df = importance_df.sort_values(by='shap_importance', ascending=False)
    total_importance = importance_df['shap_importance'].sum()
    if total_importance > 0:
        importance_df['shap_importance_norm'] = importance_df['shap_importance'] / total_importance
    else:
        # Handle case where all importances are zero
        importance_df['shap_importance_norm'] = 0.0

    # Select features above threshold
    selected_features_stage1 = importance_df[importance_df['shap_importance_norm'] > shap_threshold]['feature'].tolist()
    print(f"Stage 1 selected {len(selected_features_stage1)} features.")

    # Stage 2: (Placeholder)
    final_features = selected_features_stage1

    # Return cleaned data for plotting
    return final_features, shap_values, X_cleaned

def main(asset: str, timeframe: str):
    """
    Main script to run feature selection for a given asset.
    """
    LABELED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled')
    REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load the labeled dataset
    labeled_path = os.path.join(LABELED_DATA_DIR, f'{asset}_{timeframe}_labeled.parquet')
    try:
        df = pd.read_parquet(labeled_path)
        print(f"Loaded labeled data for {asset} from {labeled_path}")
    except FileNotFoundError:
        print(f"Error: Labeled file not found at {labeled_path}. Please run 'apply_labels.py' first.")
        return

    # 2. Prepare data for model
    # Define features (X) and target (y)
    # Exclude non-feature columns and others that were transformed
    # This list should be more robust, but for now we list them manually
    cols_to_drop = [col for col in ['open', 'high', 'low', 'close', 'volume', 'label', 'oi_value', 'SPY', 'VIX', 'DXY'] if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df['label']

    # 3. Run feature selection
    final_features, shap_values_all, X_for_plot = select_features(X, y)

    # 4. Save the results
    # Save the list of final features
    features_list_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_selected_features.txt')
    with open(features_list_path, 'w') as f:
        for feature in final_features:
            f.write(f"{feature}\n")
    print(f"\nSaved selected features list to: {features_list_path}")

    # Save the SHAP summary plot
    shap_plot_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_shap_summary.png')
    shap.summary_plot(shap_values_all, X_for_plot, show=False)
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary plot to: {shap_plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature Selection Orchestrator")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to process.")
    parser.add_argument("--timeframe", type=str, default="1h", help="The OHLCV timeframe to use.")
    args = parser.parse_args()

    main(asset=args.asset, timeframe=args.timeframe)
