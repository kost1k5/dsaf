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
    # (FIX) Ensure only numeric features are retained for the model
    X_numeric = X.select_dtypes(include=np.number)

    # Align y with the numeric features before any more cleaning
    y_aligned = y.loc[X_numeric.index]

    # Data Cleaning: Handle potential infinities and NaNs
    X_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)

    # A. Handle Sparse Columns
    min_required_data = 100 # Minimum number of non-NaN values for a column to be kept
    cols_before = X_numeric.columns
    X_numeric = X_numeric.dropna(axis=1, thresh=min_required_data)
    cols_after = X_numeric.columns
    dropped_cols = set(cols_before) - set(cols_after)
    if dropped_cols:
        print(f"Dropped columns with less than {min_required_data} data points: {list(dropped_cols)}")

    # B. Handle Sparse Rows
    # Important: Align X and y after dropping rows with NaNs.
    rows_to_drop = X_numeric.isnull().any(axis=1)

    X_cleaned = X_numeric[~rows_to_drop]
    y_cleaned = y_aligned[~rows_to_drop]

    # C. Impute any remaining NaNs (should be few, if any)
    X_cleaned.fillna(0, inplace=True)

    if X_cleaned.isnull().sum().sum() > 0:
        raise ValueError("NaN values remain in X after preprocessing.")
    if len(X_cleaned) != len(y_cleaned):
        raise ValueError("X and y have mismatched lengths after cleaning.")

    # ===================================================================
    # Label Encoding (Essential for LGBM Multi-class)
    # ===================================================================
    # Converts labels (e.g., -1, 0, 1) to (0, 1, 2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_cleaned)

    print("Stage 1: Calculating SHAP values for multi-class model...")

    # Train the model
    model = lgb.LGBMClassifier(objective='multiclass', n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_cleaned, y_encoded)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_cleaned)

    # ===================================================================
    # Robust SHAP Aggregation Logic (The Fix)
    # ===================================================================

    shap_sum = None

    # Case 1: 3D Array Format (As observed in the logs)
    # Format: (n_samples, n_features, n_classes)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        print("[SHAP Aggregation] Detected format: 3D Array (Samples, Features, Classes).")
        # Take absolute values
        abs_shap = np.abs(shap_values)
        # Mean across samples (axis=0) and classes (axis=2) -> (n_features,)
        shap_sum = abs_shap.mean(axis=(0, 2))

    # Case 2: Standard Multi-class (List of 2D arrays)
    # Format: List of [ (n_samples, n_features) ] * n_classes
    elif isinstance(shap_values, list) and len(shap_values) > 1 and all(isinstance(sv, np.ndarray) for sv in shap_values):
        print("[SHAP Aggregation] Detected format: List of 2D arrays (Standard Multi-class).")
        try:
            # Stack along a new axis (axis=0) -> (n_classes, n_samples, n_features)
            stacked_abs_shap = np.stack([np.abs(sv) for sv in shap_values])
            # Mean across classes (axis=0) and samples (axis=1) -> (n_features,)
            shap_sum = stacked_abs_shap.mean(axis=(0, 1))
        except ValueError:
             raise ValueError("Cannot aggregate SHAP values (List format) due to inconsistent dimensions.")

    # Case 3: Binary Classification or Regression (2D array)
    # Format: (n_samples, n_features)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        print("[SHAP Aggregation] Detected format: 2D Array (Binary/Regression).")
        # Mean across samples (axis=0) -> (n_features,)
        shap_sum = np.abs(shap_values).mean(axis=0)

    # Case 4: Edge case handling (e.g., list with one element)
    elif isinstance(shap_values, list) and len(shap_values) <= 1:
         if shap_values:
             print("[SHAP Aggregation] Detected format: List with single 2D array.")
             sv_data = shap_values[0]
             if isinstance(sv_data, np.ndarray) and sv_data.ndim == 2:
                  shap_sum = np.abs(sv_data).mean(axis=0)
             else:
                 raise ValueError(f"Unexpected dimensions or type for single element in SHAP list.")
         else:
             raise ValueError("SHAP values list is empty.")

    else:
        # Handle unexpected formats
        raise ValueError(f"Unexpected format for shap_values. Type: {type(shap_values)}. Dimensions (if applicable): {getattr(shap_values, 'ndim', 'N/A')}")

    # Finalization
    if shap_sum is None:
         raise RuntimeError("SHAP aggregation failed to produce a result.")

    # Ensure the result is flat
    shap_sum = np.ravel(shap_sum)

    # ===================================================================
    # Verification and Selection
    # ===================================================================
    if len(X_cleaned.columns) != len(shap_sum):
        # This error should not occur with the logic above, but is kept as a safeguard.
        raise ValueError(f"ERROR: Dimension mismatch remains. Features: {len(X_cleaned.columns)}, SHAP scores: {len(shap_sum)}")

    # Create importance DataFrame (This should now succeed)
    importance_df = pd.DataFrame({'feature': X_cleaned.columns, 'shap_importance': shap_sum})

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

    # Stage 2: (Placeholder for correlation analysis if implemented)
    # ...

    final_features = selected_features_stage1 # Placeholder

    # Return the final list of features, the raw shap values, and the cleaned data for plotting
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
