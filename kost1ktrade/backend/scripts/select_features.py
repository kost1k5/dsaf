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
    # FIX 1: Preprocessing and Cleaning
    # ===================================================================
    X_cleaned = X.copy()

    # A. Handle Sparse Columns
    # Drop columns that are entirely NaN (like 'news_sentiment') or have very few data points.
    min_required_data = 100 # Define a minimum threshold for data presence
    X_cleaned = X_cleaned.dropna(axis=1, how='all')

    insufficient_data_cols = X_cleaned.columns[X_cleaned.isnull().sum() > (len(X_cleaned) - min_required_data)]
    if len(insufficient_data_cols) > 0:
        print(f"Dropping columns with insufficient data: {list(insufficient_data_cols)}")
        X_cleaned = X_cleaned.drop(columns=insufficient_data_cols)

    # B. Impute remaining NaNs
    # We fill remaining NaNs (e.g., resulting from indicator calculations or lags) with 0.
    X_cleaned = X_cleaned.fillna(0)

    if X_cleaned.isnull().sum().sum() > 0:
        raise ValueError("NaN values remain in X after preprocessing.")

    # ===================================================================
    # FIX 2: Label Encoding
    # ===================================================================
    # LGBM multi-class requires labels to be 0, 1, ..., n_classes-1.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Stage 1: Calculating SHAP values for multi-class model...")

    # Train the model on cleaned data
    model = lgb.LGBMClassifier(objective='multiclass', n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_cleaned, y_encoded)

    # Calculate SHAP values using the cleaned data
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_cleaned)

    # ===================================================================
    # FIX 3: Correct SHAP Aggregation (Resolves the ValueError)
    # ===================================================================

    if isinstance(shap_values, list) and len(shap_values) > 1:
        # Multi-class case: We aggregate importance across all classes and samples.

        # 1. Take the absolute value of SHAP values for each class and stack them.
        # Resulting shape: (n_classes, n_samples, n_features)
        try:
            stacked_abs_shap = np.stack([np.abs(sv) for sv in shap_values])
        except ValueError as e:
            print("Error stacking SHAP values. Check if all classes have consistent output shapes.")
            raise e

        # 2. Calculate the mean across classes (axis=0) AND samples (axis=1)
        # Resulting shape: (n_features,) -> A 1D array
        shap_sum = stacked_abs_shap.mean(axis=(0, 1))

    elif isinstance(shap_values, np.ndarray) or (isinstance(shap_values, list) and len(shap_values) <= 1):
        # Binary, regression, or edge case
        if isinstance(shap_values, list) and shap_values:
             shap_values = shap_values[0]
        elif isinstance(shap_values, list) and not shap_values:
             raise ValueError("SHAP values list is empty.")

        shap_sum = np.abs(shap_values).mean(axis=0)

    else:
        raise ValueError("Unexpected format for shap_values.")

    # Ensure the result is flat (safety check)
    shap_sum = np.ravel(shap_sum)

    # Create importance DataFrame (This line should now succeed)
    importance_df = pd.DataFrame({'feature': X_cleaned.columns, 'shap_importance': shap_sum})

    # ... (The rest of the function logic for normalization, selection, visualization)

    # IMPORTANT: The main script expects the function to return the features list,
    # the shap values, and the DataFrame used for plotting (X_for_plot).
    # We must return the cleaned data (X_cleaned).

    # Example continuation (adjust based on the rest of your script's implementation):

    # Normalize importance (Example)
    importance_df = importance_df.sort_values(by='shap_importance', ascending=False)
    importance_df['shap_importance_norm'] = importance_df['shap_importance'] / importance_df['shap_importance'].sum()

    # Select features above threshold (Example)
    selected_features_stage1 = importance_df[importance_df['shap_importance_norm'] > shap_threshold]['feature'].tolist()
    print(f"Stage 1 selected {len(selected_features_stage1)} features.")

    # Stage 2: (Placeholder if correlation analysis follows)
    # ...

    final_features = selected_features_stage1 # Placeholder

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
