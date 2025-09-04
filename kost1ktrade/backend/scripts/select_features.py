import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import argparse
import matplotlib.pyplot as plt

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def select_features(X: pd.DataFrame, y: pd.Series, correlation_threshold: float = 0.9):
    """
    Performs a two-stage feature selection process.
    1. Rank features by SHAP importance.
    2. Remove highly correlated features.
    """
    print("Starting two-stage feature selection...")

    # --- Stage 1: SHAP Importance Ranking ---
    print("Stage 1: Calculating SHAP values for multi-class model...")
    # Using a multi-class LightGBM model for SHAP ranking, matching the main model.
    # We need to map y from {-1, 0, 1} to {0, 1, 2} for multiclass objective.
    y_mapped = y + 1
    model = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)
    model.fit(X, y_mapped)

    explainer = shap.TreeExplainer(model)
    # For multi-class models, shap_values returns a list of arrays (one for each class)
    shap_values = explainer.shap_values(X)

    # Aggregate SHAP values across all classes by taking the mean of mean absolute values.
    shap_sum = np.mean([np.abs(s).mean(0) for s in shap_values], axis=0)

    importance_df = pd.DataFrame({'feature': X.columns, 'shap_importance': shap_sum})
    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    print("Top 10 features by SHAP importance:")
    print(importance_df.head(10))

    # --- Stage 2: Correlation-based Pruning ---
    print("\nStage 2: Pruning features based on correlation...")

    # Get the correlation matrix
    corr_matrix = X[importance_df['feature']].corr()

    selected_features = []
    dropped_features = set()

    for feature in importance_df['feature']:
        if feature not in dropped_features:
            selected_features.append(feature)
            # Find highly correlated features that are less important
            highly_correlated = corr_matrix[feature][corr_matrix[feature] > correlation_threshold].index.tolist()

            # Add them to the drop list, but don't drop the feature itself
            for correlated_feature in highly_correlated:
                if correlated_feature != feature:
                    print(f"  - Dropping '{correlated_feature}' (correlation with '{feature}' > {correlation_threshold})")
                    dropped_features.add(correlated_feature)

    print(f"\nFeature selection complete. Kept {len(selected_features)} out of {len(X.columns)} features.")

    return selected_features, shap_values, X

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
