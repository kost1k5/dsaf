import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import json
import argparse
from sklearn.preprocessing import LabelEncoder

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings

def select_features(X: pd.DataFrame, y: pd.Series, asset: str, timeframe: str):
    """
    Performs a two-stage feature selection process and saves the final list.
    """
    REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(REPORTS_DIR, exist_ok=True)

    shap_threshold = settings.ML.SHAP_THRESHOLD
    corr_threshold = settings.ML.CORR_THRESHOLD
    print(f"  [Feature Selection] Running on training data of shape: {X.shape}")
    print(f"  [Feature Selection] Using SHAP threshold: {shap_threshold}, Correlation threshold: {corr_threshold}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Use a simple, fast model for feature selection
    model = lgb.LGBMClassifier(objective='binary', n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y_encoded)

    print("  Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_sum = np.abs(np.stack(shap_values)).mean(axis=(0, 1))
    else:
        shap_sum = np.abs(shap_values).mean(axis=0)

    if shap_sum.ndim > 1:
        shap_sum = shap_sum.mean(axis=1)

    importance_df = pd.DataFrame({'feature': X.columns, 'shap_importance': shap_sum}).sort_values(by='shap_importance', ascending=False)
    total_importance = importance_df['shap_importance'].sum()
    importance_df['shap_importance_norm'] = importance_df['shap_importance'] / total_importance if total_importance > 0 else 0

    # Stage 1: Select features based on SHAP importance
    selected_features_stage1 = importance_df[importance_df['shap_importance_norm'] > shap_threshold]['feature'].tolist()
    print(f"  [Stage 1] Selected {len(selected_features_stage1)} features based on SHAP threshold.")

    # Stage 2: Remove highly correlated features
    X_shap_selected = X[selected_features_stage1]
    corr_matrix = X_shap_selected.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for column in upper_tri.columns:
        correlated_features = upper_tri.index[upper_tri[column] > corr_threshold].tolist()
        for feature in correlated_features:
            if feature not in to_drop and column not in to_drop:
                # Compare features and drop the one with lower SHAP importance
                imp_col = importance_df.loc[importance_df['feature'] == column, 'shap_importance'].iloc[0]
                imp_feat = importance_df.loc[importance_df['feature'] == feature, 'shap_importance'].iloc[0]
                if imp_col < imp_feat:
                    to_drop.add(column)
                else:
                    to_drop.add(feature)

    final_features = [f for f in selected_features_stage1 if f not in to_drop]
    print(f"  [Stage 2] Removed {len(to_drop)} highly correlated features.")
    print(f"  [Feature Selection] Completed. Final number of features: {len(final_features)}")

    # Save the final list of features
    features_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_selected_features.json')
    print(f"  Saving selected feature list to: {features_path}")
    with open(features_path, 'w') as f:
        json.dump(final_features, f)

    return final_features

def main(asset: str, timeframe: str):
    """
    Main function to load data, run feature selection, and save the results.
    """
    print(f"\n--- Running Feature Selection for {asset} ({timeframe}) ---")
    LABELED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled')

    # Load labeled data
    labeled_path = os.path.join(LABELED_DATA_DIR, f'{asset}_{timeframe}_labeled.parquet')
    print(f"Loading labeled data from: {labeled_path}")
    try:
        df = pd.read_parquet(labeled_path)
    except FileNotFoundError:
        print(f"ERROR: Labeled data not found at {labeled_path}.")
        sys.exit(1)

    # Prepare data for feature selection
    df.set_index('index', inplace=True, drop=False)

    cols_to_drop = ['open', 'high', 'low', 'close', 'volume', 'signal', 'event_end_time', 'created_at', 'index', 'label']
    feature_cols = [col for col in df.columns if col not in cols_to_drop]
    X = df[feature_cols].copy()
    y = df['label'].copy()

    # Clean data just in case
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.ffill(inplace=True)
    X.bfill(inplace=True)
    X.fillna(0, inplace=True)

    # Run selection process
    select_features(X, y, asset, timeframe)
    print(f"\n--- Feature Selection for {asset} ({timeframe}) Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Selects and saves the best features for a given asset.")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to process.")
    parser.add_argument("--timeframe", type=str, default="4h", help="The OHLCV timeframe to use.")
    args = parser.parse_args()
    main(asset=args.asset, timeframe=args.timeframe)
