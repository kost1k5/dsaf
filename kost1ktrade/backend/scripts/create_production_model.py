import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import optuna
import joblib
import json
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from functools import partial

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.validation import PurgedTimeSeriesSplit

def select_features(X: pd.DataFrame, y: pd.Series, shap_threshold=0.01, corr_threshold=0.75):
    """
    Performs a two-stage feature selection process on a given training set.
    1. SHAP-based selection to find important features.
    2. Correlation-based pruning to remove redundant features.
    """
    print("\n--- Starting Feature Selection ---")
    print(f"Feature selection running on training data of shape: {X.shape}")

    # --- Stage 1: SHAP-based Feature Importance ---
    print("Stage 1: Calculating SHAP values to identify important features...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = lgb.LGBMClassifier(objective='multiclass', n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y_encoded)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Restore robust SHAP aggregation logic
    if isinstance(shap_values, list): # Standard multi-class output
        shap_sum = np.abs(np.stack(shap_values)).mean(axis=(0, 1))
    else: # Binary classification or other formats
        shap_sum = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({'feature': X.columns, 'shap_importance': shap_sum})
    importance_df = importance_df.sort_values(by='shap_importance', ascending=False)

    total_importance = importance_df['shap_importance'].sum()
    importance_df['shap_importance_norm'] = importance_df['shap_importance'] / total_importance if total_importance > 0 else 0

    selected_features_stage1 = importance_df[importance_df['shap_importance_norm'] > shap_threshold]['feature'].tolist()
    print(f"Stage 1 (SHAP) selected {len(selected_features_stage1)} features.")

    # --- Stage 2: Correlation-based Pruning ---
    print(f"Stage 2: Pruning highly correlated features (threshold: {corr_threshold})...")
    X_shap_selected = X[selected_features_stage1]
    corr_matrix = X_shap_selected.corr().abs()

    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for column in upper_tri.columns:
        correlated_features = upper_tri.index[upper_tri[column] > corr_threshold].tolist()
        for feature in correlated_features:
            if feature not in to_drop and column not in to_drop:
                importance_of_col = importance_df.loc[importance_df['feature'] == column, 'shap_importance'].iloc[0]
                importance_of_feature = importance_df.loc[importance_df['feature'] == feature, 'shap_importance'].iloc[0]
                if importance_of_col < importance_of_feature: to_drop.add(column)
                else: to_drop.add(feature)

    if to_drop: print(f"Dropped {len(to_drop)} features due to high correlation: {list(to_drop)}")

    final_features = [f for f in selected_features_stage1 if f not in to_drop]
    print(f"--- Feature Selection Complete: {len(final_features)} features selected ---")

    return final_features, shap_values, X

def objective(trial, X, y, event_end_times, selected_features):
    """
    Objective function for Optuna hyperparameter tuning, using a fixed set of features.
    """
    params = {
        'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
        'verbosity': -1, 'boosting_type': 'gbdt', 'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
    }

    X_selected = X[selected_features]
    y_encoded = LabelEncoder().fit_transform(y)

    tscv = PurgedTimeSeriesSplit(n_splits=3, purge_buffer_days=5)
    scores = []

    for train_index, val_index in tscv.split(X_selected, y_encoded, event_end_times=event_end_times):
        if len(train_index) == 0 or len(val_index) == 0: continue

        X_train_fold, X_val_fold = X_selected.iloc[train_index], X_selected.iloc[val_index]
        y_train_fold, y_val_fold = y_encoded[train_index], y_encoded[val_index]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        scores.append(f1_score(y_val_fold, preds, average='weighted', zero_division=0.0))

    return np.mean(scores) if scores else -1.0

def main(asset: str, timeframe: str):
    print(f"\n--- Creating Production Model for {asset} ({timeframe}) ---")

    # --- 1. Load Data ---
    LABELED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled')
    labeled_path = os.path.join(LABELED_DATA_DIR, f'{asset}_{timeframe}_labeled.parquet')
    try:
        df = pd.read_parquet(labeled_path)
        if 'timestamp' in df.columns: df.set_index('timestamp', inplace=True)
        if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index, utc=True)
    except Exception as e:
        print(f"Error loading data: {e}"); return

    # --- 2. Prepare Full Dataset ---
    metadata_cols = ['label', 'event_end_time']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    X_full = df[feature_cols].select_dtypes(include=np.number)
    y_full = df['label']
    event_end_times_full = df['event_end_time']

    # --- 3. Robust Data Cleaning ---
    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop columns with too many NaNs before imputing
    min_required_data = len(X_full) * 0.8
    X_full.dropna(axis=1, thresh=min_required_data, inplace=True)
    # Simple imputation for any remaining NaNs
    X_full.fillna(method='ffill', inplace=True)
    X_full.fillna(0, inplace=True) # Fill any remaining at the start

    # Align all dataframes after cleaning X
    y_full = y_full.loc[X_full.index]
    event_end_times_full = event_end_times_full.loc[X_full.index]

    # --- 4. Feature Selection on a Representative Training Set ---
    # Use the first 80% of data for a stable feature selection
    train_size = int(len(X_full) * 0.8)
    X_train_fs = X_full.iloc[:train_size]
    y_train_fs = y_full.iloc[:train_size]

    final_selected_features, shap_values_plot, X_plot = select_features(X_train_fs, y_train_fs)

    REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(REPORTS_DIR, exist_ok=True)
    shap_plot_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_shap_summary.png')
    shap.summary_plot(shap_values_plot, X_plot, show=False, max_display=40)
    plt.savefig(shap_plot_path, bbox_inches='tight'); plt.close()
    print(f"\nSaved SHAP summary plot to: {shap_plot_path}")

    # --- 5. Hyperparameter Tuning using Selected Features ---
    print("\nRunning Optuna hyperparameter search...")
    objective_with_features = partial(objective, selected_features=final_selected_features)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_with_features(trial, X_full, y_full, event_end_times_full), n_trials=50)

    best_params = study.best_params
    print(f"Best trial F1-score: {study.best_value}")
    print(f"Best params found: {best_params}")

    # --- 6. Train Final Production Model ---
    print("\nTraining final production model...")
    final_params = { 'objective': 'multiclass', 'num_class': 3, 'random_state': 42, **best_params }

    X_final = X_full[final_selected_features]
    y_final_encoded = LabelEncoder().fit_transform(y_full)

    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_final, y_final_encoded)
    print("Final model training complete.")

    # --- 7. Save Model and Feature List ---
    PROD_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'production')
    os.makedirs(PROD_MODEL_DIR, exist_ok=True)
    joblib.dump(final_model, os.path.join(PROD_MODEL_DIR, f"prod_model_{asset}_{timeframe}.lgb"))
    with open(os.path.join(PROD_MODEL_DIR, f"prod_features_{asset}_{timeframe}.json"), 'w') as f:
        json.dump(final_selected_features, f)
    print("Production model and feature list saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Production Model Creation, Selection, and Tuning Orchestrator")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to process.")
    parser.add_argument("--timeframe", type=str, default="4h", help="The OHLCV timeframe to use.")
    args = parser.parse_args()
    main(asset=args.asset, timeframe=args.timeframe)
