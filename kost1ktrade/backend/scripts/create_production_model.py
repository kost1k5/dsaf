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
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from functools import partial

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.validation import PurgedTimeSeriesSplit
from src.core.config import settings

def select_features(X: pd.DataFrame, y: pd.Series):
    """
    Performs a two-stage feature selection process on a given training set.
    Returns the final list of features and data for plotting.
    """
    shap_threshold = settings.ML.SHAP_THRESHOLD
    corr_threshold = settings.ML.CORR_THRESHOLD
    print(f"  [Feature Selection] Running on training data of shape: {X.shape}")
    print(f"  [Feature Selection] Using SHAP threshold: {shap_threshold}, Correlation threshold: {corr_threshold}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = lgb.LGBMClassifier(objective='binary', n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y_encoded)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_sum = np.abs(np.stack(shap_values)).mean(axis=(0, 1))
    else:
        shap_sum = np.abs(shap_values).mean(axis=0)

    # If shap_sum is 2D (e.g., from a multi-class model not returning a list),
    # take the mean over the class axis to get a single importance value per feature.
    if shap_sum.ndim > 1:
        shap_sum = shap_sum.mean(axis=1)

    importance_df = pd.DataFrame({'feature': X.columns, 'shap_importance': shap_sum}).sort_values(by='shap_importance', ascending=False)
    total_importance = importance_df['shap_importance'].sum()
    importance_df['shap_importance_norm'] = importance_df['shap_importance'] / total_importance if total_importance > 0 else 0

    selected_features_stage1 = importance_df[importance_df['shap_importance_norm'] > shap_threshold]['feature'].tolist()

    X_shap_selected = X[selected_features_stage1]
    corr_matrix = X_shap_selected.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for column in upper_tri.columns:
        correlated_features = upper_tri.index[upper_tri[column] > corr_threshold].tolist()
        for feature in correlated_features:
            if feature not in to_drop and column not in to_drop:
                imp_col = importance_df.loc[importance_df['feature'] == column, 'shap_importance'].iloc[0]
                imp_feat = importance_df.loc[importance_df['feature'] == feature, 'shap_importance'].iloc[0]
                if imp_col < imp_feat: to_drop.add(column)
                else: to_drop.add(feature)

    final_features = [f for f in selected_features_stage1 if f not in to_drop]
    print(f"  [Feature Selection] Completed. Selected {len(final_features)} features.")

    # For the plot, we'll use the SHAP values corresponding to the final selected features
    final_shap_values = []
    if isinstance(shap_values, list):
        # Case 1: shap_values is a list of 2D arrays (one per class)
        for i in range(len(shap_values)):
            sh_df = pd.DataFrame(shap_values[i], columns=X.columns)
            final_shap_values.append(sh_df[final_features].values)
    else:
        # Case 2: shap_values is a numpy array (can be 2D for binary or 3D for multiclass)
        if shap_values.ndim == 3:
            # For a 3D array (samples, features, classes), we slice along the feature axis.
            feature_indices = [X.columns.get_loc(f) for f in final_features]
            final_shap_values = shap_values[:, feature_indices, :]
        else:
            # For a 2D array, we can use a DataFrame to select columns.
            sh_df = pd.DataFrame(shap_values, columns=X.columns)
            final_shap_values = sh_df[final_features].values

    return final_features, final_shap_values, X[final_features]

def objective(trial, X_train, y_train, event_end_times, selected_features):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', lgb.LGBMClassifier(objective='binary', random_state=42, **params))
    ])

    X_selected = X_train[selected_features]
    y_encoded = LabelEncoder().fit_transform(y_train)

    inner_cv = PurgedTimeSeriesSplit(n_splits=3, purge_buffer_days=5, embargo_pct=0.01)
    scores = []
    for inner_train_idx, inner_val_idx in inner_cv.split(X_selected, y_encoded, event_end_times=event_end_times.loc[X_selected.index]):
        if len(inner_train_idx) == 0 or len(inner_val_idx) == 0: continue

        X_inner_train, X_inner_val = X_selected.iloc[inner_train_idx], X_selected.iloc[inner_val_idx]
        y_inner_train, y_inner_val = y_encoded[inner_train_idx], y_encoded[inner_val_idx]

        pipeline.fit(X_inner_train, y_inner_train)
        preds = pipeline.predict(X_inner_val)
        scores.append(f1_score(y_inner_val, preds, average='weighted', zero_division=0.0))

    return np.mean(scores) if scores else -1.0

def main(asset: str, timeframe: str):
    print(f"\n--- Creating Production Model for {asset} ({timeframe}) ---")

    # --- Define Dirs ---
    LABELED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled')
    REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
    PROD_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'production')
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PROD_MODEL_DIR, exist_ok=True)

    # --- 1. Load and Prepare Data ---
    labeled_path = os.path.join(LABELED_DATA_DIR, f'{asset}_{timeframe}_labeled.parquet')
    try:
        df = pd.read_parquet(labeled_path)
        # The timestamp is in a column named 'index' after being reset in apply_labels.py
        if 'index' in df.columns:
            df.set_index('index', inplace=True)

        # Ensure the index is a DatetimeIndex and is timezone-aware (UTC)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

    except Exception as e:
        print(f"Error loading data: {e}"); return

    # Ensure event_end_time is also a tz-aware datetime
    df['event_end_time'] = pd.to_datetime(df['event_end_time'])
    if df['event_end_time'].dt.tz is None:
        df['event_end_time'] = df['event_end_time'].dt.tz_localize('UTC')
    else:
        df['event_end_time'] = df['event_end_time'].dt.tz_convert('UTC')


    metadata_cols = ['label', 'event_end_time']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    X_full = df[feature_cols].select_dtypes(include=np.number).copy()
    y_full = df['label'].copy()
    event_end_times_full = df['event_end_time'].copy()

    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_full.ffill(inplace=True)
    X_full.fillna(0, inplace=True)
    y_full = y_full.loc[X_full.index]
    event_end_times_full = event_end_times_full.loc[X_full.index]

    # --- 2. Feature Selection on a Representative Training Set ---
    train_size = int(len(X_full) * 0.8)
    X_train_fs = X_full.iloc[:train_size]
    y_train_fs = y_full.iloc[:train_size]

    final_selected_features, shap_values_plot, X_plot = select_features(X_train_fs, y_train_fs)

    shap_plot_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_shap_summary.png')
    print(f"\nSaving SHAP summary plot to: {shap_plot_path}")
    shap.summary_plot(shap_values_plot, X_plot, show=False, max_display=40)
    plt.savefig(shap_plot_path, bbox_inches='tight'); plt.close()

    # --- 3. Walk-Forward Validation and Hyperparameter Tuning ---
    outer_cv = PurgedTimeSeriesSplit(n_splits=5, purge_buffer_days=5, embargo_pct=0.01)
    all_reports = []
    successful_folds = 0

    print("\n--- Starting Walk-Forward Validation with Hyperparameter Tuning ---")
    n_splits = outer_cv.get_n_splits()
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full, event_end_times_full)):
        print(f"\n--- Processing Fold {fold+1}/{n_splits} for {asset} ---")
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

        # Add a check for minimum fold size to prevent crashes on small datasets
        if len(X_train) < settings.ML.MIN_TRAIN_SAMPLES:
            print(f"  WARNING: Skipping Fold {fold+1}/{n_splits} for {asset}: Insufficient training data ({len(X_train)} samples < min {settings.ML.MIN_TRAIN_SAMPLES}).")
            continue

        print("  [Hyperparameter Tuning] Running Optuna study for this fold...")
        objective_with_data = partial(objective, X_train=X_train, y_train=y_train, event_end_times=event_end_times_full, selected_features=final_selected_features)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_with_data, n_trials=settings.ML.OPTUNA_TRIALS)

        best_params = study.best_params
        print(f"  Best F1-score in fold tuning: {study.best_value:.4f}")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', lgb.LGBMClassifier(objective='binary', random_state=42, **best_params))
        ])
        y_train_encoded = LabelEncoder().fit_transform(y_train)
        pipeline.fit(X_train[final_selected_features], y_train_encoded)

        y_test_encoded = LabelEncoder().fit_transform(y_test)
        preds = pipeline.predict(X_test[final_selected_features])
        report = classification_report(y_test_encoded, preds, output_dict=True, zero_division=0)
        all_reports.append(report)
        print(f"  Fold {fold+1} Out-of-Sample F1-Score (weighted): {report['weighted avg']['f1-score']:.4f}")
        successful_folds += 1

    # --- 4. Aggregate and Display Final Results ---
    if all_reports:
        avg_f1 = np.mean([r['weighted avg']['f1-score'] for r in all_reports])
        print(f"\n--- Walk-Forward Validation Complete ({successful_folds}/{n_splits} folds successful) ---")
        print(f"Average Out-of-Sample F1-Score across all folds: {avg_f1:.4f}")
    else:
        print("\n--- Walk-Forward Validation Failed: No folds were processed. ---")
        print("This is likely because the dataset is too small for the number of CV splits.")
        print("Consider using a longer data history.")
        return # Exit gracefully

    # --- 5. Train and Save Final Production Model ---
    if not final_selected_features:
        print("Error: No features were selected. Cannot train final model.")
        return

    print("\nTraining final production model on the full dataset...")
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', lgb.LGBMClassifier(objective='binary', random_state=42, **best_params))
    ])
    y_full_encoded = LabelEncoder().fit_transform(y_full)
    final_pipeline.fit(X_full[final_selected_features], y_full_encoded)
    print("Final model training complete.")

    joblib.dump(final_pipeline, os.path.join(PROD_MODEL_DIR, f"prod_model_{asset}_{timeframe}.joblib"))
    with open(os.path.join(PROD_MODEL_DIR, f"prod_features_{asset}_{timeframe}.json"), 'w') as f:
        json.dump(final_selected_features, f)
    print(f"Production model and feature list ({len(final_selected_features)} features) saved.")

    # Also save features to the location expected by run_backtest.py
    backtest_features_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_selected_features.txt')
    with open(backtest_features_path, 'w') as f:
        for feature in final_selected_features:
            f.write(f"{feature}\n")
    print(f"Saved feature list for backtester at: {backtest_features_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Production Model Creation, Selection, and Tuning Orchestrator")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to process.")
    parser.add_argument("--timeframe", type=str, default="4h", help="The OHLCV timeframe to use (e.g., '1h', '4h').")
    args = parser.parse_args()
    main(asset=args.asset, timeframe=args.timeframe)
