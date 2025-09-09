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

def select_features(X: pd.DataFrame, y: pd.Series, shap_threshold=0.01, corr_threshold=0.75):
    """
    Performs a two-stage feature selection process on a given training set.
    """
    print(f"  [Feature Selection] Running on training data of shape: {X.shape}")

    # --- Stage 1: SHAP-based Feature Importance ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = lgb.LGBMClassifier(objective='multiclass', n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y_encoded)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_sum = np.abs(np.stack(shap_values)).mean(axis=(0, 1))
    else:
        shap_sum = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({'feature': X.columns, 'shap_importance': shap_sum}).sort_values(by='shap_importance', ascending=False)
    total_importance = importance_df['shap_importance'].sum()
    importance_df['shap_importance_norm'] = importance_df['shap_importance'] / total_importance if total_importance > 0 else 0

    selected_features_stage1 = importance_df[importance_df['shap_importance_norm'] > shap_threshold]['feature'].tolist()

    # --- Stage 2: Correlation-based Pruning ---
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
    return final_features

def objective(trial, X_train, y_train, event_end_times, selected_features):
    """
    Objective function for Optuna hyperparameter tuning within a single CV fold.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, **params))
    ])

    X_selected = X_train[selected_features]
    y_encoded = LabelEncoder().fit_transform(y_train)

    # Inner CV loop for hyperparameter tuning
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

    # --- 1. Load and Prepare Data ---
    LABELED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled')
    labeled_path = os.path.join(LABELED_DATA_DIR, f'{asset}_{timeframe}_labeled.parquet')
    try:
        df = pd.read_parquet(labeled_path)
        if 'timestamp' in df.columns: df.set_index('timestamp', inplace=True)
        if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index, utc=True)
    except Exception as e:
        print(f"Error loading data: {e}"); return

    metadata_cols = ['label', 'event_end_time']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    X_full = df[feature_cols].select_dtypes(include=np.number).copy()
    y_full = df['label'].copy()
    event_end_times_full = df['event_end_time'].copy()

    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_full.fillna(method='ffill', inplace=True); X_full.fillna(0, inplace=True)

    # --- 2. Walk-Forward Validation and In-Fold Feature Selection ---
    outer_cv = PurgedTimeSeriesSplit(n_splits=5, purge_buffer_days=5, embargo_pct=0.01)
    all_reports = []
    final_features_list = []

    print("\n--- Starting Walk-Forward Validation with In-Fold Feature Selection ---")
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full, event_end_times_full)):
        print(f"\n--- Processing Fold {fold+1}/{outer_cv.get_n_splits()} ---")
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

        # Step 2a: Select features ONL.Y on the current training data
        selected_features = select_features(X_train, y_train)
        final_features_list.append(selected_features)

        # Step 2b: Tune hyperparameters using an inner CV loop on the training data
        print("  [Hyperparameter Tuning] Running Optuna study for this fold...")
        objective_with_data = partial(objective, X_train=X_train, y_train=y_train, event_end_times=event_end_times_full, selected_features=selected_features)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_with_data, n_trials=25) # Fewer trials for speed in the inner loop

        # Step 2c: Train model with best params on the full training data for this fold
        print("  [Model Training] Training fold model with best parameters...")
        best_params = study.best_params
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, **best_params))
        ])
        y_train_encoded = LabelEncoder().fit_transform(y_train)
        pipeline.fit(X_train[selected_features], y_train_encoded)

        # Step 2d: Evaluate on the test set for this fold
        print("  [Evaluation] Evaluating model on out-of-sample test data...")
        y_test_encoded = LabelEncoder().fit_transform(y_test)
        preds = pipeline.predict(X_test[selected_features])
        report = classification_report(y_test_encoded, preds, output_dict=True)
        all_reports.append(report)
        print(f"  Fold {fold+1} F1-Score (weighted): {report['weighted avg']['f1-score']:.4f}")

    # --- 3. Aggregate and Display Final Results ---
    avg_f1 = np.mean([r['weighted avg']['f1-score'] for r in all_reports])
    print("\n--- Walk-Forward Validation Complete ---")
    print(f"Average F1-Score across all folds: {avg_f1:.4f}")

    # --- 4. Train and Save Final Production Model ---
    print("\nTraining final production model on the full dataset...")
    # Use features from the last, most recent fold for the final model
    final_features_to_save = final_features_list[-1] if final_features_list else []

    if not final_features_to_save:
        print("Error: No features were selected. Cannot train final model.")
        return

    # Retrain on the full dataset with the best overall params (or params from last fold's study)
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, **best_params))
    ])
    y_full_encoded = LabelEncoder().fit_transform(y_full)
    final_pipeline.fit(X_full[final_features_to_save], y_full_encoded)
    print("Final model training complete.")

    # Save the model and feature list
    PROD_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'production')
    os.makedirs(PROD_MODEL_DIR, exist_ok=True)
    joblib.dump(final_pipeline, os.path.join(PROD_MODEL_DIR, f"prod_model_{asset}_{timeframe}.joblib"))
    with open(os.path.join(PROD_MODEL_DIR, f"prod_features_{asset}_{timeframe}.json"), 'w') as f:
        json.dump(final_features_to_save, f)
    print(f"Production model and feature list ({len(final_features_to_save)} features) saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Production Model Creation, Selection, and Tuning Orchestrator")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to process.")
    parser.add_argument("--timeframe", type=str, default="4h", help="The OHLCV timeframe to use.")
    args = parser.parse_args()
    main(asset=args.asset, timeframe=args.timeframe)
