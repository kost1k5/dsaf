import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import json
import argparse
from sklearn.metrics import classification_report, precision_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from functools import partial
import lightgbm as lgb


# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings

def objective(trial, X_train, y_train, selected_features):
    """
    Objective function for Optuna hyperparameter tuning, targeting PR-AUC,
    with proper scaling within each CV fold to prevent data leakage.
    """
    # Calculate scale_pos_weight for handling class imbalance within the fold
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'random_state': 42,
        'verbosity': -1,
        'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': scale_pos_weight
    }

    # No pipeline here to correctly handle scaling with early stopping
    model = lgb.LGBMClassifier(**params)

    X_selected = X_train[selected_features]
    y_encoded = LabelEncoder().fit_transform(y_train)

    inner_cv = TimeSeriesSplit(n_splits=3)
    scores = []
    for inner_train_idx, inner_val_idx in inner_cv.split(X_selected):
        if len(inner_train_idx) == 0 or len(inner_val_idx) == 0: continue

        X_inner_train, X_inner_val = X_selected.iloc[inner_train_idx], X_selected.iloc[inner_val_idx]
        y_inner_train, y_inner_val = y_encoded[inner_train_idx], y_encoded[inner_val_idx]

        # --- Scaling within the fold ---
        scaler = StandardScaler()
        try:
            scaler.set_output(transform="pandas")
        except AttributeError:
            pass # Older scikit-learn versions do not have this
        X_inner_train_scaled = scaler.fit_transform(X_inner_train)
        X_inner_val_scaled = scaler.transform(X_inner_val)
        # --- End Scaling ---

        model.fit(X_inner_train_scaled, y_inner_train,
                  eval_set=[(X_inner_val_scaled, y_inner_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

        preds_proba = model.predict_proba(X_inner_val_scaled)[:, 1]
        scores.append(average_precision_score(y_inner_val, preds_proba))

    return np.mean(scores) if scores else 0.0

def main(asset: str, timeframe: str):
    print(f"\n--- Creating Production Model for {asset} ({timeframe}) ---")

    # --- Define Dirs ---
    LABELED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled')
    REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
    PROD_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'production')
    os.makedirs(PROD_MODEL_DIR, exist_ok=True)

    # --- 1. Load Labeled Data ---
    labeled_path = os.path.join(LABELED_DATA_DIR, f'{asset}_{timeframe}_labeled.parquet')
    try:
        df = pd.read_parquet(labeled_path)
        if 'index' in df.columns:
            df.set_index('index', inplace=True)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr); sys.exit(1)

    # --- 2. Load Definitive Feature List ---
    features_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_selected_features.json')
    print(f"Loading selected features from: {features_path}")
    try:
        with open(features_path, 'r') as f:
            final_selected_features = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Selected features file not found at {features_path}. Run select_features.py first.", file=sys.stderr)
        sys.exit(1)
    print(f"Successfully loaded {len(final_selected_features)} selected features.")

    # --- 3. Prepare Data for Model ---
    X_full = df[final_selected_features].copy()
    y_full = df['label'].copy()

    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_full.ffill(inplace=True)
    X_full.fillna(0, inplace=True)
    y_full = y_full.loc[X_full.index]

    # --- 4. Walk-Forward Validation and Hyperparameter Tuning ---
    outer_cv = TimeSeriesSplit(n_splits=3)
    all_reports = []
    successful_folds = 0
    best_params = {}

    print("\n--- Starting Walk-Forward Validation with Hyperparameter Tuning ---")
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_full)):
        print(f"\n--- Processing Fold {fold+1}/{outer_cv.get_n_splits()} for {asset} ---")
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

        if len(X_train) < settings.ML.MIN_TRAIN_SAMPLES:
            print(f"  WARNING: Skipping Fold... Insufficient training data ({len(X_train)} < {settings.ML.MIN_TRAIN_SAMPLES}).")
            continue

        print("  [Hyperparameter Tuning] Running Optuna study for this fold...")
        objective_with_data = partial(objective, X_train=X_train, y_train=y_train, selected_features=final_selected_features)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_with_data, n_trials=settings.ML.OPTUNA_TRIALS)

        best_params = study.best_params
        print(f"  Best PR-AUC in fold tuning: {study.best_value:.4f}")

        # Final model for this fold
        final_fold_params = best_params.copy()
        final_fold_params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

        scaler = StandardScaler()
        try:
            scaler.set_output(transform="pandas")
        except AttributeError:
            pass

        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', lgb.LGBMClassifier(objective='binary', random_state=42, **final_fold_params))
        ])
        y_train_encoded = LabelEncoder().fit_transform(y_train)
        pipeline.fit(X_train, y_train_encoded)

        y_test_encoded = LabelEncoder().fit_transform(y_test)
        preds = pipeline.predict(X_test)
        report = classification_report(y_test_encoded, preds, output_dict=True, zero_division=0)
        all_reports.append(report)
        print(f"  Fold {fold+1} Out-of-Sample Precision (weighted): {report['weighted avg']['precision']:.4f}")
        successful_folds += 1

    # --- 5. Aggregate and Display Final Results ---
    if not all_reports:
        print("\n--- Walk-Forward Validation Failed: No folds were processed. ---")
        return

    avg_precision = np.mean([r['weighted avg']['precision'] for r in all_reports])
    print(f"\n--- Walk-Forward Validation Complete ({successful_folds} successful folds) ---")
    print(f"Average Out-of-Sample Weighted Precision: {avg_precision:.4f}")

    # --- 6. Train and Save Final Production Model ---
    if not best_params:
        print("Error: No best parameters found from tuning. Cannot train final model.")
        return

    print("\nTraining final production model on the full dataset...")
    final_prod_params = best_params.copy()
    final_prod_params['scale_pos_weight'] = (y_full == 0).sum() / (y_full == 1).sum() if (y_full == 1).sum() > 0 else 1

    scaler = StandardScaler()
    try:
        scaler.set_output(transform="pandas")
    except AttributeError:
        print("Warning: scikit-learn version is too old for set_output. Update to >=1.2 for full feature name support.")

    final_pipeline = Pipeline([
        ('scaler', scaler),
        ('model', lgb.LGBMClassifier(objective='binary', random_state=42, **final_prod_params))
    ])
    y_full_encoded = LabelEncoder().fit_transform(y_full)
    final_pipeline.fit(X_full, y_full_encoded)
    print("Final model training complete.")

    joblib.dump(final_pipeline, os.path.join(PROD_MODEL_DIR, f"prod_model_{asset}_{timeframe}.joblib"))
    with open(os.path.join(PROD_MODEL_DIR, f"prod_features_{asset}_{timeframe}.json"), 'w') as f:
        json.dump(final_selected_features, f)
    print(f"Production model and feature list ({len(final_selected_features)} features) saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Production Model Creation, Selection, and Tuning Orchestrator")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to process.")
    parser.add_argument("--timeframe", type=str, default="4h", help="The OHLCV timeframe to use.")
    args = parser.parse_args()
    main(asset=args.asset, timeframe=args.timeframe)
