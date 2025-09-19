import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score
import joblib
import os
import sys
import argparse
import json
import optuna
import lightgbm as lgb


# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.config import settings

# --- Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'ml', 'models')
LABELED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')


def sanitize_symbol(symbol: str) -> str:
    """Converts a symbol for use in filenames."""
    return symbol.replace('/', '_')

def optimize_hyperparameters(X_train, y_train):
    """
    Performs hyperparameter optimization using Optuna with refined search space and early stopping.
    """
    # Calculate scale_pos_weight for handling class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'scale_pos_weight': scale_pos_weight,
            'n_estimators': trial.suggest_int('n_estimators', 800, 2000), # Increased for early stopping
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05), # Constrained
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 5), # Constrained
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
        }

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_index, val_index in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            if X_val_fold.empty or y_val_fold.empty:
                continue

            model = lgb.LGBMClassifier(**param)

            # Use early stopping
            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      eval_metric='logloss',
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

            preds = model.predict(X_val_fold)
            scores.append(precision_score(y_val_fold, preds, zero_division=0))

        return -1.0 * np.mean(scores) if scores else 0

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=settings.ML.OPTUNA_TRIALS)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {-trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params

def train_model(asset: str, timeframe: str):
    """
    Loads pre-processed data and a definitive feature list, then tunes and trains a model.
    """
    print(f"\n--- Starting Model Training for {asset} on {timeframe} data ---")

    # 1. Load Labeled Data
    labeled_path = os.path.join(LABELED_DATA_DIR, f'{asset}_{timeframe}_labeled.parquet')
    print(f"Loading labeled data from: {labeled_path}")
    try:
        labeled_df = pd.read_parquet(labeled_path)
    except FileNotFoundError:
        print(f"ERROR: Labeled data not found at {labeled_path}. Halting.", file=sys.stderr)
        sys.exit(1)
    print(f"Successfully loaded {len(labeled_df)} labeled events.")

    # 2. Load Definitive Feature List
    features_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_selected_features.json')
    print(f"Loading selected features from: {features_path}")
    try:
        with open(features_path, 'r') as f:
            selected_features = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Selected features file not found at {features_path}. Run select_features.py first.", file=sys.stderr)
        sys.exit(1)
    print(f"Successfully loaded {len(selected_features)} selected features.")

    # 3. Prepare Data for Model
    labeled_df.set_index('index', inplace=True)

    X = labeled_df[selected_features].copy()
    y = labeled_df['label'].copy()

    y = y.loc[X.index]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.ffill(inplace=True)
    X.bfill(inplace=True)
    X.fillna(0, inplace=True)

    print(f"Data prepared for training. Final feature shape: {X.shape}")

    # 4. Train/Test Split
    train_size = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    print(f"Train/Test split complete. Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 5. Pre-training Checks and Scaling
    if len(X_train) < settings.ML.MIN_TRAIN_SAMPLES:
        print(f"WARNING: Training data has only {len(X_train)} samples. Skipping training.")
        return

    print("\n--- Scaling Features (StandardScaler) ---")
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    print("Features scaled successfully.")

    # 6. Hyperparameter Optimization
    print("\n--- Hyperparameter Optimization (Optuna) ---")
    best_params = optimize_hyperparameters(X_train, y_train)

    best_params['objective'] = 'binary'
    best_params['random_state'] = 42
    # Calculate final scale_pos_weight on the full training set
    best_params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

    # 7. Model Training with Best Parameters
    print("\n--- Model Training (with optimized parameters) ---")
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    # 8. Evaluation
    print(f"\n--- Final Evaluation for {asset} ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)'], zero_division=0))

    # 9. Save Model and Features
    sanitized_asset = sanitize_symbol(asset)
    model_file = os.path.join(MODEL_DIR, f"lgbm_classifier_{sanitized_asset}_{timeframe}.joblib")
    features_file = os.path.join(MODEL_DIR, f"features_{sanitized_asset}_{timeframe}.json")
    scaler_file = os.path.join(MODEL_DIR, f"scaler_{sanitized_asset}_{timeframe}.joblib")

    print(f"Saving model for {asset} to {model_file}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    with open(features_file, 'w') as f:
        json.dump(selected_features, f)

    print(f"Feature list ({len(selected_features)} features) saved to {features_file}")
    print(f"Scaler saved to {scaler_file}")
    print(f"--- Training for {asset} Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on pre-processed and pre-selected features.")
    parser.add_argument("--symbols", type=str, default="BTC", help="The trading symbol to train on (e.g., 'BTC').")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe for candles (e.g., '1h', '4h').")
    args = parser.parse_args()

    train_model(asset=args.symbols, timeframe=args.timeframe)
