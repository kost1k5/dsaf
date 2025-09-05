import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import json
import argparse
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def objective(trial, X, y):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    # Define the hyperparameter search space
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }

    # Use TimeSeriesSplit for more robust validation during tuning
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_index, val_index in tscv.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        scores.append(f1_score(y_val_fold, preds, average='weighted', zero_division=0.0))

    return np.mean(scores)


def main(asset: str, timeframe: str):
    """
    Main script to create and save a production-ready model.
    """
    print(f"\n--- Creating Production Model for {asset} ({timeframe}) ---")

    # Define directories
    LABELED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled')
    REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
    PROD_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'production')
    os.makedirs(PROD_MODEL_DIR, exist_ok=True)

    # 1. Load the labeled dataset
    labeled_path = os.path.join(LABELED_DATA_DIR, f'{asset}_{timeframe}_labeled.parquet')
    try:
        df = pd.read_parquet(labeled_path)
    except FileNotFoundError:
        print(f"Error: Labeled file not found at {labeled_path}. Run previous pipeline steps first.")
        return

    # 2. Load the selected features
    features_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_selected_features.txt')
    try:
        with open(features_path, 'r') as f:
            selected_features = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: Selected features file not found at {features_path}. Run `select_features.py` first.")
        return

    print(f"Loaded {len(selected_features)} selected features.")

    # 3. Prepare data for model
    X = df[selected_features]
    y = df['label'] + 1 # Convert labels from {-1, 0, 1} to {0, 1, 2} for LGBM

    # 4. Find Optimal Hyperparameters with Optuna
    print("\nRunning Optuna hyperparameter search on the full dataset...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100) # Increase trials for better results

    best_params = study.best_params
    print(f"Best trial F1-score: {study.best_value}")
    print(f"Best params found: {best_params}")

    # 5. Train Final Production Model
    print("\nTraining final production model with best params on the entire dataset...")
    final_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'random_state': 42,
        **best_params
    }
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X, y)
    print("Final model training complete.")

    # 6. Save Model and Feature List
    model_filename = f"prod_model_{asset}_{timeframe}.lgb"
    features_filename = f"prod_features_{asset}_{timeframe}.json"

    model_path = os.path.join(PROD_MODEL_DIR, model_filename)
    features_path = os.path.join(PROD_MODEL_DIR, features_filename)

    # Save the model
    joblib.dump(final_model, model_path)
    print(f"Production model saved to: {model_path}")

    # Save the feature list
    with open(features_path, 'w') as f:
        json.dump(selected_features, f)
    print(f"Production feature list saved to: {features_path}")

    print("\n--- Production Model Creation Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Production Model Creation Orchestrator")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to process.")
    parser.add_argument("--timeframe", type=str, default="1h", help="The OHLCV timeframe to use.")
    args = parser.parse_args()

    main(asset=args.asset, timeframe=args.timeframe)
