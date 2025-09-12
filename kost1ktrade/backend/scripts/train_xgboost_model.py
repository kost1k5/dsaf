"""
XGBoost Model Trainer

This script trains a classification model using the XGBoost algorithm based on
pre-processed and labeled data.

Workflow:
1. Loads the labeled feature data from a .parquet file.
2. Defines the feature set (X) and target (y).
3. Splits data into training and testing sets using a time-series approach.
4. Performs hyperparameter optimization using Optuna.
5. Trains the final XGBoost model on the full training set with the best parameters.
6. Evaluates the model on the test set and prints a classification report.
7. Saves the trained model artifact and the list of features used.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
import argparse
import json
import optuna
import shap
from sklearn.metrics import f1_score

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'production')
LABELED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled')


def optimize_hyperparameters_xgb(X_train, y_train, n_trials=50):
    """
    Performs hyperparameter optimization for XGBoost using Optuna.
    """
    def objective(trial):
        # Define the search space for the hyperparameters
        param = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'verbosity': 0,
            'booster': 'gbtree',
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-7, 1.0, log=True),
            'lambda': trial.suggest_float('lambda', 1e-7, 1.0, log=True),  # L2 regularization
            'alpha': trial.suggest_float('alpha', 1e-7, 1.0, log=True),   # L1 regularization
            'tree_method': 'gpu_hist',  # Use GPU for training
            'random_state': 42,
            'n_jobs': -1
        }

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_index, val_index in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model = xgb.XGBClassifier(**param)
            model.fit(X_train_fold, y_train_fold)

            # Predict and calculate F1 score
            preds = model.predict(X_val_fold)
            # Use macro F1-score for multiclass classification
            score = f1_score(y_val_fold, preds, average='macro', zero_division=0.0)
            scores.append(score)

        # Optuna minimizes the objective, so we return the negative mean F1 score
        return -1.0 * np.mean(scores)

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print("Best trial for XGBoost:")
    trial = study.best_trial
    print(f"  Value (Negative Macro F1): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params


def train_xgboost_model(symbol: str, timeframe: str):
    """
    Main function to train an XGBoost model from a labeled feature file.
    """
    print(f"\n--- Starting XGBoost Model Training for {symbol} ---")

    # 1. Load Data
    labeled_data_path = os.path.join(LABELED_DATA_DIR, f'{symbol}_{timeframe}_labeled.parquet')
    try:
        labeled_df = pd.read_parquet(labeled_data_path)
        print(f"Loaded labeled data from '{labeled_data_path}'")
    except FileNotFoundError:
        print(f"ERROR: Labeled data file not found at '{labeled_data_path}'.")
        print("Please run 'process_features.py' and 'apply_labels.py' first.")
        return

    # 2. Prepare Data (define features and target)
    # Drop non-feature columns
    cols_to_drop = ['symbol', 'interval', 'open', 'high', 'low', 'close', 'volume', 'event_end_time', 'label']
    X = labeled_df.drop(columns=cols_to_drop, errors='ignore')

    # Ensure all feature columns are numeric
    X = X.select_dtypes(include=np.number)

    y = labeled_df['label'].copy()

    # The labels are already -1, 0, 1. XGBoost can handle this if we map them to 0, 1, 2.
    y_mapped = y.replace({-1: 0, 0: 1, 1: 2})

    print(f"Feature set shape: {X.shape}")
    print(f"Target distribution:\n{y_mapped.value_counts(normalize=True)}")


    # 3. Feature Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    # 4. Train/Test Split (Time-series aware)
    train_size = int(len(X_scaled) * 0.9)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_mapped[:train_size], y_mapped[train_size:]

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # 5. Model Training with Hyperparameter Optimization
    print("\n--- Hyperparameter Tuning for XGBoost (Optuna) ---")
    best_params = optimize_hyperparameters_xgb(X_train, y_train, n_trials=50)

    print("\n--- Training Final XGBoost Model with Best Parameters ---")
    final_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        tree_method='gpu_hist',  # Use GPU for training final model
        **best_params
    )
    final_model.fit(X_train, y_train)

    # 6. Final Evaluation
    print(f"\n--- Final Evaluation for XGBoost on {symbol} ---")
    y_pred = final_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Short (-1)', 'Neutral (0)', 'Long (1)'], zero_division=0.0))

    # 7. Save Model, Scaler, and Features
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_file = os.path.join(MODEL_DIR, f"xgb_model_{symbol}_{timeframe}.json")
    scaler_file = os.path.join(MODEL_DIR, f"scaler_{symbol}_{timeframe}.joblib")
    features_file = os.path.join(MODEL_DIR, f"features_{symbol}_{timeframe}.json")

    print(f"Saving XGBoost model to {model_file}")
    final_model.save_model(model_file)

    print(f"Saving scaler to {scaler_file}")
    joblib.dump(scaler, scaler_file)

    feature_list = list(X.columns)
    with open(features_file, 'w') as f:
        json.dump(feature_list, f)

    print(f"Feature list saved to {features_file}")
    print(f"--- XGBoost Training for {symbol} Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost prediction models from labeled data.")
    # The --symbols argument now expects the base asset name (e.g., 'BTC', 'ETH')
    parser.add_argument("--symbols", nargs='+', default=["BTC"], help="List of asset symbols (e.g., BTC ETH) to train on.")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe for candles (e.g., '1h', '4h').")

    # Removed start_date and end_date as the script now uses the full pre-processed dataset
    args = parser.parse_args()

    for symbol in args.symbols:
        train_xgboost_model(symbol=symbol, timeframe=args.timeframe)
