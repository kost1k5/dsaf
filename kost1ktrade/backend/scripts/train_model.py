import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, precision_score
import joblib
import os
import sys
import argparse
from datetime import datetime, timedelta
import json
import optuna
import shap
from statsmodels.tsa.stattools import adfuller

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collector.data_cacher import DataCacher
from src.data_collector.external_data import get_fear_and_greed_index, get_onchain_metrics
from src.ml.feature_generator import create_features, create_labels

# --- Configuration ---
MODEL_DIR = "src/ml/models"

def sanitize_symbol(symbol: str) -> str:
    """Converts a symbol like 'BTC/USDT' to 'BTC_USDT' for filenames."""
    return symbol.replace('/', '_')

def optimize_hyperparameters(X_train, y_train):
    """
    Performs hyperparameter optimization using Optuna.
    """
    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'is_unbalance': True,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100), # Increased min_child_samples
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
        }

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_index, val_index in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            if X_val_fold.empty:
                continue

            model = lgb.LGBMClassifier(**param)
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            scores.append(precision_score(y_val_fold, preds, zero_division=0))

        return -1.0 * np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100) # Increased to 100

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {-trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params

def train_model(symbol: str, timeframe: str, start_date: str, end_date: str):
    """
    Fetches data, prepares it, tunes and trains a LightGBM model, and saves it.
    """
    print(f"\n--- Starting Model Training for {symbol} ({start_date} to {end_date}) ---")

    # 1. Fetch Data
    print(f"Fetching historical data for {symbol}...")
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    cacher = DataCacher(db_path='data/historical_data.db')
    df = cacher.fetch_and_cache_data(symbol, timeframe, start_dt, end_dt)
    cacher.close()
    if df.empty:
        print(f"Could not fetch data for {symbol}. Skipping.")
        return
    df.reset_index(inplace=True)
    print(f"Successfully fetched {len(df)} candles for {symbol}.")

    # 2. Fetch and Merge External Data
    print("Fetching external data (Fear & Greed, On-chain)...")
    days_in_data = (end_dt - start_dt).days
    fng_df = get_fear_and_greed_index(limit=days_in_data)
    if not fng_df.empty:
        df['date'] = df['open_time'].dt.date
        fng_df['date'] = fng_df['date'] + timedelta(days=1)
        df = pd.merge(df, fng_df, on='date', how='left')
        df['fng_value'] = df['fng_value'].ffill()
        df.drop(columns=['date'], inplace=True)

    # 3. Create Features & Labels
    print("Creating features and labels...")
    features_df = create_features(df)
    labeled_df = create_labels(features_df)

    # 4. Clean Data
    missing_ratios = labeled_df.isnull().sum() / len(labeled_df)
    cols_to_drop_missing = missing_ratios[missing_ratios > 0.5].index
    if not cols_to_drop_missing.empty:
        print(f"Dropping columns with >50% missing values: {cols_to_drop_missing.tolist()}")
        labeled_df.drop(columns=cols_to_drop_missing, inplace=True)

    labeled_df.dropna(inplace=True)
    print(f"Data shape after cleaning NaNs: {labeled_df.shape}")
    if labeled_df.empty:
        print(f"Not enough data for {symbol} after cleaning. Skipping.")
        return

    # 5. Define Features (X) and Target (y)
    X = labeled_df.drop(columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'target'])
    y = labeled_df['target'].copy()
    y_mapped = y.map({1: 1, -1: 0})
    X.columns = [str(col) for col in X.columns]

    # 6. Train/Test Split
    train_size = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y_mapped.iloc[:train_size], y_mapped.iloc[train_size:]

    # 7. Pre-training Checks and Scaling
    print(f"Training data shape: {X_train.shape}")
    if len(X_train) < 1000:
        print(f"WARNING: Training data has only {len(X_train)} samples, which is less than 1000. Skipping model training for {symbol}.")
        return

    print("\n--- Scaling Features (StandardScaler) ---")
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    print("Features scaled successfully.")

    # 8. Feature Selection
    print("\n--- Feature Selection ---")
    baseline_model = lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True)
    baseline_model.fit(X_train, y_train)
    feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': baseline_model.feature_importances_}).set_index('feature')
    print("Top 15 Most Important Features (before selection):\n", feature_importances.sort_values('importance', ascending=False).head(15))

    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    if X_train.empty:
        print(f"ERROR: No features remaining after correlation filtering for {symbol}. Skipping training.")
        return

    print("--- Feature Selection Complete ---\n")

    # 9. Model Training
    print("--- Model Training (default parameters) ---")
    model = lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True)
    model.fit(X_train, y_train)

    # 10. Evaluation
    print(f"--- Final Evaluation for {symbol} ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Down (-1)', 'Up (1)']))

    # 11. Save Model and Features
    sanitized_symbol = sanitize_symbol(symbol)
    model_file = os.path.join(MODEL_DIR, f"lgbm_classifier_{sanitized_symbol}.joblib")
    features_file = os.path.join(MODEL_DIR, f"features_{sanitized_symbol}.json")
    print(f"Saving model for {symbol} to {model_file}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, model_file)
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{sanitized_symbol}.joblib")) # Save the scaler
    feature_list = list(X_train.columns) # Save final features after selection
    with open(features_file, 'w') as f:
        json.dump(feature_list, f)
    print(f"Feature list for {symbol} saved to {features_file}")
    print(f"--- Training for {symbol} Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train price prediction models for one or more symbols.")
    end_date_default = datetime.now().strftime('%Y-%m-%d')
    start_date_default = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    parser.add_argument("--symbols", nargs='+', default=["BTC/USDT"], help="List of trading symbols to train on (e.g., 'BTC/USDT' 'ETH/USDT').")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe for candles.")
    parser.add_argument("--start_date", type=str, default=start_date_default, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default=end_date_default, help="End date (YYYY-MM-DD).")
    args = parser.parse_args()
    for symbol in args.symbols:
        train_model(symbol, args.timeframe, args.start_date, args.end_date)
