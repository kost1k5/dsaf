import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report
import joblib
import os
import sys
import argparse
from datetime import datetime, timedelta
import json
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from typing import List

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collector.data_cacher import DataCacher
from src.ml.feature_generator import create_features, create_labels

# --- Configuration ---
MODEL_DIR = "src/ml/models"

def sanitize_symbol(symbol: str) -> str:
    """Converts a symbol like 'BTC/USDT' to 'BTC_USDT' for filenames."""
    return symbol.replace('/', '_')

def train_model(symbol: str, timeframe: str, start_date: str, end_date: str):
    """
    Fetches data, prepares it, tunes and trains a LightGBM model, and saves it
    with a symbol-specific name.
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

    # 2. Prepare Data
    print("Creating features and labels...")
    features_df = create_features(df)
    labeled_df = create_labels(features_df)

    # 3. Clean and Split Data
    missing_ratios = labeled_df.isnull().sum() / len(labeled_df)
    cols_to_drop = missing_ratios[missing_ratios > 0.5].index
    if not cols_to_drop.empty:
        print(f"Dropping columns with >50% missing values: {cols_to_drop.tolist()}")
        labeled_df.drop(columns=cols_to_drop, inplace=True)

    labeled_df.dropna(inplace=True)
    print(f"Data shape after cleaning NaNs: {labeled_df.shape}")

    if labeled_df.empty:
        print(f"Not enough data for {symbol} after cleaning. Skipping.")
        return

    X = labeled_df.drop(columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'target'])
    y = labeled_df['target']
    X.columns = [str(col) for col in X.columns]

    train_size = int(len(X) * 0.9)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Training/tuning data shape for {symbol}: {X_train.shape}")

    # 4. Hyperparameter Tuning
    print("--- Hyperparameter Tuning ---")
    param_dist = {
        'n_estimators': sp_randint(100, 500),
        'max_depth': sp_randint(3, 10),
        'learning_rate': sp_uniform(0.01, 0.2),
    }
    lgbm = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, verbose=-1)
    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        lgbm, param_distributions=param_dist, n_iter=15, cv=tscv,
        scoring='f1_weighted', random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    print(f"Best parameters for {symbol}: {random_search.best_params_}")
    best_model = random_search.best_estimator_

    # 5. Final Evaluation
    print(f"--- Final Evaluation for {symbol} ---")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Down (-1)', 'Sideways (0)', 'Up (1)']))

    # 6. Save Model and Features with symbol-specific names
    sanitized_symbol = sanitize_symbol(symbol)
    model_file = os.path.join(MODEL_DIR, f"lgbm_classifier_{sanitized_symbol}.joblib")
    features_file = os.path.join(MODEL_DIR, f"features_{sanitized_symbol}.json")

    print(f"Saving model for {symbol} to {model_file}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, model_file)

    feature_list = list(X.columns)
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
