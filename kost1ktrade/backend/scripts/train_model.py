import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report
import joblib
import os
import sys
import argparse
from datetime import datetime, timedelta
import json
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Add the project root to the python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collector.data_cacher import DataCacher
from src.ml.feature_generator import create_features, create_labels

# --- Configuration ---
MODEL_DIR = "src/ml/models"
MODEL_FILE = os.path.join(MODEL_DIR, "lgbm_classifier.joblib")
FEATURES_FILE = os.path.join(MODEL_DIR, "features.json")

def train_model(symbol: str, timeframe: str, start_date: str, end_date: str):
    """
    Fetches data, prepares it, tunes and trains a LightGBM model, and saves it.
    """
    print(f"--- Starting Model Training for {symbol} ({start_date} to {end_date}) ---")

    # 1. Fetch Data
    print("Fetching historical data via cacher...")
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    cacher = DataCacher(db_path='data/historical_data.db')
    df = cacher.fetch_and_cache_data(symbol, timeframe, start_dt, end_dt)
    cacher.close()

    if df.empty:
        print("Could not fetch data. Aborting.")
        return

    df.reset_index(inplace=True)
    print(f"Successfully fetched {len(df)} candles.")

    # 2. Prepare Data
    print("Creating features and labels...")
    features_df = create_features(df)
    labeled_df = create_labels(features_df)

    if labeled_df.empty:
        print("Not enough data to create labels. Aborting.")
        return

    # 3. Clean and Split Data
    # First, drop columns that are mostly NaN, which can happen with some indicators
    # on certain datasets. This is more robust than failing completely.
    missing_ratios = labeled_df.isnull().sum() / len(labeled_df)
    cols_to_drop = missing_ratios[missing_ratios > 0.5].index
    if not cols_to_drop.empty:
        print(f"Dropping columns with >50% missing values: {cols_to_drop.tolist()}")
        labeled_df.drop(columns=cols_to_drop, inplace=True)

    # Now, drop rows with any remaining NaNs. This handles the warm-up period for
    # indicators and the look-forward period for labels.
    labeled_df.dropna(inplace=True)
    print(f"Data shape after cleaning NaNs: {labeled_df.shape}")

    if labeled_df.empty:
        print("Not enough data after cleaning NaNs. Aborting.")
        return

    X = labeled_df.drop(columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'target'])
    y = labeled_df['target']
    X.columns = [str(col) for col in X.columns]

    # Hold out the last 10% of data for a final, unbiased evaluation
    train_size = int(len(X) * 0.9)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Full data shape: {X.shape}")
    print(f"Training/tuning data shape: {X_train.shape}")
    print(f"Final hold-out test shape: {X_test.shape}")

    # 4. Hyperparameter Tuning with TimeSeriesSplit
    print("\n--- Hyperparameter Tuning using RandomizedSearchCV ---")

    # Define parameter distribution for Randomized Search
    param_dist = {
        'n_estimators': sp_randint(100, 500),
        'max_depth': sp_randint(3, 10),
        'learning_rate': sp_uniform(0.01, 0.2),
        'num_leaves': sp_randint(20, 60),
        'feature_fraction': sp_uniform(0.6, 0.4),
        'bagging_fraction': sp_uniform(0.6, 0.4),
        'bagging_freq': sp_randint(1, 7)
    }

    lgbm = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, verbose=-1)

    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Setup RandomizedSearchCV
    n_iter_search = 20 # Number of parameter settings that are sampled
    random_search = RandomizedSearchCV(
        lgbm,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        cv=tscv,
        scoring='f1_weighted',
        random_state=42,
        n_jobs=-1 # Use all available cores
    )

    print("Starting hyperparameter search...")
    random_search.fit(X_train, y_train)

    print("\nBest parameters found: ", random_search.best_params_)
    best_model = random_search.best_estimator_

    # 5. Evaluate Best Model on Hold-out Test Set
    print("\n--- Final Model Evaluation on Hold-out Test Set ---")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Down (-1)', 'Sideways (0)', 'Up (1)']))

    # 6. Save Model and Features
    print("Saving best model and feature list...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_FILE)

    feature_list = list(X.columns)
    with open(FEATURES_FILE, 'w') as f:
        json.dump(feature_list, f)

    print(f"\nModel saved to: {MODEL_FILE}")
    print(f"Feature list saved to: {FEATURES_FILE}")
    print("--- Training Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a price prediction model.")
    end_date_default = datetime.now().strftime('%Y-%m-%d')
    start_date_default = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol to train on.")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe for candles.")
    parser.add_argument("--start_date", type=str, default=start_date_default, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default=end_date_default, help="End date (YYYY-MM-DD).")
    args = parser.parse_args()
    train_model(args.symbol, args.timeframe, args.start_date, args.end_date)
