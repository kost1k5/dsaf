import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import sys

# Add the project root to the python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collector.collector import DataCollector
from src.ml.feature_generator import create_features, create_labels

# --- Configuration ---
MODEL_DIR = "src/ml/models"
MODEL_FILE = os.path.join(MODEL_DIR, "lgbm_classifier.joblib")
FEATURES_FILE = os.path.join(MODEL_DIR, "features.json")

def train_model():
    """
    Fetches data, prepares it, trains a LightGBM model, and saves it.
    """
    print("--- Starting Model Training ---")

    # 1. Fetch Data
    print("Fetching historical data...")
    collector = DataCollector(exchange_id='okx')

    # HARDCODED SYMBOL as a final attempt to bypass configuration loading issues.
    target_symbol = "BTC/USDT"
    print(f"Fetching data for symbol: {target_symbol}")

    candles_list = collector.fetch_candles(symbol=target_symbol, timeframe="1h", limit=5000)
    if not candles_list:
        print(f"Could not fetch data for {target_symbol}. Aborting.")
        return

    df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    print(f"Fetched {len(df)} candles.")

    # 2. Prepare Data
    print("Creating features and labels...")
    features_df = create_features(df)
    labeled_df = create_labels(features_df)

    if labeled_df.empty:
        print("Not enough data to create labels. Aborting.")
        return

    # 3. Split Data
    X = labeled_df.drop(columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'target'])
    y = labeled_df['target']

    X.columns = [str(col) for col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # 4. Train Model
    print("Training LightGBM model...")
    lgbm = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)
    lgbm.fit(X_train, y_train)

    # 5. Evaluate Model
    print("\n--- Model Evaluation ---")
    y_pred = lgbm.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Down (-1)', 'Sideways (0)', 'Up (1)']))

    # 6. Save Model and Features
    print("Saving model and feature list...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lgbm, MODEL_FILE)

    feature_list = list(X.columns)
    with open(FEATURES_FILE, 'w') as f:
        import json
        json.dump(feature_list, f)

    print(f"\nModel saved to: {MODEL_FILE}")
    print(f"Feature list saved to: {FEATURES_FILE}")
    print("--- Training Complete ---")

if __name__ == "__main__":
    train_model()
