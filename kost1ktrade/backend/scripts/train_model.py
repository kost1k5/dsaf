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
import optuna
import shap
from sklearn.metrics import f1_score
from statsmodels.tsa.stattools import adfuller

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collector.data_cacher import DataCacher
from src.data_collector.external_data import get_fear_and_greed_index, get_news_sentiment, get_onchain_metrics
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
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
        }

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_index, val_index in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model = lgb.LGBMClassifier(**param)
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            scores.append(f1_score(y_val_fold, preds))

        return -1.0 * np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50) # n_trials can be adjusted

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {-trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params

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

    # 2. Fetch and Merge External Data
    print("Fetching external data (Fear & Greed, News, On-chain)...")
    days_in_data = (end_dt - start_dt).days
    fng_df = get_fear_and_greed_index(limit=days_in_data)
    news_df = get_news_sentiment(days_back=days_in_data)
    onchain_df = get_onchain_metrics(days_back=days_in_data)


    # Merge external data into the main dataframe
    df['date'] = df['open_time'].dt.date

    # To prevent lookahead bias, we shift the daily data by 1 day,
    # so each candle only uses data from the previous day.
    if not fng_df.empty:
        fng_df['date'] = fng_df['date'] + timedelta(days=1)
        df = pd.merge(df, fng_df, on='date', how='left')
        df['fng_value'] = df['fng_value'].fillna(method='ffill')
    if not news_df.empty:
        news_df['date'] = news_df['date'] + timedelta(days=1)
        df = pd.merge(df, news_df, on='date', how='left')
        df['sentiment_score'] = df['sentiment_score'].fillna(0)
    if not onchain_df.empty:
        onchain_df['date'] = onchain_df['date'] + timedelta(days=1)
        df = pd.merge(df, onchain_df, on='date', how='left')
        for col in ['net_exchange_flow', 'sopr', 'mvrv']:
             if col in df.columns:
                df[col] = df[col].fillna(method='ffill')

    df.drop(columns=['date'], inplace=True)


    # 3. Prepare Data
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

    # 4. Verify Stationarity of key features
    print("\n--- Verifying Feature Stationarity (ADF Test) ---")
    def run_adf_test(series, name):
        result = adfuller(series.dropna()) # Drop NaNs for the test
        p_value = result[1]
        if p_value > 0.05:
            print(f"WARNING: Feature '{name}' may be non-stationary (p-value: {p_value:.4f})")
        else:
            print(f"Feature '{name}' appears stationary (p-value: {p_value:.4f})")

    # Test a few representative transformed features
    key_features_to_test = [col for col in labeled_df.columns if 'SMA_50_normalized' in col or 'OBV_pct_change' in col or 'RSI' in col]
    for feature_name in key_features_to_test:
        run_adf_test(labeled_df[feature_name], feature_name)
    print("--- Stationarity Check Complete ---\n")


    if labeled_df.empty:
        print(f"Not enough data for {symbol} after cleaning. Skipping.")
        return

    X = labeled_df.drop(columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'target'])
    y = labeled_df['target'].copy() # Use copy to avoid SettingWithCopyWarning

    # Map labels from {-1, 1} to {0, 1} for LGBM binary classification
    y_mapped = y.copy()
    y_mapped[y_mapped == -1] = 0

    X.columns = [str(col) for col in X.columns]

    # 5. Feature Selection
    print("\n--- Feature Selection ---")
    # First, train a baseline model to get feature importances
    baseline_model = lgb.LGBMClassifier(objective='binary', random_state=42)
    baseline_model.fit(X, y_mapped)

    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': baseline_model.feature_importances_
    }).set_index('feature')

    print("Top 15 Most Important Features (before selection):")
    print(feature_importances.sort_values('importance', ascending=False).head(15))

    # Correlation Analysis and Pruning
    print("\nPruning highly correlated features (threshold > 0.9)...")
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    cols_to_drop = set()
    for column in upper_tri.columns:
        highly_correlated_with = upper_tri[column][upper_tri[column] > 0.9].index.tolist()
        if highly_correlated_with:
            for correlated_feature in highly_correlated_with:
                # Compare importances and decide which one to drop
                if feature_importances.loc[column, 'importance'] >= feature_importances.loc[correlated_feature, 'importance']:
                    cols_to_drop.add(correlated_feature)
                else:
                    cols_to_drop.add(column)

    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} highly correlated features: {list(cols_to_drop)}")
        X = X.drop(columns=list(cols_to_drop))
    else:
        print("No feature pairs found with correlation > 0.9 to prune.")

    print("--- Feature Selection Complete ---\n")

    # 6. Train/Test Split (after feature selection)
    train_size = int(len(X) * 0.9)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_mapped[:train_size], y_mapped[train_size:]

    print(f"Training data shape after feature selection: {X_train.shape}")

    # 7. Model Training & Hyperparameter Tuning
    # The user can uncomment the following block to run Bayesian Optimization with Optuna.
    # The default path (below this block) uses standard parameters for a baseline.
    # ---
    # print("--- Hyperparameter Tuning (Optuna) ---")
    # best_params = optimize_hyperparameters(X_train, y_train)
    # print("\n--- Training Final Model with Best Parameters ---")
    # best_model = lgb.LGBMClassifier(objective='binary', random_state=42, **best_params)
    # best_model.fit(X_train, y_train)
    # ---

    # Active path: Train with default parameters for baseline
    print("--- Model Training (default parameters) ---")
    best_model = lgb.LGBMClassifier(objective='binary', random_state=42)
    best_model.fit(X_train, y_train)

    # SHAP Analysis
    print("\n--- SHAP Analysis ---")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    # We can't plot, so we'll log the mean absolute SHAP values
    # For binary classification, shap_values is a list of two arrays.
    # We're interested in the explanations for the positive class (1).
    shap_sum = np.abs(shap_values[1]).mean(axis=0)

    shap_importance_df = pd.DataFrame([X_train.columns.tolist(), shap_sum.tolist()]).T
    shap_importance_df.columns = ['feature', 'mean_abs_shap_value']
    shap_importance_df = shap_importance_df.sort_values('mean_abs_shap_value', ascending=False)

    print("Top 15 Features by Mean Absolute SHAP Value:")
    print(shap_importance_df.head(15))

    # --- Example of Local SHAP analysis for a single prediction ---
    # This code can be uncommented and used locally for debugging specific predictions.
    #
    # sample_idx = 0 # Index of the sample in the test set to explain
    # shap_values_single = explainer.shap_values(X_test.iloc[sample_idx])
    # # For binary classification, shap_values is a list of two arrays (for class 0 and 1)
    # # We are interested in the explanation for the positive class (Up)
    # shap_values_for_class_1 = shap_values[1]
    #
    # print(f"\n--- Local SHAP Explanation for Test Sample {sample_idx} ---")
    # print("To visualize this, use a force plot in a Jupyter environment:")
    # print("shap.initjs()")
    # print("shap.force_plot(explainer.expected_value[1], shap_values_for_class_1, X_test.iloc[sample_idx])")
    #

    print("--- Feature Analysis Complete ---\n")


    # 6. Final Evaluation
    print(f"--- Final Evaluation for {symbol} ---")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Down (-1)', 'Up (1)']))

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
