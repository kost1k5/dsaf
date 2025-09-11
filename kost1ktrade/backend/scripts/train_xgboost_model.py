"""
XGBoost Model Trainer

This script is a standalone tool to train a classification model using the XGBoost algorithm.
It mirrors the main `train_model.py` script but provides an alternative model for comparison.

The script trains a model and saves the artifact to `src/ml/models/`, but it does NOT
automatically update the live prediction service to use this new model. The `predictor.py`
service is currently hardcoded to use the LightGBM model.

Workflow:
1. Run this script to train an XGBoost model for a symbol.
2. The model artifact (`xgb_classifier_SYMBOL.json`) will be saved.
3. To use this model for live predictions, `src/ml/predictor.py` would need to be modified
   to load and use this model type instead of the LightGBM one.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import joblib
import os
import sys
import argparse
from datetime import datetime, timedelta
import json
import optuna
import shap
from sklearn.metrics import f1_score

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

def optimize_hyperparameters_xgb(X_train, y_train):
    """
    Performs hyperparameter optimization for XGBoost using Optuna.
    """
    def objective(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'use_label_encoder': False,
            'booster': 'gbtree',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': 42,
        }

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_index, val_index in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model = xgb.XGBClassifier(**param)
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            scores.append(f1_score(y_val_fold, preds, zero_division=0.0))

        return -1.0 * np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50) # n_trials can be adjusted

    print("Best trial for XGBoost:")
    trial = study.best_trial
    print(f"  Value (Negative F1): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params

def train_xgboost_model(symbol: str, timeframe: str, start_date: str, end_date: str):
    """
    Main function to train an XGBoost model. Mirrors the LightGBM training script.
    """
    print(f"\n--- Starting XGBoost Model Training for {symbol} ---")

    # Construct the full symbol name required by the data cacher
    full_symbol = f"{symbol}/USDT:USDT"

    # Steps 1-5: Data Fetching, Feature Creation, and Feature Selection
    # This part is identical to the LightGBM script to ensure a fair comparison.

    # 1. Fetch Data
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    cacher = DataCacher(db_path='data/historical_data.db')
    df = cacher.fetch_and_cache_data(full_symbol, timeframe, start_dt, end_dt)
    cacher.close()
    if df.empty: return

    # 2. External Data
    days_in_data = (end_dt - start_dt).days
    fng_df = get_fear_and_greed_index(limit=days_in_data)
    onchain_df = get_onchain_metrics(days_back=days_in_data)
    df['date'] = df['open_time'].dt.date
    if not fng_df.empty:
        fng_df['date'] = fng_df['date'] + timedelta(days=1)
        df = pd.merge(df, fng_df, on='date', how='left').fillna(method='ffill')
    if not onchain_df.empty:
        onchain_df['date'] = onchain_df['date'] + timedelta(days=1)
        df = pd.merge(df, onchain_df, on='date', how='left').fillna(method='ffill')
    df.drop(columns=['date'], inplace=True)

    # 3. Prepare Data
    features_df = create_features(df.reset_index())
    features_df.ffill(inplace=True)
    features_df.dropna(inplace=True)
    labeled_df = create_labels(features_df)

    missing_ratios = labeled_df.isnull().sum() / len(labeled_df)
    cols_to_drop = missing_ratios[missing_ratios > 0.5].index
    labeled_df.drop(columns=cols_to_drop, inplace=True)
    labeled_df.dropna(inplace=True)
    if labeled_df.empty: return

    X = labeled_df.drop(columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'target'])
    y = labeled_df['target'].copy()
    y_mapped = y.copy()
    y_mapped[y_mapped == -1] = 0
    X.columns = [str(col) for col in X.columns]

    # 4. Feature Selection (using a baseline LGBM for speed and consistency with the other script)
    baseline_model = lgb.LGBMClassifier(objective='binary', random_state=42)
    baseline_model.fit(X, y_mapped)
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': baseline_model.feature_importances_}).set_index('feature')
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop_corr = set()
    for column in upper_tri.columns:
        highly_correlated_with = upper_tri[column][upper_tri[column] > 0.9].index.tolist()
        if highly_correlated_with:
            for correlated_feature in highly_correlated_with:
                if feature_importances.loc[column, 'importance'] >= feature_importances.loc[correlated_feature, 'importance']:
                    cols_to_drop_corr.add(correlated_feature)
                else:
                    cols_to_drop_corr.add(column)
    X = X.drop(columns=list(cols_to_drop_corr))

    # 5. Train/Test Split
    train_size = int(len(X) * 0.9)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_mapped[:train_size], y_mapped[train_size:]

    # 6. Model Training with XGBoost
    print("--- Hyperparameter Tuning for XGBoost (Optuna) ---")
    best_params = optimize_hyperparameters_xgb(X_train, y_train)

    print("\n--- Training Final XGBoost Model with Best Parameters ---")
    best_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42, **best_params)
    best_model.fit(X_train, y_train)

    # 7. SHAP Analysis for XGBoost
    print("\n--- SHAP Analysis for XGBoost ---")
    explainer = shap.Explainer(best_model) # shap.Explainer is generic and works for XGB
    shap_values = explainer(X_test)

    # For binary classification, we can get mean abs shap values for the positive class
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_importance_df = pd.DataFrame({'feature': X_train.columns, 'mean_abs_shap_value': mean_abs_shap})
    shap_importance_df = shap_importance_df.sort_values('mean_abs_shap_value', ascending=False)
    print("Top 15 Features by Mean Absolute SHAP Value (XGBoost):")
    print(shap_importance_df.head(15))

    # 8. Final Evaluation
    print(f"\n--- Final Evaluation for XGBoost on {symbol} ---")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Down (-1)', 'Up (1)']))

    # 9. Save Model and Features
    sanitized_symbol = sanitize_symbol(symbol)
    model_file = os.path.join(MODEL_DIR, f"xgb_classifier_{sanitized_symbol}.json")
    features_file = os.path.join(MODEL_DIR, f"features_xgb_{sanitized_symbol}.json")

    print(f"Saving XGBoost model for {symbol} to {model_file}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model.save_model(model_file)

    feature_list = list(X.columns)
    with open(features_file, 'w') as f:
        json.dump(feature_list, f)

    print(f"Feature list for XGBoost model saved to {features_file}")
    print(f"--- XGBoost Training for {symbol} Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost prediction models.")
    end_date_default = datetime.now().strftime('%Y-%m-%d')
    start_date_default = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    parser.add_argument("--symbols", nargs='+', default=["BTC/USDT"], help="List of trading symbols to train on.")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe for candles.")
    parser.add_argument("--start_date", type=str, default=start_date_default, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default=end_date_default, help="End date (YYYY-MM-DD).")
    args = parser.parse_args()

    for symbol in args.symbols:
        train_xgboost_model(symbol, args.timeframe, args.start_date, args.end_date)
