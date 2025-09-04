import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import argparse
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_sharpe_for_optuna(predictions: pd.DataFrame) -> float:
    """
    A simplified backtest simulation to calculate Sharpe ratio for Optuna.
    This is a performance-critical function.
    """
    # Use a fixed threshold for evaluation within Optuna, e.g., 0.5, since we are optimizing params not threshold here.
    # The final threshold will be optimized in the evaluation script.
    threshold = 0.5
    initial_capital = 10000.0
    risk_per_trade = 0.01
    tp_atr_mult = 2.0
    sl_atr_mult = 1.0

    capital = initial_capital
    equity_curve = [initial_capital]

    trade_signals = predictions[predictions['y_pred_proba'] > threshold]

    if len(trade_signals) < 5: # Not enough trades to calculate a meaningful Sharpe
        return -1.0 # Return a poor score

    for _, trade in trade_signals.iterrows():
        if capital <= 0: break
        risk_in_money = capital * risk_per_trade
        stop_loss_distance_usd = sl_atr_mult * trade['atr']
        if stop_loss_distance_usd == 0: continue
        position_size_asset = risk_in_money / stop_loss_distance_usd

        if trade['y_true'] == 1:
            pnl = position_size_asset * (tp_atr_mult * trade['atr'])
        else:
            pnl = -position_size_asset * (sl_atr_mult * trade['atr'])

        capital += pnl
        equity_curve.append(capital)

    equity_ser = pd.Series(equity_curve)
    returns = equity_ser.pct_change().dropna()

    if returns.std() > 0 and len(returns) > 1:
        sharpe_ratio = returns.mean() / returns.std()
        # Simple annualization assumption for hourly data, penalize for fewer trades
        # This helps select models that trade more frequently and are more stable.
        annualized_sharpe = sharpe_ratio * np.sqrt(252 * 24) * (len(trade_signals) / len(predictions))
        return annualized_sharpe
    else:
        return -1.0


def objective(trial, X, y, metadata):
    """
    Objective function for Optuna hyperparameter tuning, optimizing for Sharpe Ratio.
    """
    # Define the hyperparameter search space
    params = {
        'objective': 'binary',
        'metric': 'logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }

    # Create a nested validation set from the training data
    # Use the last 25% of the data for validation
    split_point = int(len(X) * 0.75)
    X_train_inner, X_val_inner = X.iloc[:split_point], X.iloc[split_point:]
    y_train_inner, y_val_inner = y.iloc[:split_point], y.iloc[split_point:]
    metadata_val_inner = metadata.loc[X_val_inner.index]

    if len(X_val_inner) == 0:
        return -1.0 # Cannot evaluate if validation set is empty

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_inner, y_train_inner)

    # Make predictions on the inner validation set
    y_pred_proba = model.predict_proba(X_val_inner)[:, 1]

    # Combine predictions with metadata for evaluation
    validation_results = pd.DataFrame({
        'y_true': y_val_inner.values,
        'y_pred_proba': y_pred_proba,
        'close': metadata_val_inner['close'].values,
        'atr': metadata_val_inner['ATRr_14'].values
    }, index=X_val_inner.index)

    # Calculate Sharpe Ratio
    sharpe = calculate_sharpe_for_optuna(validation_results)
    return sharpe


def run_walk_forward_validation(X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame, n_splits: int = 5):
    """
    Performs walk-forward validation with nested hyperparameter tuning.
    """
    print(f"Starting Walk-Forward Validation with {n_splits} splits...")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    out_of_sample_preds = []

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n--- Processing Fold {i+1}/{n_splits} ---")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        metadata_train, metadata_test = metadata.loc[X_train.index], metadata.loc[X_test.index]

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # --- Nested Hyperparameter Tuning with Optuna ---
        print("Running Optuna hyperparameter search (optimizing for Sharpe Ratio)...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, metadata_train), n_trials=50) # More trials needed for this complex objective

        best_params = study.best_params
        print(f"Best params for this fold: {best_params}")

        # --- Final Model Training for this Fold ---
        print("Training final model for this fold with best params...")
        base_model = lgb.LGBMClassifier(random_state=42, **best_params)

        # Calibrate the model
        print("Calibrating model probabilities with CalibratedClassifierCV...")
        # Using 'isotonic' as it's a non-parametric method that can correct any monotonic distortion.
        # cv=3 is a reasonable default for the inner cross-validation of the calibrator.
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)


        # --- Prediction on Out-of-Sample (OOS) Data ---
        y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1] # Probability of class 1

        fold_results = pd.DataFrame({
            'timestamp': X_test.index,
            'y_true': y_test.values,
            'y_pred_proba': y_pred_proba,
            'close': metadata_test['close'].values,
            'atr': metadata_test['ATRr_14'].values
        })
        out_of_sample_preds.append(fold_results)

    print("\nWalk-Forward Validation complete.")
    return pd.concat(out_of_sample_preds)


def main(asset: str, timeframe: str):
    """
    Main script to run the full backtest for a given asset.
    """
    LABELED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled')
    REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load the labeled dataset
    labeled_path = os.path.join(LABELED_DATA_DIR, f'{asset}_{timeframe}_labeled.parquet')
    try:
        df = pd.read_parquet(labeled_path)
    except FileNotFoundError:
        print(f"Error: Labeled file not found at {labeled_path}.")
        return

    # 2. Load the selected features
    features_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_selected_features.txt')
    try:
        with open(features_path, 'r') as f:
            selected_features = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: Selected features file not found at {features_path}.")
        return

    print(f"Loaded {len(selected_features)} selected features for {asset}.")

    # 3. Prepare data for model
    X = df[selected_features]
    y = df['label']
    metadata = df[['close', 'ATRr_14']]

    # 4. Run Walk-Forward Validation
    oos_predictions = run_walk_forward_validation(X, y, metadata, n_splits=5)

    # 5. Save the out-of-sample predictions
    output_path = os.path.join(RESULTS_DIR, f'{asset}_{timeframe}_oos_predictions.parquet')
    oos_predictions.to_parquet(output_path, index=False)
    print(f"\nOut-of-sample predictions saved to: {output_path}")
    print(oos_predictions.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Walk-Forward Validation Orchestrator")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to process.")
    parser.add_argument("--timeframe", type=str, default="1h", help="The OHLCV timeframe to use.")
    args = parser.parse_args()

    main(asset=args.asset, timeframe=args.timeframe)
