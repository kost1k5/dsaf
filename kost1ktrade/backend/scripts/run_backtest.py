import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import argparse
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_sortino_for_optuna(predictions: pd.DataFrame) -> float:
    """
    A simplified backtest to calculate Sortino ratio for Optuna.
    Penalizes only for downside volatility.
    """
    # For multiclass, we need a strategy to turn predictions into trades.
    # A simple strategy: Buy on proba_buy > threshold, Sell on proba_sell > threshold.
    # We'll use a fixed threshold for optimization.
    threshold = 0.6 # A bit higher to be more selective
    initial_capital = 10000.0
    risk_per_trade = 0.01
    tp_atr_mult = 2.0
    sl_atr_mult = 1.0

    capital = initial_capital
    equity_curve = [initial_capital]

    for _, trade in predictions.iterrows():
        if capital <= 0: break

        position_size = 0
        pnl = 0

        # Decide trade direction
        if trade['proba_buy'] > threshold:
            position_size = (capital * risk_per_trade) / (trade['atr'] * sl_atr_mult)
            # Original y_true: -1 (Sell), 0 (Hold), 1 (Buy). Mapped y_true: 0, 1, 2.
            # We are buying, so we win if original y_true was 1 (mapped to 2).
            if trade['y_true'] == 2:
                pnl = position_size * (trade['atr'] * tp_atr_mult)
            else:
                pnl = -position_size * (trade['atr'] * sl_atr_mult)
        elif trade['proba_sell'] > threshold:
            position_size = (capital * risk_per_trade) / (trade['atr'] * sl_atr_mult)
            # We are selling, so we win if original y_true was -1 (mapped to 0).
            # A win for a short is hitting the lower barrier (TP), defined by sl_atr_mult.
            if trade['y_true'] == 0:
                pnl = position_size * (trade['atr'] * sl_atr_mult)
            else: # A loss for a short is hitting the upper barrier (SL), defined by tp_atr_mult.
                pnl = -position_size * (trade['atr'] * sl_atr_mult)

        if position_size > 0:
            capital += pnl
            equity_curve.append(capital)

    if len(equity_curve) < 10: return -1.0 # Not enough trades

    equity_ser = pd.Series(equity_curve)
    returns = equity_ser.pct_change().dropna()

    if returns.empty: return -1.0

    # Calculate Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()

    if downside_std > 0:
        sortino_ratio = returns.mean() / downside_std
        return sortino_ratio
    elif returns.mean() > 0:
        return 100.0 # Great performance, no downside
    else:
        return -1.0


def objective(trial, X, y, metadata, metric: str):
    """
    Objective function for Optuna hyperparameter tuning, optimizing for a selected metric.
    """
    # Define the hyperparameter search space
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 50.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 50.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }

    # Create a nested validation set from the training data
    split_point = int(len(X) * 0.75)
    X_train_inner, X_val_inner = X.iloc[:split_point], X.iloc[split_point:]
    y_train_inner, y_val_inner = y.iloc[:split_point], y.iloc[split_point:]
    metadata_val_inner = metadata.loc[X_val_inner.index]

    if len(X_val_inner) == 0:
        return 0.0

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_inner, y_train_inner)

    y_pred_proba = model.predict_proba(X_val_inner)

    if metric == 'f1':
        y_pred_class = np.argmax(y_pred_proba, axis=1)
        return f1_score(y_val_inner, y_pred_class, average='weighted', zero_division=0.0)
    elif metric == 'sortino':
        # Combine predictions with metadata for evaluation
        validation_results = pd.DataFrame({
            'y_true': y_val_inner.values,
            'proba_sell': y_pred_proba[:, 0],
            'proba_hold': y_pred_proba[:, 1],
            'proba_buy': y_pred_proba[:, 2],
            'close': metadata_val_inner['close'].values,
            'atr': metadata_val_inner['atr'].values
        }, index=X_val_inner.index)
        return calculate_sortino_for_optuna(validation_results)
    else:
        raise ValueError(f"Unsupported metric for optimization: {metric}")


def run_walk_forward_validation(X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame, metric: str, n_splits: int = 5):
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
        print(f"Running Optuna hyperparameter search (optimizing for {metric.upper()})...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, metadata_train, metric), n_trials=50)

        best_params = study.best_params
        best_params['objective'] = 'multiclass'
        best_params['num_class'] = 3

        print(f"Best params for this fold: {best_params}")

        # --- Final Model Training for this Fold ---
        print("Training final model for this fold with best params...")
        base_model = lgb.LGBMClassifier(random_state=42, **best_params)

        # Calibrate the model
        print("Calibrating model probabilities with CalibratedClassifierCV...")
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)

        # --- Prediction on Out-of-Sample (OOS) Data ---
        y_pred_proba = calibrated_model.predict_proba(X_test)

        # Combine results into a dataframe
        results_df = pd.DataFrame({
            'timestamp': X_test.index,
            'y_true': y_test.values,
            'proba_sell': y_pred_proba[:, 0],
            'proba_hold': y_pred_proba[:, 1],
            'proba_buy': y_pred_proba[:, 2],
            'close': metadata_test['close'].values,
            'atr': metadata_test['atr'].values
        })
        out_of_sample_preds.append(results_df)

    print("\nWalk-Forward Validation complete.")
    return pd.concat(out_of_sample_preds)


def main(asset: str, timeframe: str, metric: str):
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
    y = df['label'] + 1
    metadata = df[['close', 'ATRr_14']].rename(columns={'ATRr_14': 'atr'})

    # 4. Run Walk-Forward Validation
    oos_predictions = run_walk_forward_validation(X, y, metadata, metric, n_splits=5)

    # 5. Save the out-of-sample predictions
    output_path = os.path.join(RESULTS_DIR, f'{asset}_{timeframe}_oos_predictions.parquet')
    oos_predictions.to_parquet(output_path, index=False)
    print(f"\nOut-of-sample predictions saved to: {output_path}")
    print(oos_predictions.head())

    # (A) Run the new detailed backtest for logging and dynamic sizing
    run_detailed_backtest(oos_predictions, asset, timeframe)


def run_detailed_backtest(predictions: pd.DataFrame, asset: str, timeframe: str, initial_capital=10000.0):
    """
    Runs a detailed backtest simulation based on model predictions,
    implements dynamic position sizing, and logs the last 100 trades.
    """
    print("\n--- Running Detailed Backtest Simulation ---")
    # FIX: Log to a temporary, asset-specific file to avoid parallel write conflicts.
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f'backtest_log_{asset}_{timeframe}.txt')

    capital = initial_capital
    equity_curve = [initial_capital]
    trades = []

    # These would typically come from config, but are fixed here for simplicity
    # as per the triple-barrier method used in labeling.
    tp_atr_mult = 1.5
    sl_atr_mult = 1.0

    # More realistic risk percentages
    confidence_threshold = 0.6
    high_confidence_threshold = 0.8
    high_risk_pct = 0.02 # 2% for high confidence
    med_risk_pct = 0.01 # 1% for medium confidence
    low_risk_pct = 0.005 # 0.5% for low confidence

    for _, row in predictions.iterrows():
        if capital <= 0:
            print("  - Backtest ended: Capital reached zero.")
            break

        confidence = 0
        decision = "HOLD"
        pnl = 0

        # Determine trade direction based on highest probability
        if row['proba_buy'] > row['proba_sell'] and row['proba_buy'] > row['proba_hold']:
            confidence = row['proba_buy']
            decision = "BUY"
        elif row['proba_sell'] > row['proba_buy'] and row['proba_sell'] > row['proba_hold']:
            confidence = row['proba_sell']
            decision = "SELL"

        # Apply dynamic position sizing based on confidence
        if confidence > confidence_threshold:
            if confidence > high_confidence_threshold:
                risk_percentage = high_risk_pct
            elif confidence > 0.7:
                risk_percentage = med_risk_pct
            else:
                risk_percentage = low_risk_pct

            # FIX: The core logical error was here. PnL is calculated based on a fixed risk amount,
            # not a compounding capital figure that includes the current trade's hypothetical outcome.
            # This simulates risking a percentage of the *current* capital at the time of the trade decision.
            amount_to_risk = capital * risk_percentage
            entry_price = row['close']
            atr_at_trade = row['atr']

            if atr_at_trade is None or atr_at_trade == 0:
                continue # Cannot calculate position size if ATR is zero, skip trade

            # Determine PnL based on whether the trade was correct (y_true matches the barrier hit)
            if decision == "BUY":
                # y_true: 0 (Sell), 1 (Hold), 2 (Buy). We win if y_true is 2.
                if row['y_true'] == 2:
                    pnl = amount_to_risk * tp_atr_mult # Simplified Reward
                else:
                    pnl = -amount_to_risk # Loss
            elif decision == "SELL":
                # y_true: 0 (Sell), 1 (Hold), 2 (Buy). We win if y_true is 0.
                if row['y_true'] == 0:
                    pnl = amount_to_risk * tp_atr_mult # Simplified Reward
                else:
                    pnl = -amount_to_risk # Loss

            # This is a more realistic position size for logging purposes
            stop_loss_price_distance = atr_at_trade * sl_atr_mult
            position_size_asset = amount_to_risk / stop_loss_price_distance
            position_size_usd = position_size_asset * entry_price

            sl_price = entry_price - stop_loss_price_distance if decision == "BUY" else entry_price + stop_loss_price_distance
            tp_price = entry_price + (atr_at_trade * tp_atr_mult) if decision == "BUY" else entry_price - (atr_at_trade * tp_atr_mult)


            capital += pnl
            equity_curve.append(capital)

            # Log the trade
            trades.append({
                "entry_time": row['timestamp'],
                "exit_time": row['timestamp'] + pd.Timedelta(hours=1), # Simplified exit time
                "asset": asset,
                "timeframe": timeframe,
                "decision": decision,
                "confidence": f"{confidence:.2%}",
                "entry_price": f"{entry_price:.4f}",
                "take_profit": f"{tp_price:.4f}",
                "stop_loss": f"{sl_price:.4f}",
                "position_size_usd": f"{position_size_usd:.2f}",
                "pnl_usd": f"{pnl:.2f}",
                "capital_after_trade": f"{capital:.2f}"
            })

    print(f"  - Backtest complete. Final Capital: ${capital:.2f}")
    print(f"  - Total trades taken: {len(trades)}")

    # Write the last 100 trades to the log file
    if trades:
        print(f"  - Writing last 100 trades to {log_path}")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"""--- Trade Log for {asset} ({timeframe}) ---
""")
            f.write(f"""--- Final Capital: ${capital:.2f} ---

""")

            log_trades = trades[-100:]
            for i, trade in enumerate(log_trades):
                f.write(f"Trade #{len(trades) - len(log_trades) + i + 1}\n")
                for key, value in trade.items():
                    f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
                f.write("-" * 30 + "\n")
    else:
        print("  - No trades were taken, log file not written.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Walk-Forward Validation Orchestrator")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to process.")
    parser.add_argument("--timeframe", type=str, default="1h", help="The OHLCV timeframe to use.")
    parser.add_argument(
        "--metric",
        type=str,
        default="sortino", # (E) Default to financial metric for optimization
        choices=['f1', 'sortino'],
        help="The metric to optimize for in Optuna ('f1' or 'sortino'). Default is 'sortino'."
    )
    args = parser.parse_args()

    main(asset=args.asset, timeframe=args.timeframe, metric=args.metric)
