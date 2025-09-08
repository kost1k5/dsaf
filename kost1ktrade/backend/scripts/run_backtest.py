import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import argparse
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from zoneinfo import ZoneInfo

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
            'atr': metadata_test['atr'].values,
            'EMA_200': metadata_test['EMA_200'].values
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
        print(f"[DEBUG] Columns in df on load in backtester: {df.columns.tolist()}")
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
    # Add EMA_200 to the metadata so it's available for the backtest
    metadata_cols = ['close', 'ATRr_14', 'EMA_200']
    # Ensure EMA_200 exists before trying to access it
    if 'EMA_200' not in df.columns:
        raise ValueError("EMA_200 not found in the dataset. Please run process_features.py again.")
    metadata = df[metadata_cols].rename(columns={'ATRr_14': 'atr'})


    # 4. Run Walk-Forward Validation
    oos_predictions = run_walk_forward_validation(X, y, metadata, metric, n_splits=5)

    # 5. Save the out-of-sample predictions
    output_path = os.path.join(RESULTS_DIR, f'{asset}_{timeframe}_oos_predictions.parquet')
    oos_predictions.to_parquet(output_path, index=False)
    print(f"\nOut-of-sample predictions saved to: {output_path}")
    print(oos_predictions.head())

    # (A) Run the new detailed backtest for logging and dynamic sizing
    run_detailed_backtest(oos_predictions, df, asset, timeframe)


def run_detailed_backtest(predictions: pd.DataFrame, full_data: pd.DataFrame, asset: str, timeframe: str, initial_capital=10000.0, max_leverage=10.0, commission_rate=0.001):
    """
    Runs a realistic, event-driven backtest simulation with enhanced logic.
    """
    print("\n--- Running Realistic Event-Driven Backtest Simulation (with Enhanced Logic) ---")

    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f'backtest_log_{asset}_{timeframe}.txt')

    capital = initial_capital
    equity_curve = [initial_capital]
    trades = []
    minsk_tz = ZoneInfo("Europe/Minsk")

    # --- Strategy Parameters (MODIFIED) ---
    tp_atr_mult = 2.0  # Increased R:R to 2:1
    sl_atr_mult = 1.0
    confidence_threshold = 0.6
    max_holding_period = 24 # candles
    high_confidence_threshold = 0.8
    high_risk_pct = 0.02
    med_risk_pct = 0.01
    low_risk_pct = 0.005
    loss_streak_threshold = 3 # Circuit breaker after 3 losses
    trading_pause_duration = pd.Timedelta(hours=24)

    # --- Backtest State (MODIFIED) ---
    in_position = False
    current_trade = {}
    consecutive_losses = 0
    trade_disabled_until = None

    # (Fix) Ensure the predictions DataFrame is indexed by timestamp.
    if 'timestamp' in predictions.columns and predictions.index.name != 'timestamp':
        predictions = predictions.set_index('timestamp')

    # (Fix) Merge predictions with full OHLC data for simulation.
    ohlc_data = full_data[['open', 'high', 'low', 'EMA_200']] # Add EMA_200
    simulation_df = predictions.join(ohlc_data, how='inner', lsuffix='_pred')

    # (Fix) Remove duplicate timestamps from the combined DataFrame index.
    simulation_df = simulation_df.loc[~simulation_df.index.duplicated(keep='first')]

    # --- Volatility Filter Prep (NEW) ---
    simulation_df['atr_sma_100'] = simulation_df['atr'].rolling(window=100).mean()

    # --- Main Event Loop ---
    for i in range(len(simulation_df)):
        if capital <= 0:
            print("  - Backtest ended: Capital reached zero.")
            break

        current_timestamp = simulation_df.index[i]
        row = simulation_df.iloc[i] # Use iloc for performance

        # --- Exit Logic ---
        if in_position:
            entry_idx = simulation_df.index.get_loc(current_trade['entry_time'])
            current_idx = i

            # Check if the trade has been open for too long
            if current_idx - entry_idx >= max_holding_period:
                exit_price = row['close']
                exit_reason = "Time Stop"
                pnl = (exit_price - current_trade['entry_price']) * current_trade['position_size_asset']
                if current_trade['decision'] == 'SELL': pnl = -pnl
            else:
                exit_price, exit_reason, pnl = None, None, 0
                if current_trade['decision'] == 'BUY':
                    if row['low'] <= current_trade['stop_loss']:
                        exit_price, exit_reason = current_trade['stop_loss'], "Stop Loss"
                        pnl = -current_trade['actual_amount_risked']
                    elif row['high'] >= current_trade['take_profit']:
                        exit_price, exit_reason = current_trade['take_profit'], "Take Profit"
                        pnl = current_trade['actual_amount_risked'] * current_trade['reward_to_risk_ratio']
                elif current_trade['decision'] == 'SELL':
                    if row['high'] >= current_trade['stop_loss']:
                        exit_price, exit_reason = current_trade['stop_loss'], "Stop Loss"
                        pnl = -current_trade['actual_amount_risked']
                    elif row['low'] <= current_trade['take_profit']:
                        exit_price, exit_reason = current_trade['take_profit'], "Take Profit"
                        pnl = current_trade['actual_amount_risked'] * current_trade['reward_to_risk_ratio']

            # If an exit condition was met, close the trade
            if exit_reason:
                commission = current_trade['position_size_usd'] * commission_rate * 2
                pnl -= commission
                capital += pnl
                equity_curve.append(capital)

                # --- Circuit Breaker Logic (NEW) ---
                if pnl < 0:
                    consecutive_losses += 1
                    if consecutive_losses >= loss_streak_threshold:
                        trade_disabled_until = current_timestamp + trading_pause_duration
                        print(f"  - INFO: Circuit breaker triggered at {current_timestamp}. Trading paused until {trade_disabled_until}.")
                else:
                    consecutive_losses = 0 # Reset on a winning trade

                exit_time_minsk = current_timestamp.tz_localize('UTC').tz_convert(minsk_tz)
                current_trade.update({
                    "exit_time_minsk": exit_time_minsk.strftime('%Y-%m-%d %H:%M:%S %Z'),
                    "exit_reason": exit_reason,
                    "exit_price": f"{exit_price:.4f}",
                    "pnl_usd": f"{pnl:.2f}",
                    "commission_usd": f"{commission:.2f}",
                    "capital_after_trade": f"{capital:.2f}"
                })
                trades.append(current_trade)
                in_position = False
                current_trade = {}
                continue

        # --- Entry Logic (MODIFIED) ---
        if not in_position:
            # --- Filter 1: Circuit Breaker ---
            if trade_disabled_until and current_timestamp < trade_disabled_until:
                continue

            # --- Filter 2: Volatility Filter ---
            atr_sma = row.get('atr_sma_100')
            if pd.notna(atr_sma) and atr_sma > 0:
                if row['atr'] < (atr_sma * 0.5) or row['atr'] > (atr_sma * 3.0):
                    continue

            decision = "HOLD"
            confidence = 0
            if row['proba_buy'] > row['proba_sell'] and row['proba_buy'] > row['proba_hold']:
                confidence, decision = row['proba_buy'], "BUY"
            elif row['proba_sell'] > row['proba_buy'] and row['proba_sell'] > row['proba_hold']:
                confidence, decision = row['proba_sell'], "SELL"

            if confidence > confidence_threshold:
                # --- Filter 3: Trend Filter ---
                ema_200 = row.get('EMA_200')
                if pd.notna(ema_200):
                    if (decision == 'BUY' and row['close'] < ema_200) or \
                       (decision == 'SELL' and row['close'] > ema_200):
                        continue # Trade against the trend is filtered

                if confidence > high_confidence_threshold: risk_percentage = high_risk_pct
                elif confidence > 0.7: risk_percentage = med_risk_pct
                else: risk_percentage = low_risk_pct

                entry_price = row['close']
                atr_at_trade = row['atr']
                if atr_at_trade is None or atr_at_trade <= 0: continue

                sl_distance = atr_at_trade * sl_atr_mult
                tp_distance = atr_at_trade * tp_atr_mult
                position_size_asset = (capital * risk_percentage) / sl_distance
                position_size_usd = position_size_asset * entry_price
                if position_size_usd > capital * max_leverage:
                    position_size_usd = capital * max_leverage
                    position_size_asset = position_size_usd / entry_price
                actual_amount_risked = position_size_asset * sl_distance

                in_position = True
                entry_time_minsk = current_timestamp.tz_localize('UTC').tz_convert(minsk_tz)
                current_trade = {
                    "entry_time": current_timestamp,
                    "entry_time_minsk": entry_time_minsk.strftime('%Y-%m-%d %H:%M:%S %Z'),
                    "asset": asset, "timeframe": timeframe, "decision": decision,
                    "confidence": f"{confidence:.2%}", "risk_percentage": f"{risk_percentage:.3%}",
                    "entry_price": entry_price,
                    "stop_loss": entry_price - sl_distance if decision == 'BUY' else entry_price + sl_distance,
                    "take_profit": entry_price + tp_distance if decision == 'BUY' else entry_price - tp_distance,
                    "position_size_asset": position_size_asset, "position_size_usd": position_size_usd,
                    "actual_amount_risked": actual_amount_risked,
                    "actual_amount_risked_usd": f"{actual_amount_risked:.2f}",
                    "reward_to_risk_ratio": tp_atr_mult / sl_atr_mult
                }

    print(f"  - Backtest complete. Final Capital: ${capital:.2f}")
    print(f"  - Total trades taken: {len(trades)}")

    # Write the last 100 trades to the log file
    if trades:
        print(f"  - Writing last 100 trades to {log_path}")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"--- Trade Log for {asset} ({timeframe}) ---\n")
            f.write(f"--- Final Capital: ${capital:.2f} ---\n\n")

            log_trades = trades[-100:]
            for i, trade in enumerate(log_trades):
                f.write(f"Trade #{len(trades) - len(log_trades) + i + 1}\n")
                # Format for display
                trade_to_log = trade.copy()
                trade_to_log['stop_loss'] = f"{trade['stop_loss']:.4f}"
                trade_to_log['take_profit'] = f"{trade['take_profit']:.4f}"
                del trade_to_log['entry_time']
                del trade_to_log['position_size_asset']
                del trade_to_log['actual_amount_risked']
                del trade_to_log['reward_to_risk_ratio']

                for key, value in trade_to_log.items():
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
