"""
Hybrid & Rule-Based Backtester

This script runs a vectorized and an event-driven backtest for different trading strategies.
It is designed to be a pure evaluation script. It DOES NOT train or optimize any models.

Supported Strategies:
- 'basic': Confluence Strategy 2 (Rules only, no ADX/DMI filter).
- 'advanced': Confluence Strategy 3 (Rules only, with ADX/DMI filter).
- 'hybrid': Confluence Strategy 4 (Advanced rules + ML confirmation).
"""
import os
import pandas as pd
import numpy as np
import argparse
import joblib
import json
from zoneinfo import ZoneInfo

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.confluence_engine import generate_signals
from src.core.config import settings

def run_backtest_simulation(
    df: pd.DataFrame,
    strategy_type: str,
    initial_capital=10000.0,
    commission_rate=0.0005,
    slippage_pct=0.0005
):
    """
    Runs a detailed, event-driven backtest simulation.
    """
    print(f"\n--- Running Backtest Simulation for Strategy: '{strategy_type.upper()}' ---")

    # --- Load Strategy Parameters from Settings ---
    params = settings.STRATEGY
    ml_params = settings.ML

    sl_atr_mult = params.RISK_SL_ATR_MULT
    tp_atr_mult = params.RISK_TP_ATR_MULT
    confidence_threshold = ml_params.HYBRID_CONFIDENCE_THRESHOLD

    # --- Initialize Backtest State ---
    capital = initial_capital
    equity_curve = [initial_capital]
    trades = []
    in_position = False
    current_trade = {}

    # --- Prepare Data ---
    # 1. Generate signals for the given strategy type
    signal_strategy = 'advanced' if strategy_type == 'hybrid' else strategy_type
    df_sim = generate_signals(df, strategy_type=signal_strategy)

    # --- Main Event Loop ---
    for i in range(len(df_sim)):
        if capital <= 0:
            print("  - Backtest ended: Capital reached zero.")
            break

        row = df_sim.iloc[i]
        current_timestamp = df_sim.index[i]

        # --- Exit Logic ---
        if in_position:
            exit_price, exit_reason, pnl = None, None, 0

            # Check for SL/TP hits on the current bar
            if current_trade['type'] == 'LONG':
                if row['low'] <= current_trade['stop_loss']:
                    exit_price, exit_reason = current_trade['stop_loss'], "Stop Loss"
                elif row['high'] >= current_trade['take_profit']:
                    exit_price, exit_reason = current_trade['take_profit'], "Take Profit"
            elif current_trade['type'] == 'SHORT':
                if row['high'] >= current_trade['stop_loss']:
                    exit_price, exit_reason = current_trade['stop_loss'], "Stop Loss"
                elif row['low'] <= current_trade['take_profit']:
                    exit_price, exit_reason = current_trade['take_profit'], "Take Profit"

            if exit_reason:
                # Calculate PnL
                if current_trade['type'] == 'LONG':
                    pnl = (exit_price - current_trade['entry_price']) * current_trade['size_asset']
                else:
                    pnl = (current_trade['entry_price'] - exit_price) * current_trade['size_asset']

                commission = (current_trade['size_usd'] * commission_rate) * 2 # Entry and Exit
                pnl -= commission
                capital += pnl
                equity_curve.append(capital)

                current_trade.update({
                    "exit_time": current_timestamp,
                    "exit_reason": exit_reason,
                    "exit_price": exit_price,
                    "pnl_usd": pnl,
                    "commission_usd": commission,
                    "capital_after_trade": capital
                })
                trades.append(current_trade)
                in_position = False
                current_trade = {}
                continue

        # --- Entry Logic ---
        if not in_position and row['signal'] != 0:
            enter_trade = False
            if strategy_type in ['basic', 'advanced']:
                enter_trade = True
            elif strategy_type == 'hybrid':
                if 'y_pred_proba' in df_sim.columns and row['y_pred_proba'] > confidence_threshold:
                    enter_trade = True

            if enter_trade:
                entry_price = row['close'] * (1 + slippage_pct if row['signal'] == 1 else 1 - slippage_pct)
                atr_at_trade = row['ATR']
                if pd.isna(atr_at_trade) or atr_at_trade <= 0: continue

                # Position sizing (simplified: risk 1% of capital per trade)
                risk_per_trade_usd = capital * 0.01
                sl_distance_price = atr_at_trade * sl_atr_mult
                size_asset = risk_per_trade_usd / sl_distance_price
                size_usd = size_asset * entry_price

                trade_type = 'LONG' if row['signal'] == 1 else 'SHORT'

                if trade_type == 'LONG':
                    stop_loss = entry_price - sl_distance_price
                    take_profit = entry_price + (atr_at_trade * tp_atr_mult)
                else: # SHORT
                    stop_loss = entry_price + sl_distance_price
                    take_profit = entry_price - (atr_at_trade * tp_atr_mult)

                in_position = True
                current_trade = {
                    "entry_time": current_timestamp,
                    "type": trade_type,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "size_asset": size_asset,
                    "size_usd": size_usd,
                    "confidence": row.get('y_pred_proba', np.nan)
                }

    # --- Performance Metrics ---
    print("\n--- Backtest Results ---")
    total_trades = len(trades)
    if total_trades == 0:
        print("No trades were executed.")
        return

    final_capital = equity_curve[-1]
    total_return_pct = (final_capital / initial_capital - 1) * 100

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl_usd'] > 0]
    losses = trades_df[trades_df['pnl_usd'] <= 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0

    print(f"Strategy: {strategy_type.upper()}")
    print(f"Final Capital: ${final_capital:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    if not wins.empty: print(f"Average Win: ${wins['pnl_usd'].mean():.2f}")
    if not losses.empty: print(f"Average Loss: ${losses['pnl_usd'].mean():.2f}")

    equity_ser = pd.Series(equity_curve, index=pd.to_datetime([t['entry_time'] for t in trades] + [df_sim.index[0]], unit='ms') if trades else [df_sim.index[0]])
    daily_returns = equity_ser.resample('D').last().pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0

    print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}")


def main(asset: str, timeframe: str, strategy: str):
    """
    Main script to orchestrate the backtest.
    """
    # --- Define Paths ---
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

    # 1. Load feature data
    features_path = os.path.join(PROCESSED_DATA_DIR, f'{asset}_{timeframe}_features.parquet')
    try:
        features_df = pd.read_parquet(features_path)
        if 'open_time' in features_df.columns:
            features_df['open_time'] = pd.to_datetime(features_df['open_time'])
            features_df.set_index('open_time', inplace=True)
        print(f"Loaded feature data for {asset} from {features_path}")
    except FileNotFoundError:
        print(f"ERROR: Feature file not found: '{features_path}'. Please run 'process_features.py'.")
        return

    # 2. If hybrid, load predictions and merge
    if strategy == 'hybrid':
        preds_path = os.path.join(RESULTS_DIR, f'{asset}_{timeframe}_oos_predictions.parquet')
        try:
            preds_df = pd.read_parquet(preds_path)
            # The predictions file has 'y_true', 'y_pred_proba' and is indexed by integer.
            # We need to align it with the features_df index.
            # Assuming the labeled data was generated from features_df and not shuffled.
            # The predictions correspond to the filtered, labeled events.
            # A robust join is needed. Let's assume 'open_time' is in preds_df after reset_index.
            if 'open_time' in preds_df.columns:
                 preds_df['open_time'] = pd.to_datetime(preds_df['open_time'])
                 preds_df.set_index('open_time', inplace=True)

            # Join predictions onto the full feature set
            features_df = features_df.join(preds_df[['y_pred_proba']], how='left')
            # Forward-fill probabilities to carry them until the next signal event
            features_df['y_pred_proba'].fillna(method='ffill', inplace=True)
            print("Successfully loaded and merged ML predictions.")
        except FileNotFoundError:
            print(f"ERROR: Predictions file not found for hybrid strategy: '{preds_path}'.")
            print("Please run 'train_xgboost_model.py' first.")
            return

    # 3. Run the simulation
    run_backtest_simulation(features_df, strategy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hybrid and Rule-Based Backtester")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to process.")
    parser.add_argument("--timeframe", type=str, default="4h", help="The OHLCV timeframe to use.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid",
        choices=['basic', 'advanced', 'hybrid'],
        help="The strategy to backtest."
    )
    args = parser.parse_args()

    main(asset=args.asset, timeframe=args.timeframe, strategy=args.strategy)
