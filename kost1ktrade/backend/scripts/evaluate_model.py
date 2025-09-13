import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings

def calculate_financial_metrics(
    predictions: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float
):
    """
    Simulates long and short trades with fixed fractional risk management.
    """
    # Load settings from the config object
    s = settings.EVAL
    capital = s.INITIAL_CAPITAL
    equity_curve = [s.INITIAL_CAPITAL]
    trades_list = []

    for _, row in predictions.iterrows():
        if capital <= 0: break

        trade_type = None
        if row['proba_long'] > buy_threshold:
            trade_type = 'buy'
        elif row['proba_short'] > sell_threshold:
            trade_type = 'sell'

        if trade_type is None:
            continue

        # --- Position Sizing (same for long and short) ---
        risk_in_money = capital * s.RISK_PER_TRADE
        atr_at_entry = row['ATRr_14']
        if atr_at_entry == 0: continue

        stop_loss_distance_price = s.SL_ATR_MULT * atr_at_entry
        position_size_asset = risk_in_money / stop_loss_distance_price
        position_value_usd = position_size_asset * row['close']

        # --- Cost Calculation ---
        entry_commission = position_value_usd * s.COMMISSION_RATE
        slippage_cost = position_value_usd * s.SLIPPAGE_RATE

        # --- PnL Calculation ---
        pnl = 0
        is_win = False
        # y_true is mapped: 0=Sell(-1), 1=Hold(0), 2=Buy(1)
        if trade_type == 'buy':
            if row['y_true'] == 2: # Win
                pnl = position_size_asset * (s.TP_ATR_MULT * atr_at_entry)
                is_win = True
            else: # Loss
                pnl = -position_size_asset * (s.SL_ATR_MULT * atr_at_entry)
        elif trade_type == 'sell':
            # For a short trade, a win means price hits the lower barrier (TP), loss means hitting upper barrier (SL)
            if row['y_true'] == 0: # Win
                pnl = position_size_asset * (s.SL_ATR_MULT * atr_at_entry)
                is_win = True
            else: # Loss
                pnl = -position_size_asset * (s.TP_ATR_MULT * atr_at_entry)

        # --- Net PnL and Capital Update ---
        exit_commission = (position_value_usd + pnl) * s.COMMISSION_RATE
        net_pnl = pnl - entry_commission - exit_commission - slippage_cost

        capital += net_pnl
        equity_curve.append(capital)
        trades_list.append({
            'net_pnl': net_pnl,
            'win': is_win
        })

    # --- Final Metrics Calculation ---
    if not trades_list:
        return {
            "sharpe_ratio": 0, "profit_factor": 0, "max_drawdown": 0,
            "win_rate": 0, "total_trades": 0, "final_capital": initial_capital
        }

    total_trades = len(trades_list)
    wins = sum(1 for t in trades_list if t['win'])
    win_rate = wins / total_trades if total_trades > 0 else 0

    gross_profit = sum(t['net_pnl'] for t in trades_list if t['win'])
    gross_loss = abs(sum(t['net_pnl'] for t in trades_list if not t['win']))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Max Drawdown
    equity_ser = pd.Series(equity_curve)
    peak = equity_ser.cummax()
    drawdown = (equity_ser - peak) / peak
    max_drawdown = abs(drawdown.min())

    # Sharpe Ratio (simplified, non-annualized)
    returns = equity_ser.pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

    return {
        "sharpe_ratio": sharpe_ratio,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "final_capital": capital
    }

def main(asset: str, timeframe: str):
    """
    Main script to evaluate the model's out-of-sample predictions.
    """
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load the OOS predictions
    predictions_path = os.path.join(RESULTS_DIR, f'{asset}_{timeframe}_oos_predictions.parquet')
    try:
        predictions_df = pd.read_parquet(predictions_path)

        # Also load the labeled data to get access to the ATRr_14 feature
        LABELED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled')
        labeled_data_path = os.path.join(LABELED_DATA_DIR, f'{asset}_{timeframe}_labeled.parquet')
        labeled_df = pd.read_parquet(labeled_data_path)

        # Join the ATR and close columns into the predictions dataframe for financial calculations
        # These are needed for the backtesting simulation
        predictions_df = predictions_df.join(labeled_df[['ATRr_14', 'close']])
        predictions_df.dropna(inplace=True) # Drop rows where join might have failed

    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. {e}")
        return

    # 2. Find the optimal probability thresholds via Grid Search
    print("Finding optimal probability thresholds by maximizing Sharpe Ratio...")
    thresholds = np.arange(0.5, 1.0, 0.05) # Coarser grid for speed
    results = []
    for buy_t in thresholds:
        for sell_t in thresholds:
            metrics = calculate_financial_metrics(predictions_df, buy_t, sell_t)
            results.append({'buy_threshold': buy_t, 'sell_threshold': sell_t, **metrics})

    results_df = pd.DataFrame(results)

    # --- (Z) Constraint on Threshold Optimization ---
    min_trades = settings.ML.MIN_TRADES_FOR_EVAL
    realistic_results_df = results_df[results_df['total_trades'] >= min_trades]

    if realistic_results_df.empty:
        print(f"CRITICAL: No threshold combination produced the minimum required {min_trades} trades.")
        # Optional: Print the best of the insufficient-trade results for diagnostics
        if not results_df.empty:
            best_insufficient_row = results_df.loc[results_df['sharpe_ratio'].idxmax()]
            print("Diagnostics: Best result among all combinations (below trade threshold):")
            print(best_insufficient_row)
        return

    # Find the best threshold from the realistic, filtered results
    best_threshold_row = realistic_results_df.loc[realistic_results_df['sharpe_ratio'].idxmax()]
    best_buy_threshold = best_threshold_row['buy_threshold']
    best_sell_threshold = best_threshold_row['sell_threshold']

    print(f"Optimal Buy Threshold: {best_buy_threshold:.2f}")
    print(f"Optimal Sell Threshold: {best_sell_threshold:.2f}")

    # 3. Evaluate using the best thresholds
    print("\n--- Final Evaluation Report ---")

    # Generate class predictions based on the best thresholds
    def get_pred_class(row):
        if row['proba_long'] > best_buy_threshold:
            return 2 # Buy
        elif row['proba_short'] > best_sell_threshold:
            return 0 # Sell
        else:
            # If neither Buy nor Sell threshold is met, predict Hold.
            # We can also use argmax as a fallback, but this is more explicit.
            return 1 # Hold

    y_pred = predictions_df.apply(get_pred_class, axis=1)
    y_true = predictions_df['y_true']

    # ML Metrics
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Financial Metrics from the best threshold
    final_metrics = best_threshold_row.to_dict()

    # --- Generate Report ---
    report = f"Evaluation Report for {asset} ({timeframe})\n"
    report += "="*40 + "\n"
    report += f"Optimal Buy Threshold: {final_metrics['buy_threshold']:.2f}\n"
    report += f"Optimal Sell Threshold: {final_metrics['sell_threshold']:.2f}\n"
    report += "-"*40 + "\n"
    report += "Financial Metrics (at optimal thresholds):\n"
    report += f"  - Sharpe Ratio: {final_metrics['sharpe_ratio']:.4f}\n"
    report += f"  - Profit Factor: {final_metrics['profit_factor']:.4f}\n"
    report += f"  - Max Drawdown: {final_metrics['max_drawdown']:.2%}\n"
    report += f"  - Win Rate: {final_metrics['win_rate']:.2%}\n"
    report += f"  - Total Trades: {int(final_metrics['total_trades'])}\n"
    report += f"  - Final Capital: ${final_metrics.get('final_capital', 0):,.2f}\n"
    report += "-"*40 + "\n"
    report += "ML Classification Metrics (at optimal thresholds):\n"
    report += f"  - Precision (weighted): {precision:.4f}\n"
    report += f"  - Recall (weighted): {recall:.4f}\n"
    report += f"  - F1-Score (weighted): {f1:.4f}\n"
    report += "="*40 + "\n"

    print(report)

    # 4. Save the report
    report_path = os.path.join(REPORTS_DIR, f'{asset}_{timeframe}_evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Evaluation report saved to: {report_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Evaluation Orchestrator")
    parser.add_argument("--asset", type=str, default="BTC", help="The crypto asset to evaluate.")
    parser.add_argument("--timeframe", type=str, default="1h", help="The OHLCV timeframe used.")
    args = parser.parse_args()

    main(asset=args.asset, timeframe=args.timeframe)
