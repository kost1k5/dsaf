import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_financial_metrics(
    predictions: pd.DataFrame,
    threshold: float,
    initial_capital: float = 10000.0,
    risk_per_trade: float = 0.01,
    commission_rate: float = 0.0005, # 0.05%
    slippage_rate: float = 0.0005,   # 0.05%
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0
):
    """
    Simulates trades with fixed fractional risk management and calculates financial metrics.
    """
    capital = initial_capital
    equity_curve = [initial_capital]
    trades_list = []

    # Filter for potential trade entries
    trade_signals = predictions[predictions['y_pred_proba'] > threshold]

    if len(trade_signals) == 0:
        return {
            "sharpe_ratio": 0, "profit_factor": 0, "max_drawdown": 0,
            "win_rate": 0, "total_trades": 0, "final_capital": initial_capital
        }

    for _, trade in trade_signals.iterrows():
        # Prevent taking new trades if capital is depleted
        if capital <= 0:
            break

        # --- Position Sizing ---
        risk_in_money = capital * risk_per_trade
        atr_at_entry = trade['atr']
        stop_loss_distance_usd = sl_atr_mult * atr_at_entry

        if stop_loss_distance_usd == 0:
            continue # Avoid division by zero

        position_size_asset = risk_in_money / stop_loss_distance_usd
        position_value_usd = position_size_asset * trade['close']

        # --- Cost Calculation ---
        entry_commission = position_value_usd * commission_rate
        slippage_cost = position_value_usd * slippage_rate

        # --- PnL Calculation ---
        if trade['y_true'] == 1: # Win (Take Profit)
            pnl = position_size_asset * (tp_atr_mult * atr_at_entry)
        else: # Loss (Stop Loss or Time Barrier)
            pnl = -position_size_asset * (sl_atr_mult * atr_at_entry)

        # --- Net PnL and Capital Update ---
        exit_commission = (position_value_usd + pnl) * commission_rate
        net_pnl = pnl - entry_commission - exit_commission - slippage_cost

        capital += net_pnl
        equity_curve.append(capital)
        trades_list.append({
            'net_pnl': net_pnl,
            'win': trade['y_true'] == 1
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
    except FileNotFoundError:
        print(f"Error: OOS predictions file not found at {predictions_path}.")
        return

    # 2. Find the optimal probability threshold
    print("Finding optimal probability threshold by maximizing Sharpe Ratio...")
    thresholds = np.arange(0.5, 1.0, 0.01)
    results = []
    for t in thresholds:
        metrics = calculate_financial_metrics(predictions_df, t)
        results.append({'threshold': t, **metrics})

    results_df = pd.DataFrame(results)
    best_threshold_row = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    best_threshold = best_threshold_row['threshold']

    print(f"Optimal threshold found: {best_threshold:.2f}")

    # 3. Evaluate using the best threshold
    print("\n--- Final Evaluation Report ---")

    # Get binary predictions based on the best threshold
    y_pred = (predictions_df['y_pred_proba'] > best_threshold).astype(int)
    y_true = predictions_df['y_true']

    # ML Metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Financial Metrics from the best threshold
    final_metrics = best_threshold_row.to_dict()

    # --- Generate Report ---
    report = f"Evaluation Report for {asset} ({timeframe})\n"
    report += "="*40 + "\n"
    report += f"Optimal Probability Threshold: {final_metrics['threshold']:.2f}\n"
    report += "-"*40 + "\n"
    report += "Financial Metrics (at optimal threshold):\n"
    report += f"  - Sharpe Ratio: {final_metrics['sharpe_ratio']:.4f}\n"
    report += f"  - Profit Factor: {final_metrics['profit_factor']:.4f}\n"
    report += f"  - Max Drawdown: {final_metrics['max_drawdown']:.2%}\n"
    report += f"  - Win Rate: {final_metrics['win_rate']:.2%}\n"
    report += f"  - Total Trades: {int(final_metrics['total_trades'])}\n"
    report += f"  - Final Capital: ${final_metrics.get('final_capital', 0):,.2f}\n"
    report += "-"*40 + "\n"
    report += "ML Classification Metrics (at optimal threshold):\n"
    report += f"  - Precision: {precision:.4f}\n"
    report += f"  - Recall: {recall:.4f}\n"
    report += f"  - F1-Score: {f1:.4f}\n"
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
