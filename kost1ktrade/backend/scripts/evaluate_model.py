import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

# Adjust the path to allow imports from the 'src' directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_financial_metrics(predictions: pd.DataFrame, threshold: float):
    """
    Simulates trades based on a probability threshold and calculates financial metrics.
    """
    # Generate trade signals (1 for long, 0 for hold/flat)
    signals = (predictions['y_pred_proba'] > threshold).astype(int)

    # We only act on a signal of 1
    trades = signals[signals == 1]
    if len(trades) == 0:
        return {
            "sharpe_ratio": 0, "profit_factor": 0, "max_drawdown": 1,
            "win_rate": 0, "total_trades": 0
        }

    # Get the outcomes of the trades we took
    trade_outcomes = predictions.loc[trades.index]['y_true']

    # Simplified returns: +2 for a win (TP), -1 for a loss (SL) based on our labeling
    returns = trade_outcomes.apply(lambda x: 2 if x == 1 else -1)

    # --- Sharpe Ratio ---
    # Assuming risk-free rate is 0
    sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

    # --- Win Rate ---
    win_rate = returns[returns > 0].count() / len(returns)

    # --- Profit Factor ---
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # --- Max Drawdown ---
    # This requires a proper equity curve, which is hard with simplified returns.
    # We will calculate a simplified drawdown on the cumulative returns.
    cumulative_returns = returns.cumsum()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / (peak + 1e-6) # Add epsilon to avoid division by zero
    max_drawdown = abs(drawdown.min())

    return {
        "sharpe_ratio": sharpe_ratio,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_trades": len(trades)
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
    report += f"  - Total Trades: {final_metrics['total_trades']}\n"
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
