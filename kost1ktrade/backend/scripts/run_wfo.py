"""
Walk-Forward Optimization Runner

This script is a standalone tool to perform Walk-Forward Optimization (WFO) for a given strategy.
It finds potentially optimal parameters for different time windows but does NOT automatically
update the main `strategy_params.json` configuration file.

Workflow:
1. Run this script for a strategy (e.g., `pdm run python scripts/run_wfo.py --strategy sma_crossover`).
2. Analyze the output to identify robust parameters.
3. Manually update the parameters for the chosen strategy in `strategy_params.json`.
"""
import argparse
import pandas as pd
import sys
import os

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization.optimizer import Optimizer
from src.data_collector.collector import DataCollector
from src.core.strategy_loader import get_strategy_class

def main():
    """
    Main function to run the Walk-Forward Optimization.
    """
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization Runner")
    parser.add_argument("--strategy", type=str, required=True, help="The name of the strategy to optimize (e.g., 'sma_crossover').")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="The trading symbol.")
    parser.add_argument("--timeframe", type=str, default="1d", help="The OHLCV timeframe.")
    parser.add_argument("--days", type=int, default=730, help="Total days of historical data to fetch.")
    parser.add_argument("--in_sample_len", type=int, default=180, help="Length of the in-sample (training) period in days.")
    parser.add_argument("--out_of_sample_len", type=int, default=60, help="Length of the out-of-sample (testing) period in days.")
    parser.add_argument("--step_size", type=int, default=60, help="How many days to step forward for the next fold.")
    parser.add_argument("--optimize_for", type=str, default="sharpe_ratio", help="The metric to optimize for.")

    args = parser.parse_args()

    # 1. Get Strategy Class
    try:
        StrategyClass = get_strategy_class(args.strategy)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Get Data
    print(f"Fetching {args.days} days of data for {args.symbol} ({args.timeframe})...")
    collector = DataCollector()
    # PDM requires -- to pass arguments to the script, so we use argparse
    candles_list = collector.fetch_candles(args.symbol, timeframe=args.timeframe, limit=args.days)
    if not candles_list or len(candles_list) < (args.in_sample_len + args.out_of_sample_len):
        print("Error: Not enough data fetched to perform walk-forward analysis.")
        sys.exit(1)

    candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
    candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms', utc=True)
    candles_df.set_index('open_time', inplace=True)
    candles_df.attrs = {'symbol': args.symbol, 'timeframe': args.timeframe}

    # 3. Setup and run optimizer
    optimizer = Optimizer(strategy_class=StrategyClass, data=candles_df)

    # NOTE: Parameter grids need to be defined here for each strategy.
    # This is a simple example for sma_crossover.
    # A more advanced version would load these from a config file.
    param_grids = {
        "sma_crossover": {
            "short_window": range(10, 51, 10),
            "long_window": range(50, 151, 25)
        },
        "rsi": {
            "rsi_period": range(7, 22, 7),
            "oversold_threshold": [20, 30, 40],
            "overbought_threshold": [60, 70, 80]
        }
        # Add other strategy param grids here
    }

    param_grid = param_grids.get(args.strategy)
    if not param_grid:
        print(f"Error: No parameter grid defined for strategy '{args.strategy}' in this script.")
        print("Please add it to the 'param_grids' dictionary.")
        sys.exit(1)

    optimizer.set_params(**param_grid)

    print(f"\nOptimizing '{args.strategy}' with grid: {param_grid}")

    final_results = optimizer.run_walk_forward(
        in_sample_len=args.in_sample_len,
        out_of_sample_len=args.out_of_sample_len,
        step_size=args.step_size,
        optimize_for=args.optimize_for,
        initial_cash=10000.0
    )

    if final_results:
        print(f"\nWalk-forward analysis for '{args.strategy}' generated {len(final_results)} out-of-sample reports.")
    else:
        print(f"\nWalk-forward analysis for '{args.strategy}' did not produce any results.")

if __name__ == '__main__':
    main()
