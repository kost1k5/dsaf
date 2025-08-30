import itertools
import pandas as pd
from typing import Type, Dict, Any, Generator

# Add project root to path for script execution
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.run_backtest import run_backtest
from src.strategies.base import BaseStrategy
from src.core.config import settings
from .walk_forward import walk_forward_splitter

class Optimizer:
    """
    A class to run backtest optimizations for a given strategy,
    including simple and walk-forward methods.
    """
    def __init__(self, strategy_class: Type[BaseStrategy], data: pd.DataFrame):
        self.strategy_class = strategy_class
        self.data = data
        self.param_grid: Dict[str, Any] = {}

    def set_params(self, **params: Dict[str, Any]):
        """Sets the parameter grid for the optimization."""
        self.param_grid = params

    def _generate_param_combinations(self) -> Generator[Dict[str, Any], None, None]:
        """Generates all combinations of parameters from the grid."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))

    def run_single(self, optimize_for: str = "sharpe_ratio"):
        """Runs a single optimization over the entire dataset."""
        print("--- Running Single Optimization ---")
        best_params, best_metrics = self._run_optimization_on_data(self.data, optimize_for)

        if best_params:
            print("\n--- Single Optimization Complete ---")
            print(f"Best result found by optimizing for '{optimize_for}':")
            print(f"Parameters: {best_params}")
            print(f"Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
            print(f"P/L %: {best_metrics['pnl_percent']:.2f}%")

        return best_params, best_metrics

    def _run_optimization_on_data(self, data: pd.DataFrame, optimize_for: str):
        """Helper to run optimization on a specific slice of data."""
        if not self.param_grid:
            raise ValueError("Parameter grid is not set. Use set_params() first.")

        param_combinations = list(self._generate_param_combinations())
        results = []
        # Suppress backtest report printing during optimization runs
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        for i, params in enumerate(param_combinations):
            strategy_instance = self.strategy_class(**params)
            result = run_backtest(
                strategy=strategy_instance,
                data=data.copy(),
                commission_pct=settings.BACKTEST_COMMISSION_PCT,
                slippage_pct=settings.BACKTEST_SLIPPAGE_PCT
            )
            if result:
                results.append({'params': params, 'metrics': result})

        sys.stdout.close()
        sys.stdout = original_stdout

        if not results:
            return None, None

        best_run = max(results, key=lambda x: x['metrics'][optimize_for])
        return best_run['params'], best_run['metrics']

    def run_walk_forward(self, in_sample_len: int, out_of_sample_len: int, step_size: int, optimize_for: str = "sharpe_ratio", initial_cash: float = 10000.0):
        """Runs a full walk-forward optimization."""
        print("--- Running Walk-Forward Optimization ---")

        data_splitter = walk_forward_splitter(self.data, in_sample_len, out_of_sample_len, step_size)

        out_of_sample_results = []

        for i, (in_sample_df, out_of_sample_df) in enumerate(data_splitter):
            print(f"\n--- Walk-Forward Step {i+1} ---")
            print(f"In-sample period: {in_sample_df.index[0]} - {in_sample_df.index[-1]}")

            best_params, _ = self._run_optimization_on_data(in_sample_df, optimize_for)
            if not best_params:
                print("No optimal parameters found in this step. Skipping.")
                continue

            print(f"Best params for this window: {best_params}")

            print(f"Testing on out-of-sample: {out_of_sample_df.index[0]} - {out_of_sample_df.index[-1]}")
            strategy_instance = self.strategy_class(**best_params)
            out_of_sample_result = run_backtest(
                strategy=strategy_instance,
                data=out_of_sample_df.copy(),
                initial_cash=initial_cash, # Pass initial_cash to each backtest
                commission_pct=settings.BACKTEST_COMMISSION_PCT,
                slippage_pct=settings.BACKTEST_SLIPPAGE_PCT
            )

            if out_of_sample_result:
                out_of_sample_results.append(out_of_sample_result)

        # --- Aggregate and Analyze Walk-Forward Results ---
        if not out_of_sample_results:
            print("Walk-forward analysis produced no results.")
            return None

        print("\n--- Final Walk-Forward Analysis Report ---")

        all_trades = [trade for result in out_of_sample_results for trade in result['trades']]

        final_balance = initial_cash
        for result in out_of_sample_results:
             period_return = result['final_balance'] / result['initial_balance']
             final_balance *= period_return

        total_pnl = final_balance - initial_cash
        total_return_percent = (total_pnl / initial_cash) * 100

        print(f"Total Out-of-Sample Periods: {len(out_of_sample_results)}")
        print(f"Initial Portfolio Value: ${initial_cash:,.2f}")
        print(f"Final Portfolio Value:   ${final_balance:,.2f}")
        print(f"Total Profit/Loss:       ${total_pnl:,.2f} ({total_return_percent:.2f}%)")
        print(f"Total Trades:            {len(all_trades)}")

        return out_of_sample_results

# Example Usage
if __name__ == '__main__':
    from src.strategies.sma_crossover import SmaCrossoverStrategy
    from src.data_collector.collector import DataCollector

    # 1. Get data
    print("Fetching data for Walk-Forward Analysis...")
    collector = DataCollector()
    candles_list = collector.fetch_candles('BTC/USDT', timeframe='1d', limit=500)
    candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
    candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms', utc=True)
    candles_df.attrs = {'symbol': 'BTC/USDT', 'timeframe': '1d'}

    # 2. Setup and run optimizer
    optimizer = Optimizer(strategy_class=SmaCrossoverStrategy, data=candles_df)

    optimizer.set_params(
        short_window=range(10, 31, 10),
        long_window=range(40, 71, 15)
    )

    final_results = optimizer.run_walk_forward(
        in_sample_len=180,
        out_of_sample_len=60,
        step_size=60,
        optimize_for="sharpe_ratio",
        initial_cash=10000.0
    )

    if final_results:
        print(f"\nWalk-forward analysis generated {len(final_results)} out-of-sample reports.")
