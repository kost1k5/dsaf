import itertools
import pandas as pd
from typing import Type, Dict, Any, Generator

# Add project root to path for script execution
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.run_backtest import run_backtest
from src.strategies.base import BaseStrategy
from src.core.config import settings

class Optimizer:
    """
    A class to run backtest optimizations for a given strategy.
    """
    def __init__(self, strategy_class: Type[BaseStrategy], data: pd.DataFrame):
        self.strategy_class = strategy_class
        self.data = data
        self.param_grid: Dict[str, Any] = {}
        self.results = []

    def set_params(self, **params: Dict[str, Any]):
        """
        Sets the parameter grid for the optimization.
        Example: optimizer.set_params(short_window=range(10, 50, 5), long_window=range(50, 100, 10))
        """
        self.param_grid = params

    def _generate_param_combinations(self) -> Generator[Dict[str, Any], None, None]:
        """Generates all combinations of parameters from the grid."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))

    def run(self, optimize_for: str = "sharpe_ratio"):
        """
        Runs the optimization process.
        :param optimize_for: The metric to maximize (e.g., 'sharpe_ratio', 'pnl_percent').
        :return: A tuple of (best_params, best_result_dict).
        """
        if not self.param_grid:
            raise ValueError("Parameter grid is not set. Use set_params() first.")

        param_combinations = list(self._generate_param_combinations())
        print(f"Starting optimization for {len(param_combinations)} combinations...")

        self.results = []
        for i, params in enumerate(param_combinations):
            print(f"\n--- Running combo {i+1}/{len(param_combinations)}: {params} ---")
            strategy_instance = self.strategy_class(**params)

            result = run_backtest(
                strategy=strategy_instance,
                data=self.data.copy(), # Use a copy of the data for each run
                commission_pct=settings.BACKTEST_COMMISSION_PCT,
                slippage_pct=settings.BACKTEST_SLIPPAGE_PCT
            )

            if result:
                self.results.append({'params': params, 'metrics': result})

        if not self.results:
            print("Optimization produced no valid results.")
            return None, None

        # Find the best result
        best_run = max(self.results, key=lambda x: x['metrics'][optimize_for])

        print("\n--- Optimization Complete ---")
        print(f"Best result found by optimizing for '{optimize_for}':")
        print(f"Parameters: {best_run['params']}")
        print(f"Sharpe Ratio: {best_run['metrics']['sharpe_ratio']:.2f}")
        print(f"P/L %: {best_run['metrics']['pnl_percent']:.2f}%")
        print(f"Max Drawdown: {best_run['metrics']['max_drawdown']:.2f}%")

        return best_run['params'], best_run['metrics']

# Example Usage
if __name__ == '__main__':
    from src.strategies.sma_crossover import SmaCrossoverStrategy
    from src.data_collector.collector import DataCollector

    # 1. Get data
    print("Fetching data for optimization...")
    collector = DataCollector()
    symbol_to_optimize = 'BTC/USDT'
    timeframe_to_optimize = '1d'
    candles_list = collector.fetch_candles(symbol_to_optimize, timeframe=timeframe_to_optimize, limit=365)
    candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
    candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms', utc=True)
    candles_df.attrs = {'symbol': symbol_to_optimize, 'timeframe': timeframe_to_optimize}

    # 2. Setup and run optimizer
    optimizer = Optimizer(strategy_class=SmaCrossoverStrategy, data=candles_df)

    optimizer.set_params(
        short_window=range(10, 31, 5), # 10, 15, 20, 25, 30
        long_window=range(40, 61, 10) # 40, 50, 60
    )

    optimizer.run(optimize_for="sharpe_ratio")
