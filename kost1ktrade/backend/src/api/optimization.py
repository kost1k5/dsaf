from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List

from src.core.strategy_loader import get_strategy_class
from src.data_collector.collector import DataCollector
from src.optimization.optimizer import Optimizer
import pandas as pd
from datetime import datetime

router = APIRouter()

class OptimizationRequest(BaseModel):
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    param_grid: Dict[str, List[Any]]
    optimize_for: str = "sharpe_ratio"

@router.post("/run", tags=["Optimization"])
async def run_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """
    Runs a parameter optimization for a given strategy.
    """
    try:
        # 1. Fetch Data
        collector = DataCollector(exchange_id='okx')
        start_timestamp = int(datetime.strptime(request.start_date, "%Y-%m-%d").timestamp() * 1000)
        end_timestamp = int(datetime.strptime(request.end_date, "%Y-%m-%d").timestamp() * 1000)

        candles_list = collector.fetch_candles(
            request.symbol,
            request.timeframe,
            since=start_timestamp,
            limit=5000
        )

        if not candles_list:
            raise ValueError("Could not fetch candle data for the specified range.")

        candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
        candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms')
        candles_df.attrs = {'symbol': request.symbol, 'timeframe': request.timeframe}

        # 2. Run Optimizer
        StrategyClass = get_strategy_class(request.strategy_name)
        optimizer = Optimizer(strategy_class=StrategyClass, data=candles_df)

        optimizer.set_params(**request.param_grid)

        best_params, best_metrics = optimizer.run_single(optimize_for=request.optimize_for)

        if not best_params:
            raise HTTPException(status_code=404, detail="Optimization did not yield any results.")

        return {
            "best_params": best_params,
            "best_metrics": best_metrics
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during optimization: {str(e)}")
