from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
from pydantic import BaseModel
from typing import Literal, Dict, Any
import json

from datetime import datetime
from typing import List
from src.core.bot_state import bot_state
from src.core.bot_controller import start_bot_loop, stop_bot_loop, signal_trading_loop
from src.core.grid_bot_controller import start_grid_bot, stop_grid_bot, stop_all_grid_bots
from src.core.master_controller import start_master_bot, stop_master_bot, master_trading_loop
from src.core.backtester import Backtester
from src.core.strategy_loader import get_strategy_class
from src.data_collector.collector import DataCollector
from src.api.optimization import router as optimization_router
import pandas as pd

app = FastAPI(title="Kost1kTrade API")
router = APIRouter()

@router.get("/balance", tags=["Account"])
async def get_balance():
    """
    Fetches the current account balance from the active engine.
    """
    engine = bot_state.signal_bot_engine or bot_state.grid_bot_engine
    if not engine:
        raise HTTPException(status_code=400, detail="No bot is currently active. Start a bot to fetch balance.")

    balances = engine.get_balance()
    if balances is None:
        raise HTTPException(status_code=500, detail="Failed to fetch balance from the exchange.")

    return balances

# --- Signal Bot Control ---

class SignalBotStartRequest(BaseModel):
    mode: Literal['real', 'demo']
    strategy_name: str
    strategy_params: Dict[str, Any]

@router.post("/signal-bot/start", tags=["Signal Bot Control"])
async def start_signal_bot(request: SignalBotStartRequest, background_tasks: BackgroundTasks):
    """
    Starts the signal-based trading bot with a specific strategy and parameters.
    The bot will run in a background task.
    """
    try:
        print(f"Received request to start signal bot in '{request.mode}' mode with strategy '{request.strategy_name}'.")
        start_bot_loop(
            mode=request.mode,
            strategy_name=request.strategy_name,
            strategy_params=request.strategy_params
        )
        # Add the main loop to background tasks
        background_tasks.add_task(signal_trading_loop)
        return {"message": f"Signal bot '{request.strategy_name}' start process initiated in {request.mode} mode."}
    except (ValueError, ConnectionError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/signal-bot/stop", tags=["Signal Bot Control"])
async def stop_signal_bot():
    """
    Stops the signal-based trading bot if it is running.
    """
    try:
        response = stop_bot_loop()
        return {"message": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/signal-bot/status", tags=["Signal Bot Control"])
async def get_signal_bot_status():
    """
    Returns the current status of the signal-based bot.
    """
    return {
        "mode": bot_state.signal_bot_mode,
        "strategy_name": bot_state.signal_bot_strategy_name
    }

# --- Grid Bot Control ---

class GridBotConfig(BaseModel):
    grid_range_low: float
    grid_range_high: float
    num_grids: int
    amount_per_grid: float

class GridBotStartRequest(BaseModel):
    mode: Literal['real', 'demo']
    symbol: str
    config: GridBotConfig

class GridBotStopRequest(BaseModel):
    symbol: str

@router.get("/grid-bot/status", tags=["Grid Bot Control"])
async def get_grid_bots_status():
    """
    Returns the current status of all grid bots.
    """
    return {
        "mode": bot_state.grid_bots_mode,
        "active_grids": list(bot_state.grid_bot_configs.keys())
    }

@router.get("/grid-bot/configs", tags=["Grid Bot Control"])
async def get_grid_bot_configs():
    """
    Returns the configurations for all currently active grid bots.
    """
    return bot_state.grid_bot_configs

@router.post("/grid-bot/start", tags=["Grid Bot Control"])
async def start_grid_bot_endpoint(request: GridBotStartRequest):
    """
    Starts a grid trading bot for a specific symbol.
    The bot will run in a background thread managed by the controller.
    """
    try:
        response = start_grid_bot(
            mode=request.mode,
            symbol=request.symbol,
            config=request.config.dict()
        )
        return {"message": response}
    except (ValueError, ConnectionError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/grid-bot/stop", tags=["Grid Bot Control"])
async def stop_grid_bot_endpoint(request: GridBotStopRequest):
    """
    Stops the grid trading bot for a specific symbol.
    """
    try:
        response = stop_grid_bot(symbol=request.symbol)
        return {"message": response}
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/grid-bot/stop-all", tags=["Grid Bot Control"])
async def stop_all_grid_bots_endpoint():
    """
    Stops all running grid trading bots.
    """
    try:
        response = stop_all_grid_bots()
        return {"message": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Master Bot Control ---

@router.post("/master-bot/start", tags=["Master Bot Control"])
async def start_master_bot_endpoint(background_tasks: BackgroundTasks):
    """
    Starts the autonomous Master Controller.
    It will analyze the market and manage signal bots automatically.
    """
    try:
        start_master_bot()
        background_tasks.add_task(master_trading_loop)
        return {"message": "Master Controller started successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/master-bot/stop", tags=["Master Bot Control"])
async def stop_master_bot_endpoint():
    """
    Stops the autonomous Master Controller and any active signal bot.
    """
    try:
        response = stop_master_bot()
        return {"message": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/master-bot/status", tags=["Master Bot Control"])
async def get_master_bot_status():
    """
    Returns the current status of the Master Controller.
    """
    return {
        "master_mode": getattr(bot_state, 'master_bot_mode', 'stopped'),
        "market_state": getattr(bot_state, 'market_state', None),
        "adx_value": getattr(bot_state, 'adx_value', None)
    }

# --- Master Bot Settings ---

class MasterBotSettings(BaseModel):
    target_mode: Literal['demo', 'real']

@router.get("/master-bot/settings", tags=["Master Bot Control"])
async def get_master_bot_settings():
    """
    Returns the current target mode for the Master Controller.
    """
    return {"target_mode": bot_state.master_bot_target_mode}

@router.post("/master-bot/settings", tags=["Master Bot Control"])
async def set_master_bot_settings(settings: MasterBotSettings):
    """
    Sets the target mode for the Master Controller.
    This can only be done when the bot is stopped.
    """
    if bot_state.master_bot_mode != 'stopped':
        raise HTTPException(status_code=400, detail="Cannot change settings while the Master Bot is running.")

    bot_state.master_bot_target_mode = settings.target_mode
    return {"message": f"Master Bot target mode set to {settings.target_mode}"}


# --- Health Check ---

@router.get("/health", tags=["Monitoring"])
def health_check():
    """Check if the API is running."""
    return {"status": "ok"}

# --- Strategy Management ---
STRATEGY_PARAMS_FILE = 'strategy_params.json'

class StrategyStatusRequest(BaseModel):
    statuses: Dict[str, bool]

class StrategyParamsRequest(BaseModel):
    name: str
    params: Dict[str, Any]

def load_strategy_params():
    try:
        with open(STRATEGY_PARAMS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # In a real-world app, might fallback to defaults or raise an error
        return {}

def save_strategy_params(params: dict):
    with open(STRATEGY_PARAMS_FILE, 'w') as f:
        json.dump(params, f, indent=4)

@router.get("/strategies/status", tags=["Strategies"])
async def get_strategies_status():
    """
    Returns a list of all available strategies, their current parameters from file,
    and their current activation status. This now includes the Grid Bot.
    """
    strategy_params = load_strategy_params()
    response = {}
    for name, params in strategy_params.items():
        response[name] = {
            "params": params,
            "active": bot_state.active_strategies.get(name, False),
            "type": "signal"
        }

    # Add the Grid Bot status as a special entry
    response['grid'] = {
        "params": bot_state.grid_bot_configs,  # Now returns all active grid configs
        "active": bot_state.grid_bots_mode != 'stopped',
        "type": "grid"
    }
    return response

@router.post("/strategies/status", tags=["Strategies"])
async def set_strategies_status(request: StrategyStatusRequest):
    """
    Updates the activation status for multiple strategies.
    """
    print(f"Received request to update strategy statuses: {request.statuses}")
    for name, status in request.statuses.items():
        # BUG FIX: The original code only updated the status if the strategy name
        # already existed as a key in the `active_strategies` dict.
        # On a fresh start, this dict is empty, so no statuses were ever updated.
        # The fix is to remove the conditional check and directly set the status.
        bot_state.active_strategies[name] = status
    return {"message": "Strategy statuses updated successfully.", "new_statuses": bot_state.active_strategies}

@router.post("/strategies/params", tags=["Strategies"])
async def set_strategy_params(request: StrategyParamsRequest):
    """
    Updates the parameters for a specific strategy and saves them to file.
    """
    strategy_params = load_strategy_params()
    if request.name not in strategy_params:
        raise HTTPException(status_code=404, detail="Strategy not found")

    print(f"Updating params for {request.name}: {request.params}")
    strategy_params[request.name] = request.params
    save_strategy_params(strategy_params)

    return {"message": f"Parameters for {request.name} updated successfully."}

# --- Simulation / Backtesting ---

class StrategyConfigRequest(BaseModel):
    name: str
    params: Dict[str, Any]

class SimulationRunRequest(BaseModel):
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    strategies: List[StrategyConfigRequest]

@router.post("/simulation/run", tags=["Simulation"])
async def run_simulation(request: SimulationRunRequest):
    """
    Runs a backtest simulation for a list of strategies.
    """
    try:
        # 1. Fetch Data
        collector = DataCollector(exchange_id='okx')
        start_timestamp = int(datetime.strptime(request.start_date, "%Y-%m-%d").timestamp() * 1000)
        end_timestamp = int(datetime.strptime(request.end_date, "%Y-%m-%d").timestamp() * 1000)

        # Fetch data in chunks if the range is large (simplified here)
        candles_list = collector.fetch_candles(
            request.symbol,
            request.timeframe,
            since=start_timestamp,
            limit=5000 # A high limit, assuming the range isn't excessively long
        )

        if not candles_list:
            raise ValueError("Could not fetch candle data for the specified range.")

        candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
        candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms')

        # Filter dataframe for the exact date range, as fetch might bring more
        candles_df = candles_df[
            (candles_df['open_time'] >= request.start_date) &
            (candles_df['open_time'] <= request.end_date)
        ]

        if candles_df.empty:
            raise ValueError("No candle data available for the specified date range after filtering.")

        # 2. Run Backtest for each strategy
        all_results = []
        for strategy_config in request.strategies:
            try:
                StrategyClass = get_strategy_class(strategy_config.name)
                params = strategy_config.params.copy()
                if strategy_config.name == 'grid':
                    # GridStrategy doesn't take a symbol in its constructor, so remove it if it exists
                    params.pop('symbol', None)

                strategy_instance = StrategyClass(**params)

                backtester = Backtester(strategy=strategy_instance, candles_df=candles_df.copy())
                result = backtester.run()

                all_results.append({
                    "strategy_name": strategy_config.name,
                    "params": strategy_config.params,
                    "metrics": result
                })
            except Exception as e:
                import traceback
                # Log error for a specific strategy but continue with others
                print(f"Error backtesting strategy {strategy_config.name}:")
                traceback.print_exc() # Print the full traceback
                all_results.append({
                    "strategy_name": strategy_config.name,
                    "params": strategy_config.params,
                    "error": str(e)
                })

        return all_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during simulation: {str(e)}")

app.include_router(router, prefix="/api")
app.include_router(optimization_router, prefix="/api/optimization")
