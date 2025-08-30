from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Literal, Dict, Any

from src.core.bot_state import bot_state
from src.core.bot_controller import start_bot_loop, stop_bot_loop, signal_trading_loop
from src.core.grid_bot_controller import start_grid_bot, stop_grid_bot, grid_trading_loop
from src.core.master_controller import start_master_bot, stop_master_bot, master_trading_loop

app = FastAPI(title="Kost1kTrade API")

@app.get("/balance", tags=["Account"])
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

@app.post("/signal-bot/start", tags=["Signal Bot Control"])
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

@app.post("/signal-bot/stop", tags=["Signal Bot Control"])
async def stop_signal_bot():
    """
    Stops the signal-based trading bot if it is running.
    """
    try:
        response = stop_bot_loop()
        return {"message": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/signal-bot/status", tags=["Signal Bot Control"])
async def get_signal_bot_status():
    """
    Returns the current status of the signal-based bot.
    """
    return {
        "mode": bot_state.signal_bot_mode,
        "strategy_name": bot_state.signal_bot_strategy_name
    }

# --- Grid Bot Control ---

class GridBotStartRequest(BaseModel):
    mode: Literal['real', 'demo']
    symbol: str
    amount_per_grid: float
    grid_range_low: float
    grid_range_high: float
    num_grids: int

@app.post("/grid-bot/start", tags=["Grid Bot Control"])
async def start_grid_bot_endpoint(request: GridBotStartRequest, background_tasks: BackgroundTasks):
    """
    Starts the grid trading bot with the specified configuration.
    The bot will run in a background task.
    """
    try:
        grid_config = {
            "grid_range_low": request.grid_range_low,
            "grid_range_high": request.grid_range_high,
            "num_grids": request.num_grids,
        }
        start_grid_bot(request.mode, request.symbol, grid_config, request.amount_per_grid)
        # Add the main loop to background tasks with its arguments
        background_tasks.add_task(
            grid_trading_loop,
            request.symbol,
            grid_config,
            request.amount_per_grid
        )
        return {"message": "Grid bot start process initiated."}
    except (ValueError, ConnectionError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/grid-bot/stop", tags=["Grid Bot Control"])
async def stop_grid_bot_endpoint():
    """
    Stops the grid trading bot.
    """
    try:
        response = stop_grid_bot()
        return {"message": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/grid-bot/status", tags=["Grid Bot Control"])
async def get_grid_bot_status():
    """
    Returns the current status of the grid bot.
    """
    return {"current_mode": bot_state.grid_bot_mode}


# --- Master Bot Control ---

@app.post("/master-bot/start", tags=["Master Bot Control"])
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

@app.post("/master-bot/stop", tags=["Master Bot Control"])
async def stop_master_bot_endpoint():
    """
    Stops the autonomous Master Controller and any active signal bot.
    """
    try:
        response = stop_master_bot()
        return {"message": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/master-bot/status", tags=["Master Bot Control"])
async def get_master_bot_status():
    """
    Returns the current status of the Master Controller.
    """
    return {
        "master_mode": getattr(bot_state, 'master_bot_mode', 'stopped'),
        "market_state": getattr(bot_state, 'market_state', None),
        "adx_value": getattr(bot_state, 'adx_value', None)
    }


# --- Health Check ---

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Check if the API is running."""
    return {"status": "ok"}
