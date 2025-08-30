from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

from src.core.bot_state import bot_state
from src.core.bot_controller import start_bot_loop, stop_bot_loop
from src.core.grid_bot_controller import start_grid_bot, stop_grid_bot
from src.notifications.telegram import run_bot
import threading

app = FastAPI(title="Kost1kTrade API")

# --- Signal Bot Control ---

class SignalBotControlRequest(BaseModel):
    mode: Literal['real', 'demo']

@app.post("/signal-bot/start", tags=["Signal Bot Control"])
async def start_signal_bot(request: SignalBotControlRequest):
    """
    Starts the signal-based trading bot in the specified mode.
    """
    try:
        print(f"Received request to start signal bot in '{request.mode}' mode.")
        start_bot_loop(request.mode)
        return {"message": f"Signal bot start process initiated in {request.mode} mode."}
    except (ValueError, ConnectionError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/signal-bot/stop", tags=["Signal Bot Control"])
async def stop_signal_bot():
    """
    Stops the signal-based trading bot if it is running.
    """
    try:
        print(f"Received request to stop signal bot from '{bot_state.signal_bot_mode}' mode.")
        stop_bot_loop()
        return {"message": "Signal bot stop process initiated."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/signal-bot/status", tags=["Signal Bot Control"])
async def get_signal_bot_status():
    """
    Returns the current status of the signal-based bot.
    """
    return {"current_mode": bot_state.signal_bot_mode}

# --- Grid Bot Control ---

class GridBotStartRequest(BaseModel):
    mode: Literal['real', 'demo']
    symbol: str
    amount_per_grid: float
    grid_range_low: float
    grid_range_high: float
    num_grids: int

@app.post("/grid-bot/start", tags=["Grid Bot Control"])
async def start_grid_bot_endpoint(request: GridBotStartRequest):
    """
    Starts the grid trading bot with the specified configuration.
    """
    try:
        grid_config = {
            "grid_range_low": request.grid_range_low,
            "grid_range_high": request.grid_range_high,
            "num_grids": request.num_grids,
        }
        start_grid_bot(request.mode, request.symbol, grid_config, request.amount_per_grid)
        return {"message": "Grid bot start process initiated."}
    except (ValueError, ConnectionError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/grid-bot/stop", tags=["Grid Bot Control"])
async def stop_grid_bot_endpoint():
    """
    Stops the grid trading bot.
    """
    try:
        stop_grid_bot()
        return {"message": "Grid bot stop process initiated."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/grid-bot/status", tags=["Grid Bot Control"])
async def get_grid_bot_status():
    """
    Returns the current status of the grid bot.
    """
    return {"current_mode": bot_state.grid_bot_mode}


# --- Health Check ---

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Check if the API is running."""
    return {"status": "ok"}
