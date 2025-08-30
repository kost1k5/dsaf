from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Literal

from src.core.bot_state import bot_state
# We will create this controller module next
# from src.core.bot_controller import start_bot_loop, stop_bot_loop

app = FastAPI(title="Kost1kTrade API")

class BotControlRequest(BaseModel):
    mode: Literal['real', 'demo']

@app.post("/bot/start", tags=["Bot Control"])
async def start_bot(request: BotControlRequest, background_tasks: BackgroundTasks):
    """
    Starts the trading bot in the specified mode ('real' or 'demo').
    """
    if bot_state.mode != "stopped":
        raise HTTPException(status_code=400, detail=f"Bot is already running in '{bot_state.mode}' mode.")

    print(f"Received request to start bot in '{request.mode}' mode.")
    # In a real implementation, we would start a background task.
    # For now, we just update the state.
    # background_tasks.add_task(start_bot_loop, request.mode)
    bot_state.mode = request.mode
    return {"message": f"Bot started in {request.mode} mode. (Simulation)"}

@app.post("/bot/stop", tags=["Bot Control"])
async def stop_bot():
    """
    Stops the trading bot if it is running.
    """
    if bot_state.mode == "stopped":
        raise HTTPException(status_code=400, detail="Bot is not running.")

    print(f"Received request to stop bot from '{bot_state.mode}' mode.")
    # stop_bot_loop()
    bot_state.mode = "stopped"
    bot_state.trading_engine = None
    return {"message": "Bot stopped. (Simulation)"}

@app.get("/bot/status", tags=["Bot Control"])
async def get_bot_status():
    """
    Returns the current status of the bot.
    """
    return {"current_mode": bot_state.mode}


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Check if the API is running."""
    return {"status": "ok"}
