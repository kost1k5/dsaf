import time
import pandas as pd
import json
import talib
import random
from src.core.bot_state import bot_state
from src.data_collector.collector import DataCollector
from src.core.config import settings
from src.core.bot_controller import start_bot_loop, stop_bot_loop

# --- Commander Parameters ---
STRATEGY_PARAMS_FILE = 'strategy_params.json'
COMMANDER_SYMBOL = settings.COMMANDER_SYMBOL
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = settings.MASTER_CONTROLLER.ADX_TREND_THRESHOLD
ADX_RANGE_THRESHOLD = settings.MASTER_CONTROLLER.ADX_RANGE_THRESHOLD

def load_strategy_params():
    """Loads strategy parameters from the JSON file."""
    try:
        with open(STRATEGY_PARAMS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading strategy params: {e}. Returning empty dict.")
        return {}

def master_trading_loop():
    """
    The main loop for the "Commander".
    Its sole responsibility is to analyze the market regime and set the activation status
    for all strategies based on their type ('trend' or 'range').
    """
    if bot_state.master_bot_mode == "stopped":
        return

    print("--- Commander loop started ---")

    try:
        collector = DataCollector(exchange_id='okx')
    except Exception as e:
        print(f"FATAL in Commander: Failed to initialize DataCollector: {e}")
        bot_state.master_bot_mode = "stopped"
        return

    check_interval_seconds = settings.MASTER_CONTROLLER.CHECK_INTERVAL_SECONDS

    while not bot_state.master_bot_stop_event.is_set():
        try:
            print(f"({time.ctime()}) --- Commander Cycle ---")
            
            # 1. Fetch Data for the main market symbol
            print(f"Fetching data for market analysis on {COMMANDER_SYMBOL}...")
            candles_list = collector.fetch_candles(COMMANDER_SYMBOL, settings.TIMEFRAME, limit=300)
            if not candles_list or len(candles_list) < ADX_PERIOD * 2:
                raise ValueError("Not enough data fetched for ADX analysis.")

            candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])

            # 2. Calculate ADX to determine market regime
            adx_values = talib.ADX(candles_df['high'], candles_df['low'], candles_df['close'], timeperiod=ADX_PERIOD)
            latest_adx = adx_values.iloc[-1]
            bot_state.adx_value = latest_adx

            market_regime = "indecisive"
            if latest_adx > ADX_TREND_THRESHOLD:
                market_regime = "trend"
            elif latest_adx < ADX_RANGE_THRESHOLD:
                market_regime = "range"

            bot_state.market_state = market_regime.capitalize()
            print(f"Market analysis complete: ADX = {latest_adx:.2f} -> Regime = {market_regime}")

            # 3. Load all strategies and update their active status in the bot_state
            all_strategies = load_strategy_params()
            new_active_statuses = {}

            for name, params in all_strategies.items():
                strategy_type = params.get("type")
                if strategy_type == market_regime:
                    new_active_statuses[name] = True
                else:
                    new_active_statuses[name] = False

            bot_state.active_strategies = new_active_statuses
            print(f"Updated active strategies: {', '.join([k for k, v in new_active_statuses.items() if v])}")

            # 4. Manage the active trading bot based on the new regime
            runnable_strategies = [k for k, v in new_active_statuses.items() if v]
            current_bot_strategy = bot_state.signal_bot_strategy_name
            is_bot_running = bot_state.signal_bot_mode != "stopped"

            if runnable_strategies:
                # There are suitable strategies for the current regime
                if not is_bot_running:
                    # No bot is running, so start one. Select the highest priority strategy.
                    runnable_params = {name: all_strategies[name] for name in runnable_strategies}
                    new_strategy_name = max(runnable_params, key=lambda k: runnable_params[k].get('priority', 0))

                    print(f"No bot running. Starting highest priority strategy: '{new_strategy_name}'")
                    params = all_strategies.get(new_strategy_name, {})
                    start_bot_loop(
                        mode=bot_state.master_bot_target_mode,
                        symbol=COMMANDER_SYMBOL,
                        strategy_name=new_strategy_name,
                        strategy_params=params
                    )
                elif current_bot_strategy not in runnable_strategies:
                    # The current bot is unsuitable for the new regime. Switch it.
                    print(f"Current strategy '{current_bot_strategy}' is unsuitable for '{market_regime}' regime.")
                    stop_bot_loop()
                    time.sleep(settings.MASTER_CONTROLLER.BOT_STOP_WAIT_SECONDS) # Give the bot time to stop gracefully

                    runnable_params = {name: all_strategies[name] for name in runnable_strategies}
                    new_strategy_name = max(runnable_params, key=lambda k: runnable_params[k].get('priority', 0))

                    print(f"Switching to highest priority strategy: '{new_strategy_name}'")
                    params = all_strategies.get(new_strategy_name, {})
                    start_bot_loop(
                        mode=bot_state.master_bot_target_mode,
                        symbol=COMMANDER_SYMBOL,
                        strategy_name=new_strategy_name,
                        strategy_params=params
                    )
                else:
                    print(f"Current strategy '{current_bot_strategy}' is still suitable. No change needed.")

            else:
                # No strategies are suitable for the current regime. Stop any running bot.
                if is_bot_running:
                    print(f"No suitable strategies for '{market_regime}' regime. Stopping current bot.")
                    stop_bot_loop()

            # 5. Sleep until the next cycle
            print("Commander cycle complete. Sleeping...")
            bot_state.master_bot_stop_event.wait(timeout=check_interval_seconds)

        except Exception as e:
            print(f"An error occurred in the commander loop: {e}")
            bot_state.master_bot_stop_event.wait(timeout=60)

    print("--- Commander loop has gracefully stopped ---")
    bot_state.master_bot_mode = "stopped"

def start_master_bot():
    if bot_state.master_bot_mode != "stopped":
        raise ValueError("Master Controller is already running.")

    bot_state.master_bot_mode = "running"
    bot_state.master_bot_stop_event.clear()
    print("Master Controller state prepared.")

def stop_master_bot():
    if bot_state.master_bot_mode == "stopped":
        raise ValueError("Master Controller is not running.")

    print("Signaling Master Controller to stop...")
    bot_state.master_bot_stop_event.set()

    # Also stop any active signal bot
    if bot_state.signal_bot_mode != "stopped":
        print("Stopping the active signal bot...")
        stop_bot_loop()

    return "Stop signal sent to Master Controller and any active bot."