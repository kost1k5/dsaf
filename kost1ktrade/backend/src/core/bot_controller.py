import time
import pandas as pd
import numpy as np
import talib
from typing import Dict, Any
from src.core.bot_state import bot_state
from src.data_collector.collector import DataCollector
from src.trading.engine import TradingEngine
from src.core.config import settings
from src.core.strategy_loader import get_strategy_class
from src.core.risk_manager import calculate_position_size

def _process_signal(engine: TradingEngine, strategy: Any, symbol: str, candles_df: pd.DataFrame):
    """
    Analyzes the latest signal and executes a trade if conditions are met.
    This is the core logic that was previously inside the loop.
    """
    print("Analyzing data and generating signal...")
    result_df = strategy.generate_signals(candles_df)
    latest_signal = result_df.iloc[-1]['signal']
    print(f"Strategy: {strategy.__class__.__name__} | Signal: {latest_signal}")

    base_currency, quote_currency = symbol.split('-')
    is_in_position, active_symbol = bot_state.get_position_state()

    if latest_signal == 'BUY':
        if is_in_position:
            print(f"Skipping BUY signal: Already in a position with {active_symbol}.")
            return

        balance = engine.get_balance()
        capital = balance.get(quote_currency, 0)
        current_price = candles_df.iloc[-1]['close']
        latest_atr = talib.ATR(candles_df['high'], candles_df['low'], candles_df['close'], timeperiod=14).iloc[-1]

        if pd.isna(latest_atr):
            print("Skipping trade: ATR could not be calculated.")
            return

        if capital > settings.BOT_CONTROLLER.MIN_CAPITAL_FOR_TRADE:
            amount_to_buy = calculate_position_size(
                capital=capital,
                risk_per_trade_pct=settings.RISK.RISK_PER_TRADE_PCT,
                atr_value=latest_atr,
                atr_multiplier=settings.RISK.ATR_MULTIPLIER,
                price=current_price
            )
            print(f"Risk Manager: Capital=${capital:.2f} -> Position Size: {amount_to_buy:.6f} {base_currency}")
            if amount_to_buy * current_price > settings.BOT_CONTROLLER.MIN_ORDER_SIZE_USD:
                sl_distance = latest_atr * settings.STRATEGY.RISK_SL_ATR_MULT
                tp_distance = latest_atr * settings.STRATEGY.RISK_TP_ATR_MULT

                sl_price = current_price - sl_distance
                tp_price = current_price + tp_distance

                print(f"Opening position: Buying {amount_to_buy:.6f} {base_currency}")
                engine.create_order(
                    symbol,
                    'market',
                    'buy',
                    amount_to_buy,
                    sl_price=sl_price,
                    tp_price=tp_price
                )
                bot_state.set_position_state(True, symbol) # Set state after opening position
            else:
                print("Skipping BUY order: Calculated position size is below minimum threshold.")
        else:
            print("Skipping BUY order: Not enough capital.")

    elif latest_signal == 'SELL':
        if not is_in_position or active_symbol != symbol:
            print(f"Skipping SELL signal: Not in a position with {symbol}.")
            return

        balance = engine.get_balance()
        base_balance = balance.get(base_currency, 0)
        if base_balance > 0.0001:
            print(f"Closing position: Selling {base_balance:.6f} {base_currency}")
            engine.create_order(symbol, 'market', 'sell', base_balance)
            bot_state.set_position_state(False, None) # Clear state after closing position

def signal_trading_loop():
    """
    The main loop for the signal-based trading bot.
    """
    # This check happens once at the start
    if bot_state.get_master_bot_mode() == "stopped": # Using getter
        print("Bot was stopped before the trading loop could start.")
        return

    # These are set once by start_bot_loop and shouldn't change during the loop
    engine = bot_state.signal_bot_engine
    strategy = bot_state.signal_bot_strategy
    symbol = bot_state.signal_bot_symbol
    stop_event = bot_state.get_signal_bot_stop_event()

    if not engine or not strategy or not symbol:
        print("FATAL in thread: Trading engine, strategy, or symbol not available.")
        bot_state.set_signal_bot_state("stopped", None, None, None, None, None)
        return

    collector = DataCollector(exchange_id='okx')
    sleep_duration_seconds = settings.BOT_CONTROLLER.LOOP_SLEEP_SECONDS
    candle_limit = settings.BOT_CONTROLLER.CANDLE_LIMIT

    print(f"--- Background signal bot loop started for {strategy.__class__.__name__} on {symbol} ---")

    try:
        while not stop_event.is_set():
            try:
                print(f"({time.ctime()}) --- Signal Bot Cycle ---")
                print(f"Fetching latest {candle_limit} candles for {symbol}...")
                candles_list = collector.fetch_candles(symbol, timeframe=settings.TIMEFRAME, limit=candle_limit)
                if not candles_list or len(candles_list) < 20: # Need enough for indicators
                    raise ValueError(f"Could not fetch enough candle data (got {len(candles_list)})")

                candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])

                _process_signal(engine, strategy, symbol, candles_df)

                print(f"Cycle complete. Sleeping...")
                stop_event.wait(timeout=sleep_duration_seconds)

            except Exception as e:
                print(f"An error occurred in the trading loop: {e}")
                stop_event.wait(timeout=60)
    finally:
        print(f"--- Background signal bot loop for '{symbol}' has gracefully stopped ---")
        bot_state.set_signal_bot_state("stopped", None, None, None, None, None)

# The start/stop functions remain largely the same, but should use the new setters
def start_bot_loop(mode: str, symbol: str, strategy_name: str, strategy_params: Dict[str, Any]):
    if bot_state.get_master_bot_mode() != "stopped":
        raise ValueError("Bot is already running.")

    bot_state.get_signal_bot_stop_event().clear()

    try:
        StrategyClass = get_strategy_class(strategy_name)
        strategy = StrategyClass(**strategy_params)
        engine = TradingEngine(mode=mode)

        thread = threading.Thread(target=signal_trading_loop, daemon=True)
        bot_state.set_signal_bot_state(mode, engine, strategy, strategy_name, symbol, thread)
        thread.start()
        print(f"Signal bot state prepared and thread started.")
    except Exception as e:
        bot_state.set_signal_bot_state("stopped", None, None, None, None, None)
        raise e

def stop_bot_loop():
    if bot_state.get_master_bot_mode() == "stopped":
        raise ValueError("Bot is not running.")

    print("Signaling signal bot to stop...")
    bot_state.get_signal_bot_stop_event().set()

    thread = bot_state.get_signal_bot_thread()
    if thread and thread.is_alive():
        print("Waiting for bot thread to terminate...")
        thread.join(timeout=10) # Wait up to 10 seconds
        if thread.is_alive():
            print("Warning: Bot thread did not terminate in time.")

    return "Stop signal sent and processed."
