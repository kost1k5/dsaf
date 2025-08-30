import time
import pandas as pd
import random
from src.core.bot_state import bot_state
from src.core.market_analyzer import get_market_state
from src.core.bot_controller import start_bot_loop, stop_bot_loop
from src.data_collector.collector import DataCollector
from src.core.config import settings

# --- Strategy Mapping ---
# Each market state maps to a list of suitable strategies.
STRATEGY_MAP = {
    "Trending": [
        {
            "name": "macd",
            "params": {
                "fast_period": settings.INDICATORS.MACD_FAST,
                "slow_period": settings.INDICATORS.MACD_SLOW,
                "signal_period": settings.INDICATORS.MACD_SIGNAL,
            }
        },
        {
            "name": "parabolic_sar",
            "params": {
                "acceleration": settings.INDICATORS.PSAR_ACCELERATION,
                "maximum": settings.INDICATORS.PSAR_MAXIMUM,
            }
        },
        {
            "name": "ichimoku",
            "params": {
                "tenkan_period": settings.INDICATORS.IC_TENKAN,
                "kijun_period": settings.INDICATORS.IC_KIJUN,
                "senkou_b_period": settings.INDICATORS.IC_SENKOU_B,
            }
        }
    ],
    "Ranging": [
        {
            "name": "rsi",
            "params": {
                "rsi_period": settings.INDICATORS.RSI_PERIOD,
                "oversold_threshold": 30,
                "overbought_threshold": 70,
            }
        },
        {
            "name": "stochastic",
            "params": {
                "k_period": settings.INDICATORS.STOCH_K_PERIOD,
                "d_period": settings.INDICATORS.STOCH_D_PERIOD,
                "oversold_threshold": 20,
                "overbought_threshold": 80,
            }
        }
    ],
    "Weak Trend": [
        {
            "name": "sma_crossover",
            "params": {
                "short_window": settings.INDICATORS.SMA_PERIOD,
                "long_window": settings.INDICATORS.SMA_LONG_PERIOD,
            }
        },
        {
            "name": "awesome_oscillator",
            "params": {
                "fast_period": settings.INDICATORS.AO_FAST_PERIOD,
                "slow_period": settings.INDICATORS.AO_SLOW_PERIOD,
            }
        },
        {
            "name": "keltner_channels",
            "params": {
                "length": settings.INDICATORS.KC_LENGTH,
                "multiplier": settings.INDICATORS.KC_MULTIPLIER,
                "atr_length": settings.INDICATORS.KC_ATR_LENGTH,
            }
        }
    ]
}

def master_trading_loop():
    """
    The main loop for the master controller bot.
    It analyzes market conditions and launches the appropriate strategy bot.
    """
    if bot_state.master_bot_mode == "stopped":
        return

    print("--- Master Controller loop started ---")

    try:
        collector = DataCollector(exchange_id='okx')
    except Exception as e:
        print(f"FATAL in Master Controller: Failed to initialize DataCollector: {e}")
        bot_state.master_bot_mode = "stopped"
        return

    check_interval_seconds = 3600 # Check market state every hour

    while not bot_state.master_bot_stop_event.is_set():
        try:
            print(f"({time.ctime()}) --- Master Controller Cycle ---")

            # 1. Analyze Market State
            print("Analyzing market state...")
            symbol = settings.SYMBOLS[0]
            timeframe = '1h'
            candles_list = collector.fetch_candles(symbol, timeframe, limit=200)
            candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])

            market_state, adx_value = get_market_state(candles_df)
            bot_state.market_state = market_state
            bot_state.adx_value = adx_value
            print(f"Current Market State for {symbol}: {market_state} (ADX: {adx_value})")

            # 2. Decide on Strategy
            preferred_strategies = STRATEGY_MAP.get(market_state, [])
            if not preferred_strategies:
                print(f"No strategies defined for market state '{market_state}'. Holding.")
                if bot_state.signal_bot_mode != "stopped":
                    print("Stopping current bot due to undefined market state.")
                    stop_bot_loop()
                continue

            # 3. Check Current Bot and Switch if Necessary
            current_strategy_name = getattr(bot_state, 'signal_bot_strategy_name', None)

            is_current_strategy_suitable = any(s['name'] == current_strategy_name for s in preferred_strategies)

            if not is_current_strategy_suitable:
                # Randomly select a new strategy from the suitable list
                new_strategy = random.choice(preferred_strategies)

                print(f"Switching strategy! Current: '{current_strategy_name}', New Choice: '{new_strategy['name']}' from candidates.")

                if bot_state.signal_bot_mode != "stopped":
                    print("Stopping the current signal bot...")
                    stop_bot_loop()
                    time.sleep(10)

                print(f"Starting new signal bot with strategy: {new_strategy['name']}...")
                start_bot_loop(
                    mode='demo',
                    strategy_name=new_strategy["name"],
                    strategy_params=new_strategy["params"]
                )
                # The actual background task for the signal bot is started by the API that calls start_bot_loop.
                # This controller's job is just to set the state and let the user/API start the loop.
            else:
                print(f"Current strategy '{current_strategy_name}' is suitable for {market_state}. No change needed.")

            print("Master Controller cycle complete. Sleeping...")
            bot_state.master_bot_stop_event.wait(timeout=check_interval_seconds)

        except Exception as e:
            print(f"An error occurred in the master controller loop: {e}")
            bot_state.master_bot_stop_event.wait(timeout=60) # Wait a minute before retrying on error

    print("--- Master Controller loop has gracefully stopped ---")
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
