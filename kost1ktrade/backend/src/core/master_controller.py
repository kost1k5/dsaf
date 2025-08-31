import time
import pandas as pd
import random
from src.core.bot_state import bot_state
from src.core.market_analyzer import get_market_state
from src.core.bot_controller import start_bot_loop, stop_bot_loop
from src.data_collector.collector import DataCollector
from src.core.config import settings
from src.ml.predictor import Predictor
from src.ml.feature_generator import create_features
from src.core.sentiment_analyzer import SentimentAnalyzer

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
        predictor = Predictor()
        sentiment_analyzer = SentimentAnalyzer()
    except Exception as e:
        print(f"FATAL in Master Controller: Failed to initialize components: {e}")
        bot_state.master_bot_mode = "stopped"
        return

    check_interval_seconds = 3600 # Check market state every hour

    while not bot_state.master_bot_stop_event.is_set():
        try:
            print(f"({time.ctime()}) --- Master Controller Cycle ---")

            # 1. Fetch Data
            symbol = settings.SYMBOLS[0]
            timeframe = '1h'
            candles_list = collector.fetch_candles(symbol, timeframe, limit=500) # Fetch more data for feature generation
            if not candles_list or len(candles_list) < 50:
                raise ValueError("Not enough data fetched for analysis.")

            candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
            candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms')

            # 2. Analyze Market State, Sentiment, and ML Prediction
            market_state, adx_value = get_market_state(candles_df)
            news_sentiment = sentiment_analyzer.get_sentiment(symbol.split('/')[0])

            ml_prediction = 0
            if predictor.is_ready():
                features_df = create_features(candles_df.copy())
                latest_features = features_df.tail(1)
                ml_prediction = predictor.predict(latest_features)

            bot_state.market_state = f"{market_state} (Sentiment: {news_sentiment:.2f}, ML: {ml_prediction})"
            bot_state.adx_value = adx_value
            print(f"Analysis for {symbol}: Market State={market_state}, ADX={adx_value}, News Sentiment={news_sentiment}, ML Prediction={ml_prediction}")

            # 3. Decide on Strategy
            # Enhanced logic: Check for strong negative signals first
            if ml_prediction == -1 or news_sentiment < -0.4:
                print("Bearish signal from ML or News. Stopping all bots for safety.")
                if bot_state.signal_bot_mode != "stopped": stop_bot_loop()
                continue

            preferred_strategies = STRATEGY_MAP.get(market_state, [])
            if not preferred_strategies:
                if bot_state.signal_bot_mode != "stopped": stop_bot_loop()
                continue

            # 4. Check Current Bot and Switch if Necessary
            current_strategy_name = getattr(bot_state, 'signal_bot_strategy_name', None)
            is_current_strategy_suitable = any(s['name'] == current_strategy_name for s in preferred_strategies)

            if not is_current_strategy_suitable:
                new_strategy = random.choice(preferred_strategies)
                print(f"Switching strategy! Current: '{current_strategy_name}', New Choice: '{new_strategy['name']}'")

                if bot_state.signal_bot_mode != "stopped":
                    stop_bot_loop()
                    time.sleep(10)

                print(f"Starting new signal bot with strategy: {new_strategy['name']}...")
                start_bot_loop(
                    mode='demo',
                    strategy_name=new_strategy["name"],
                    strategy_params=new_strategy["params"]
                )
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
