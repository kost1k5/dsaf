import time
import pandas as pd
import random
import json
from src.core.bot_state import bot_state
from src.core.market_analyzer import get_market_state
from src.core.bot_controller import start_bot_loop, stop_bot_loop
from src.data_collector.collector import DataCollector
from src.core.config import settings
from src.ml.predictor import Predictor
from src.ml.feature_generator import create_features
from src.core.sentiment_analyzer import SentimentAnalyzer

# --- Strategy Mapping & Parameter Loading ---
STRATEGY_PARAMS_FILE = 'strategy_params.json'

# Each market state maps to a list of suitable strategy names.
# The parameters are now loaded from a separate JSON file.
STRATEGY_MAP = {
    "Trending": ["macd", "parabolic_sar", "ichimoku"],
    "Ranging": ["rsi", "stochastic", "bollinger_bands"],
    "Weak Trend": ["sma_crossover", "awesome_oscillator", "keltner_channels"]
}

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

            # Load the latest strategy parameters at the start of each cycle
            strategy_params = load_strategy_params()

            # Randomly select a symbol to analyze for this cycle
            if not settings.SYMBOLS:
                print("No symbols configured. Skipping cycle.")
                bot_state.master_bot_stop_event.wait(timeout=check_interval_seconds)
                continue

            symbol = random.choice(settings.SYMBOLS)
            print(f"--- Analyzing selected symbol: {symbol} ---")

            # 1. Fetch Data
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

            preferred_strategy_names = STRATEGY_MAP.get(market_state, [])
            if not preferred_strategy_names:
                if bot_state.signal_bot_mode != "stopped": stop_bot_loop()
                continue

            # Filter strategies based on their active status
            runnable_strategy_names = [name for name in preferred_strategy_names if bot_state.active_strategies.get(name, False)]
            print(f"Found {len(preferred_strategy_names)} preferred strategies for {market_state}. After filtering, {len(runnable_strategy_names)} are active.")

            if not runnable_strategy_names:
                print("No active strategies available for the current market state. Stopping bot if running.")
                if bot_state.signal_bot_mode != "stopped": stop_bot_loop()
                continue

            # 4. Check Current Bot and Switch if Necessary
            current_strategy_name = getattr(bot_state, 'signal_bot_strategy_name', None)
            is_current_strategy_suitable = current_strategy_name in runnable_strategy_names

            if not is_current_strategy_suitable:
                new_strategy_name = random.choice(runnable_strategy_names)
                new_strategy_params = strategy_params.get(new_strategy_name)

                if not new_strategy_params:
                    print(f"Warning: Parameters for '{new_strategy_name}' not found in JSON file. Skipping.")
                    continue

                print(f"Switching strategy! Current: '{current_strategy_name}', New Choice: '{new_strategy_name}'")

                if bot_state.signal_bot_mode != "stopped":
                    stop_bot_loop()
                    time.sleep(10)

                print(f"Starting new signal bot with strategy: {new_strategy_name}...")
                start_bot_loop(
                    mode=bot_state.master_bot_target_mode,
                    strategy_name=new_strategy_name,
                    strategy_params=new_strategy_params
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
