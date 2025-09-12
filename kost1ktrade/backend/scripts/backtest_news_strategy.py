import pandas as pd
import numpy as np
import talib
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from datetime import datetime, timedelta
import sys
import os
import time

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.session import SessionLocal
from src.database.models import EconomicCalendarEvent
from src.core.position_sizer import calculate_position_size

def run_news_backtest():
    """
    Runs a simplified, candle-based backtest of the news trading strategy.
    """
    print("--- Starting News Strategy Backtest ---")

    # --- Strategy Parameters ---
    config = {
        "instrument": "BTC-USDT-SWAP",
        "target_events": ["CPI", "FOMC", "Non-Farm Payrolls"],
        "initial_equity": 10000.0,
        "risk_percent": 0.01,
        "atr_timeframe": "5m",
        "atr_period": 14,
        "atr_multiplier": 2.5,
        "z_score_threshold": 2.0,
        "time_exit_minutes": 30,
        "contract_size": 0.001 # For BTC-USDT-SWAP
    }
    print(f"Backtest Configuration: {config}")

    db: Session = SessionLocal()
    try:
        # 1. Load all historical economic events
        all_events = (
            db.query(EconomicCalendarEvent)
            .filter(
                EconomicCalendarEvent.actual.isnot(None),
                EconomicCalendarEvent.forecast.isnot(None)
            )
            .order_by(asc(EconomicCalendarEvent.event_datetime))
            .all()
        )
        print(f"Loaded {len(all_events)} historical economic events from the database.")

        if not all_events:
            print("No historical events found. Cannot run backtest.")
            return

        # 2. Fetch OHLCV data covering the entire event range
        start_date = all_events[0].event_datetime - timedelta(days=1) # Add buffer for initial ATR
        end_date = all_events[-1].event_datetime + timedelta(days=1) # Add buffer for final trade
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)

        print(f"Fetching OHLCV data for {config['instrument']} from {start_date} to {end_date}...")
        exchange = __import__('ccxt').okx()

        all_ohlcv = []
        since = start_timestamp
        while since < end_timestamp:
            # OKX fetch_ohlcv limit is 100, but we ask for a bit more just in case
            # The library will handle the actual limit.
            ohlcv_chunk = exchange.fetch_ohlcv(config['instrument'], config['atr_timeframe'], since=since, limit=300)
            if not ohlcv_chunk:
                break

            all_ohlcv.extend(ohlcv_chunk)
            since = ohlcv_chunk[-1][0] + 1 # +1 ms to avoid fetching the same candle

            # Be respectful of API limits
            time.sleep(exchange.rateLimit / 1000)

        if not all_ohlcv:
            print("Failed to fetch any OHLCV data for the specified range.")
            return

        price_data = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        price_data.drop_duplicates(subset=['timestamp'], inplace=True)
        price_data['datetime'] = pd.to_datetime(price_data['timestamp'], unit='ms')
        price_data.set_index('datetime', inplace=True)
        print(f"Fetched and merged {len(price_data)} unique price candles.")

        # --- Backtest Simulation ---
        equity = config['initial_equity']
        trade_log = []

        # 3. Iterate through each event
        for i, event in enumerate(all_events):
            # Find the candle corresponding to the event time
            event_time = event.event_datetime
            try:
                # Get the first candle at or just after the event
                entry_candle_index = price_data.index.searchsorted(event_time)
                entry_candle = price_data.iloc[entry_candle_index]
            except IndexError:
                continue # Skip if no price data for this event time

            # --- Z-Score Calculation (Simulated) ---
            # Get historical surprises *before* the current event
            historical_surprises = [
                float(e.actual) - float(e.forecast)
                for e in all_events[:i]
                if e.event_name == event.event_name
            ]

            if len(historical_surprises) < 2:
                continue # Not enough data for std dev

            std_dev = np.std(historical_surprises)
            if std_dev == 0:
                continue

            current_surprise = float(event.actual) - float(event.forecast)
            z_score = current_surprise / std_dev

            # 4. Check Trigger
            if abs(z_score) < config['z_score_threshold']:
                continue

            # --- Signal & Trade Simulation ---
            print(f"\n--- TRADE TRIGGER ---")
            print(f"Event: {event.event_name} at {event_time}")
            print(f"Z-Score: {z_score:.2f}")

            trade_side = None
            if any(kw.lower() in event.event_name.lower() for kw in ["cpi", "fomc", "nfp"]):
                trade_side = 'sell' if z_score > 0 else 'buy'

            if not trade_side:
                continue

            # --- Simulate Execution ---
            entry_price = entry_candle['close']

            # Calculate ATR from data *before* the entry candle
            atr_data_end_index = entry_candle_index - 1
            if atr_data_end_index < config['atr_period']:
                continue

            atr_df = price_data.iloc[:atr_data_end_index]
            atr = talib.ATR(atr_df['high'], atr_df['low'], atr_df['close'], timeperiod=config['atr_period']).iloc[-1]

            stop_loss_dist = atr * config['atr_multiplier']
            stop_loss_price = entry_price - stop_loss_dist if trade_side == 'buy' else entry_price + stop_loss_dist

            size = calculate_position_size(equity, config['risk_percent'], entry_price, stop_loss_price, config['contract_size'])

            if not size or size <= 0:
                print("Position size is zero. Skipping trade.")
                continue

            # --- Simulate Outcome ---
            exit_price = None
            exit_reason = "Time Exit"
            exit_candle_index = entry_candle_index + (config['time_exit_minutes'] // int(config['atr_timeframe'].replace('m','')))

            future_candles = price_data.iloc[entry_candle_index + 1 : exit_candle_index + 1]

            for idx, candle in future_candles.iterrows():
                if trade_side == 'buy' and candle['low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = "Stop Loss"
                    break
                elif trade_side == 'sell' and candle['high'] >= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = "Stop Loss"
                    break

            if not exit_price: # If SL not hit, exit at the end of the time window
                exit_price = future_candles.iloc[-1]['close'] if not future_candles.empty else entry_price

            # Calculate PnL
            pnl_per_contract = (exit_price - entry_price) if trade_side == 'buy' else (entry_price - exit_price)
            total_pnl = pnl_per_contract * size * config['contract_size']
            equity += total_pnl

            trade_log.append({
                "entry_time": event_time,
                "event": event.event_name,
                "side": trade_side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_loss": stop_loss_price,
                "exit_reason": exit_reason,
                "pnl": total_pnl,
                "equity": equity
            })
            print(f"Trade executed: {trade_side.upper()} at ${entry_price:.2f}. Exit at ${exit_price:.2f}. PnL: ${total_pnl:.2f}. Reason: {exit_reason}.")

    finally:
        db.close()

    # 5. Report Results
    print("\n--- Backtest Results ---")
    if not trade_log:
        print("No trades were executed during the backtest period.")
        return

    results_df = pd.DataFrame(trade_log)
    wins = results_df[results_df['pnl'] > 0]
    losses = results_df[results_df['pnl'] <= 0]

    print(f"Total Trades: {len(results_df)}")
    print(f"Wins: {len(wins)}")
    print(f"Losses: {len(losses)}")
    print(f"Win Rate: {(len(wins) / len(results_df) * 100):.2f}%")
    print(f"Total PnL: ${results_df['pnl'].sum():.2f}")
    print(f"Initial Equity: ${config['initial_equity']:.2f}")
    print(f"Final Equity: ${equity:.2f}")
    print(f"Total Return: {((equity / config['initial_equity']) - 1) * 100:.2f}%")

if __name__ == '__main__':
    run_news_backtest()
