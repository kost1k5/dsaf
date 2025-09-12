import websocket
import json
import threading
import time
import ccxt
import pandas as pd
import talib
import math

# Import the modules created in previous steps
from .z_score_calculator import calculate_z_score
from .position_sizer import calculate_position_size

class NewsWebsocketClient:
    """
    A WebSocket client to connect to the OKX API, listen for real-time
    economic calendar events, and trigger trades based on a Z-score strategy.
    """
    def __init__(self, exchange: ccxt.Exchange, config: dict):
        self._ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        self.ws = None
        self.thread = None
        self.is_running = False
        self.exchange = exchange
        self.config = config
        self.active_trade_timers = {} # To track time-based exits

        # As per user spec, focus on high-impact US news
        self.target_events = self.config.get("target_events", ["CPI", "FOMC", "Non-Farm Payrolls"])
        self.instrument = self.config.get("instrument", "BTC-USDT-SWAP")
        self.risk_percent = self.config.get("risk_percent", 0.01)
        self.atr_multiplier = self.config.get("atr_multiplier", 2.5)
        self.atr_timeframe = self.config.get("atr_timeframe", "5m")
        self.time_exit_minutes = self.config.get("time_exit_minutes", 30)

    def _on_message(self, ws, message):
        """Callback function to handle incoming messages."""
        if message == "pong":
            print("[WebSocket] Received pong.")
            return

        data = json.loads(message)

        if 'event' in data:
            if data['event'] == 'subscribe':
                print(f"Successfully subscribed to channel: {data['arg']['channel']}")
            elif data['event'] == 'error':
                print(f"Error message from server: {data['msg']}")
            return

        if 'arg' in data and data['arg']['channel'] == 'economic-calendar':
            events = data.get('data', [])
            for event in events:
                self._handle_economic_event(event)
        else:
            print(f"[WebSocket] Received unhandled message: {message}")

    def _handle_economic_event(self, event: dict):
        """Processes a single economic event to decide whether to trade."""
        event_name = event.get('event', '')
        print(f"\n--- [WebSocket] Processing event: {event_name} ---")

        if not any(keyword.lower() in event_name.lower() for keyword in self.target_events):
            return

        actual_str = event.get('actual')
        forecast_str = event.get('forecast')

        if not actual_str or not forecast_str or actual_str == "" or forecast_str == "":
            print(f"Event '{event_name}' is missing an actual or forecast value. Skipping.")
            return

        try:
            actual = float(actual_str)
            forecast = float(forecast_str)
        except (ValueError, TypeError):
            print(f"Could not parse actual/forecast for '{event_name}' as float. Skipping.")
            return

        print(f"Relevant event found: {event_name} | Actual: {actual}, Forecast: {forecast}")

        z_score = calculate_z_score(event_name, actual, forecast)
        if z_score is None:
            return

        if abs(z_score) < 2.0:
            print(f"Z-score {z_score:.2f} is within threshold. No trade.")
            return

        print(f"Z-SCORE THRESHOLD EXCEEDED: |{z_score:.2f}| > 2.0")

        trade_side = None
        if "CPI" in event_name or "FOMC" in event_name: # FOMC hawkish surprise is also bad for BTC
            if z_score > 2.0:
                trade_side = 'sell'
            elif z_score < -2.0:
                trade_side = 'buy'
        else: # NFP logic can be context-dependent, for now let's assume strong NFP is good for USD, bad for BTC
             if z_score > 2.0: # Better than expected jobs report
                trade_side = 'sell'
             elif z_score < -2.0: # Worse than expected
                trade_side = 'buy'

        if trade_side:
            print(f"---!!! TRADE SIGNAL GENERATED !!!---")
            print(f"Instrument: {self.instrument}")
            print(f"Side: {trade_side.upper()}")
            print(f"Reason: Event '{event_name}' with Z-score of {z_score:.2f}")
            self.execute_trade_logic(trade_side)

    def execute_trade_logic(self, side: str):
        """Fetches market data, calculates position size, and executes the trade."""
        print("--- Initiating Trade Execution Logic ---")
        try:
            # 1. Fetch account balance
            balance = self.exchange.fetch_balance()
            account_equity = balance['USDT']['total']
            print(f"Account equity: ${account_equity:,.2f}")

            # 2. Fetch recent candles for ATR calculation
            print(f"Fetching recent {self.atr_timeframe} candles for ATR calculation...")
            ohlcv = self.exchange.fetch_ohlcv(self.instrument, self.atr_timeframe, limit=20)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
            print(f"Latest 14-period ATR ({self.atr_timeframe}): {atr:.4f}")

            # 3. Get entry price and calculate stop loss
            ticker = self.exchange.fetch_ticker(self.instrument)
            entry_price = ticker['last']

            if side == 'buy':
                stop_loss_price = entry_price - (atr * self.atr_multiplier)
            else: # sell
                stop_loss_price = entry_price + (atr * self.atr_multiplier)

            print(f"Entry Price: ${entry_price:,.2f}, Calculated SL: ${stop_loss_price:,.2f}")

            # 4. Calculate position size
            self.exchange.load_markets()
            market = self.exchange.markets[self.instrument]
            contract_size = market['contractSize']

            num_contracts = calculate_position_size(
                account_equity, self.risk_percent, entry_price, stop_loss_price, contract_size
            )

            if not num_contracts or num_contracts <= 0:
                print("Calculated position size is zero or invalid. Aborting trade.")
                return

            print(f"Calculated Position Size: {num_contracts} contracts.")

            # 5. Place the market order
            print(f"Placing MARKET {side.upper()} order for {num_contracts} contracts...")
            market_order = self.exchange.create_market_order(self.instrument, side, num_contracts)
            print("Market order placed successfully:")
            print(json.dumps(market_order, indent=2))

            # 6. Place the stop-loss order
            sl_side = 'sell' if side == 'buy' else 'buy'
            # For OKX, stop market orders are placed by specifying the trigger price in the params
            # and setting the order type to 'market'. The 'stop' type in create_order is a unified ccxt concept.
            # We will use the 'market' type with stop parameters.
            # The most reliable way is to specify the order type in params for clarity.
            sl_params = {
                'tdMode': 'cross',
                'slTriggerPx': str(stop_loss_price),
                'slOrdPx': '-1',  # A value of -1 indicates a market order for the stop loss
            }
            print(f"Placing STOP MARKET order with trigger at ${stop_loss_price:,.2f}...")
            # Note: CCXT unifies this. A 'stop' order with no price becomes a stop-market.
            # Let's use the most explicit and correct params for OKX.
            stop_order = self.exchange.create_order(
                self.instrument, 'market', sl_side, num_contracts, params=sl_params
            )
            print("Stop-loss order placed successfully:")
            print(json.dumps(stop_order, indent=2))

            # 7. Start the time-based exit timer
            exit_seconds = self.time_exit_minutes * 60
            print(f"Starting {self.time_exit_minutes}-minute time-based exit timer...")
            timer = threading.Timer(exit_seconds, self.close_position_by_timer, args=[self.instrument, market_order['id']])
            self.active_trade_timers[market_order['id']] = timer
            timer.start()

        except ccxt.BaseError as e:
            print(f"An error occurred during trade execution: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def close_position_by_timer(self, instrument_id: str, original_order_id: str):
        """Closes an open position when the timer expires."""
        print(f"\n--- [TIMER] Time-based exit triggered for order {original_order_id} on {instrument_id} ---")
        try:
            # Check current positions
            positions = self.exchange.fetch_positions([instrument_id])
            position_to_close = None
            for pos in positions:
                if pos.get('instrument') == instrument_id and float(pos.get('contracts', 0)) != 0:
                    position_to_close = pos
                    break

            if position_to_close:
                contracts = float(position_to_close['contracts'])
                side = 'sell' if contracts > 0 else 'buy'
                print(f"Position found: {contracts} contracts. Placing closing MARKET {side.upper()} order.")

                # For one-way mode (default), no special params are needed to close.
                # Simply sending an order for the opposite side is sufficient.
                closing_order = self.exchange.create_market_order(instrument_id, side, abs(contracts))
                print("Position closed successfully:")
                print(json.dumps(closing_order, indent=2))
            else:
                print("No open position found for this instrument. It was likely closed by the stop-loss.")

        except ccxt.BaseError as e:
            print(f"An error occurred during time-based exit: {e}")
        finally:
            # Clean up the timer
            if original_order_id in self.active_trade_timers:
                del self.active_trade_timers[original_order_id]

    def _on_open(self, ws):
        print("WebSocket connection opened. Subscribing to economic-calendar channel...")
        ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "economic-calendar"}]}))

    def _on_error(self, ws, error):
        print(f"WebSocket Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket connection closed. Status: {close_status_code}, Msg: {close_msg}")
        self.is_running = False

    def start(self):
        print("Starting WebSocket client...")
        self.ws = websocket.WebSocketApp(self._ws_url, on_open=self._on_open, on_message=self._on_message, on_error=self._on_error, on_close=self._on_close)
        self.is_running = True
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True
        self.thread.start()

        ping_thread = threading.Thread(target=self._send_pings)
        ping_thread.daemon = True
        ping_thread.start()

    def _send_pings(self):
        while self.is_running:
            time.sleep(25)
            if self.ws and self.ws.sock and self.ws.sock.connected:
                try:
                    self.ws.send("ping")
                    print("[WebSocket] Sent ping.")
                except Exception as e:
                    print(f"Failed to send ping: {e}")
                    break
            else:
                print("[WebSocket] Socket not connected, stopping pings.")
                break

    def stop(self):
        if self.is_running and self.ws:
            print("Stopping WebSocket client...")
            self.is_running = False
            # Cancel any active timers
            for timer in self.active_trade_timers.values():
                timer.cancel()
            self.active_trade_timers.clear()

            self.ws.close()
            if self.thread.is_alive():
                self.thread.join()
            print("WebSocket client stopped.")

if __name__ == '__main__':
    print("--- Running NewsWebsocketClient Standalone Test ---")

    # This test requires API keys with trade permissions to be set as environment variables
    # for the authenticated parts to work.

    # Example config
    trade_config = {
        "instrument": "BTC-USDT-SWAP",
        "target_events": ["CPI", "FOMC", "Non-Farm Payrolls"],
        "risk_percent": 0.01, # 1%
        "atr_multiplier": 2.5,
        "time_exit_minutes": 30
    }

    try:
        # The ccxt library automatically loads API keys from environment variables
        # (OKX_APIKEY, OKX_SECRET, OKX_PASSWORD)
        exchange = ccxt.okx({
            'options': {
                'defaultType': 'swap',
            },
        })
        # Switch to demo trading for safety
        exchange.set_sandbox_mode(True)
        print("CCXT exchange instance created in SANDBOX mode.")

    except Exception as e:
        print(f"Could not initialize CCXT exchange. Ensure API keys are set. Error: {e}")
        exchange = None

    if exchange:
        client = NewsWebsocketClient(exchange=exchange, config=trade_config)
        client.start()

        try:
            while client.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            client.stop()
            print("\nProgram terminated.")
