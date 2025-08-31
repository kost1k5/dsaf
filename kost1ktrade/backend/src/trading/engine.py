import ccxt
import asyncio
from src.core.config import settings
from src.notifications.telegram import send_telegram_notification

class TradingEngine:
    def __init__(self, mode: str, exchange_id: str = 'okx'):
        """
        Initializes the TradingEngine in a specific mode ('real' or 'demo').
        """
        self.mode = mode

        if self.mode == 'real':
            keys = settings.OKX_REAL
            if not keys:
                raise ValueError("OKX_REAL keys are not configured in the .env file.")
        elif self.mode == 'demo':
            keys = settings.OKX_DEMO
            if not keys:
                raise ValueError("OKX_DEMO keys are not configured in the .env file.")
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'real' or 'demo'.")

        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': keys.API_KEY,
                'secret': keys.SECRET_KEY,
                'password': keys.PASSPHRASE, # Passphrase for OKX
            })

            if self.mode == 'demo':
                self.exchange.set_sandbox_mode(True)

        except AttributeError:
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize exchange '{exchange_id}' in '{self.mode}' mode: {e}")

    def get_balance(self):
        """
        Fetches the entire account balance.
        :return: A dictionary of balances or None if an error occurs.
        """
        try:
            # We are interested in the 'free' balances of assets with a non-zero amount
            balance = self.exchange.fetch_balance()
            return {
                currency: data['free']
                for currency, data in balance.items()
                if data['free'] > 0
            }
        except ccxt.errors.BaseError as e:
            print(f"An error occurred while fetching balance: {e}")
            return None

    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None):
        """
        Creates a new order on the exchange.
        :param symbol: The trading symbol (e.g., 'BTC/USD').
        :param order_type: 'market' or 'limit'.
        :param side: 'buy' or 'sell'.
        :param amount: The quantity of the asset to trade.
        :param price: The price for a limit order.
        :return: The order object from the exchange or None if an error occurs.
        """
        try:
            print(f"Creating {side} {order_type} order for {amount} {symbol}...")
            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            print("Order created successfully:")
            print(order)

            # Send notification
            side_emoji = "📈" if side == 'buy' else "📉"
            message = (
                f"{side_emoji} *New Trade Executed ({self.mode.upper()})*\n\n"
                f"**Symbol:** `{symbol}`\n"
                f"**Side:** `{side.upper()}`\n"
                f"**Type:** `{order_type.upper()}`\n"
                f"**Amount:** `{order['amount']}`\n"
                f"**Price:** `${order['price'] if order['price'] else order['average']:.2f}`"
            )
            asyncio.run(send_telegram_notification(message))

            return order
        except ccxt.InsufficientFunds as e:
            print(f"Error: Insufficient funds to create order. {e}")
        except ccxt.OrderNotFound as e:
             print(f"Error: Order not found after creation (may have been filled instantly). {e}")
        except ccxt.ExchangeError as e:
            print(f"An exchange error occurred while creating order: {e}")
        return None

    def cancel_order(self, order_id: str, symbol: str):
        """
        Cancels an existing order.
        :param order_id: The ID of the order to cancel.
        :param symbol: The trading symbol is required by some exchanges.
        :return: True if successful, False otherwise.
        """
        try:
            print(f"Cancelling order {order_id} for {symbol}...")
            self.exchange.cancel_order(order_id, symbol)
            print("Order cancelled successfully.")
            return True
        except ccxt.OrderNotFound:
            print(f"Error: Order {order_id} not found, it may have already been filled or cancelled.")
        except ccxt.ExchangeError as e:
            print(f"An exchange error occurred while cancelling order: {e}")
        return False

    def fetch_open_orders(self, symbol: str) -> list:
        """
        Fetches all open orders for a specific symbol.
        :param symbol: The trading symbol (e.g., 'BTC/USDT').
        :return: A list of open order objects or an empty list if none or an error occurs.
        """
        try:
            return self.exchange.fetch_open_orders(symbol)
        except ccxt.errors.BaseError as e:
            print(f"An error occurred while fetching open orders: {e}")
            return []

    def cancel_all_orders(self, symbol: str):
        """
        Cancels all open orders for a specific symbol.
        :param symbol: The trading symbol to cancel orders for.
        :return: True if successful, False otherwise.
        """
        try:
            print(f"Cancelling all open orders for {symbol}...")
            self.exchange.cancel_all_orders(symbol)
            print("All orders for symbol cancelled successfully.")
            return True
        except ccxt.ExchangeError as e:
            print(f"An exchange error occurred while cancelling all orders: {e}")
        return False

    def fetch_ticker(self, symbol: str) -> dict:
        """
        Fetches the latest ticker data for a symbol.
        :param symbol: The trading symbol.
        :return: A dictionary with ticker information or an empty dict if an error occurs.
        """
        try:
            return self.exchange.fetch_ticker(symbol)
        except ccxt.errors.BaseError as e:
            print(f"An error occurred while fetching ticker for {symbol}: {e}")
            return {}

# Example usage (for demonstration purposes, will not run without valid API keys)
if __name__ == '__main__':
    print("--- Trading Engine Demonstration ---")
    print("NOTE: This script requires valid OKX_DEMO keys in a .env file to run.")

    try:
        # Initialize in 'demo' mode
        engine = TradingEngine(mode='demo', exchange_id='okx')
        print(f"Successfully initialized trading engine in '{engine.mode}' mode.")

        # 1. Get balance
        balances = engine.get_balance()
        if balances is not None:
            print(f"\nFetched balances: {balances}")
        else:
            print("Could not fetch balances. The API keys might be invalid or have incorrect permissions.")

        # 2. Fetch open orders
        print("\nFetching open orders for BTC/USDT...")
        open_orders = engine.fetch_open_orders('BTC/USDT')
        print(f"Found {len(open_orders)} open orders.")

    except (ValueError, ConnectionError) as e:
        print(f"\nInitialization or operation failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
