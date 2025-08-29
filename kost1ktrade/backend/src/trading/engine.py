import ccxt
from src.core.config import settings

class TradingEngine:
    def __init__(self, exchange_id: str = 'kraken'):
        """
        Initializes the TradingEngine with a specific exchange and credentials.
        """
        if not settings.API_KEY or not settings.API_SECRET:
            raise ValueError("API_KEY and API_SECRET must be set in the .env file for live trading.")

        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': settings.API_KEY,
                'secret': settings.API_SECRET,
            })
            # For some exchanges, a sandbox mode can be enabled like this:
            # self.exchange.set_sandbox_mode(True)
        except AttributeError:
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize exchange: {e}")

    def get_balance(self, currency: str = 'USD'):
        """
        Fetches the balance for a specific currency.
        :param currency: The currency symbol (e.g., 'USD', 'USDT', 'BTC').
        :return: A dictionary with balance information or None if an error occurs.
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance.get(currency, {'free': 0, 'used': 0, 'total': 0})
        except ccxt.Error as e:
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

# Example usage (for demonstration purposes, will not run without valid API keys)
if __name__ == '__main__':
    print("--- Trading Engine Demonstration ---")
    print("NOTE: This script requires valid API_KEY and API_SECRET in a .env file to run.")

    try:
        engine = TradingEngine(exchange_id='kraken')

        # 1. Get balance
        balance = engine.get_balance('USD')
        if balance:
            print(f"\nFetched balance for USD: {balance}")

        # 2. Create a dummy limit order (this will likely fail without funds)
        # This is a small amount to avoid issues on a live account.
        # IMPORTANT: Do not run with large amounts on a real account without thorough testing.
        dummy_symbol = 'BTC/USD'
        dummy_amount = 0.0001
        # Set a price far from the current market to ensure it's not filled instantly
        dummy_price = 10000.0

        print(f"\nAttempting to create a dummy limit order for {dummy_symbol}...")
        order = engine.create_order(dummy_symbol, 'limit', 'buy', dummy_amount, dummy_price)

        # 3. Cancel the dummy order
        if order:
            engine.cancel_order(order['id'], dummy_symbol)

    except (ValueError, ConnectionError) as e:
        print(f"\nInitialization failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
