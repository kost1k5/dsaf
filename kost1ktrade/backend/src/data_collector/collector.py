import ccxt
import datetime
from typing import List

from sqlalchemy.orm import Session
from src.database.session import SessionLocal
from src.database.models import Candle
from ccxt.base.errors import NotSupported


class DataCollector:
    def __init__(self, exchange_id: str = 'okx'):
        """
        Initializes the DataCollector with a specific exchange.
        :param exchange_id: The ID of the exchange to use (e.g., 'binance', 'bybit').
        """
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class()
        except AttributeError:
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.")

        # Load markets to get symbol information
        self.exchange.load_markets()

    def fetch_candles(self, symbol: str, timeframe: str = '1h', since: int = None, limit: int = 100) -> List[list]:
        """
        Fetches historical OHLCV data for a given symbol.
        :param symbol: The trading symbol (e.g., 'BTC/USDT').
        :param timeframe: The timeframe for the candles (e.g., '1m', '5m', '1h', '1d').
        :param since: The starting time in milliseconds since the epoch.
        :param limit: The number of candles to fetch.
        :return: A list of OHLCV candles.
        """
        if not self.exchange.has['fetchOHLCV']:
            raise NotSupported("The selected exchange does not support fetching OHLCV data.")
        if symbol not in self.exchange.markets:
            raise ValueError(f"Symbol '{symbol}' not available on {self.exchange.id}")

        print(f"Fetching {limit} candles for {symbol} on timeframe {timeframe}...")
        # CCXT returns data in a list of lists format: [timestamp, open, high, low, close, volume]
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        print(f"Fetched {len(ohlcv)} candles.")
        return ohlcv

    def save_candles_to_db(self, candles: List[list], symbol: str, interval: str):
        """
        Saves a list of OHLCV candles to the database.
        It ignores duplicates based on the unique constraint (symbol, interval, open_time).
        :param candles: A list of OHLCV candles from ccxt.
        :param symbol: The trading symbol (e.g., 'BTC/USDT').
        :param interval: The timeframe for the candles (e.g., '1h').
        """
        if not candles:
            print("No candles to save.")
            return

        db: Session = SessionLocal()
        try:
            from sqlalchemy.dialects.postgresql import insert

            candle_dicts = []
            for c in candles:
                candle_dicts.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "open_time": datetime.datetime.fromtimestamp(c[0] / 1000, tz=datetime.timezone.utc),
                        "open": c[1],
                        "high": c[2],
                        "low": c[3],
                        "close": c[4],
                        "volume": c[5],
                    }
                )

            if not candle_dicts:
                return

            print(f"Attempting to save {len(candle_dicts)} candles to the database...")

            # Create an insert statement with ON CONFLICT DO NOTHING
            stmt = insert(Candle).values(candle_dicts)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=['symbol', 'interval', 'open_time']
            )

            db.execute(stmt)
            db.commit()
            print(f"Successfully processed save request for {len(candle_dicts)} candles for {symbol}.")

        except Exception as e:
            print(f"An error occurred while saving candles to the database: {e}")
            db.rollback()
        finally:
            db.close()

# Example usage:
if __name__ == '__main__':
    collector = DataCollector(exchange_id='okx')
    # Fetch the last 5 candles for BTC/USDT on the 1-hour timeframe
    try:
        candles_data = collector.fetch_candles('BTC/USDT', timeframe='1h', limit=5)
        for candle in candles_data:
            # Convert timestamp to human-readable format
            dt_object = datetime.datetime.fromtimestamp(candle[0] / 1000, tz=datetime.timezone.utc)
            print(f"Time: {dt_object}, Open: {candle[1]}, High: {candle[2]}, Low: {candle[3]}, Close: {candle[4]}, Volume: {candle[5]}")
    except (ValueError, NotSupported) as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
