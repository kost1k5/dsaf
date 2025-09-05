import ccxt
import datetime
import time
from typing import List
import requests
import zipfile
import io
import pandas as pd


from sqlalchemy.orm import Session
from src.database.session import SessionLocal
from src.database.models import Candle, FundingRate
from ccxt.base.errors import NotSupported, ExchangeError


class DataCollector:
    def __init__(self, exchange_id: str = 'okx'):
        """
        Initializes the DataCollector with a specific exchange.
        :param exchange_id: The ID of the exchange to use (e.g., 'binance', 'bybit').
        """
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'options': {
                    'defaultType': 'swap',  # Changed from 'spot' to 'swap' for perpetuals
                },
            })
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
        # print(f"Fetched {len(ohlcv)} candles.") # This becomes too verbose in a loop
        return ohlcv

    def fetch_candles_in_range(self, symbol: str, timeframe: str, since: int, end: int) -> List[list]:
        """
        Fetches all historical OHLCV data for a given symbol in a specific date range.
        It handles pagination by repeatedly calling fetch_candles.
        :param symbol: The trading symbol.
        :param timeframe: The timeframe for the candles.
        :param since: The starting time in milliseconds since the epoch.
        :param end: The ending time in milliseconds since the epoch.
        :return: A list of all OHLCV candles in the range.
        """
        all_candles = []
        current_since = since
        timeframe_duration_ms = self.exchange.parse_timeframe(timeframe) * 1000
        print(f"Fetching all candles for {symbol} from {datetime.datetime.fromtimestamp(since/1000)} to {datetime.datetime.fromtimestamp(end/1000)}")

        while current_since < end:
            try:
                candles = self.fetch_candles(symbol, timeframe, current_since, limit=1000)
                if not candles:
                    print("No more candles returned from exchange. Stopping.")
                    break

                all_candles.extend(candles)
                last_timestamp = candles[-1][0]
                print(f"  Fetched {len(candles)} candles, up to {datetime.datetime.fromtimestamp(last_timestamp/1000)}")

                # Check if we are stuck in a loop
                if last_timestamp >= current_since:
                    current_since = last_timestamp + timeframe_duration_ms
                else:
                    print("Timestamp did not advance. Breaking loop.")
                    break

                # Be polite to the API
                time.sleep(self.exchange.rateLimit / 1000)

            except ExchangeError as e:
                # Specific handling for OKX's "history data is not available" error
                if '50030' in str(e):
                    print(f"Info: Exchange returned 'no data available' (50030). Likely reached the end of available history for this asset.")
                else:
                    print(f"An exchange error occurred while fetching a chunk of data: {e}")
                break # Stop paginating on exchange errors, but keep the data we have.
            except Exception as e:
                print(f"A general error occurred while fetching a chunk of data: {e}")
                break # Also stop on general errors

        # Filter out any candles that might be outside the end date
        final_candles = [c for c in all_candles if c[0] <= end]
        print(f"Total candles fetched: {len(final_candles)}")
        return final_candles

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

    def fetch_funding_rate_history(self, symbol: str, since: int = None, limit: int = 100, params={}) -> List[dict]:
        """
        Fetches historical funding rate data for a given symbol.
        OKX specific: uses fetch_funding_rate_history
        :param symbol: The trading symbol (e.g., 'BTC-USDT-SWAP').
        :param since: The starting time in milliseconds since the epoch.
        :param limit: The number of candles to fetch. Max is 100 for this endpoint.
        :param params: Extra parameters to pass to the exchange API call.
        :return: A list of funding rate data, typically newest first.
        """
        if not self.exchange.has['fetchFundingRateHistory']:
            raise NotSupported(f"The exchange {self.exchange.id} does not support fetching funding rate history.")
        if symbol not in self.exchange.markets:
            raise ValueError(f"Symbol '{symbol}' not available on {self.exchange.id}")

        print(f"Fetching {limit} funding rates for {symbol}...")
        # Note: CCXT unified method often returns in reverse chronological order (newest first)
        funding_rates = self.exchange.fetch_funding_rate_history(symbol, since, limit, params)
        # We REMOVE sorting to respect the exchange's default order, which is required for backward pagination.
        return funding_rates


    def fetch_open_interest_history(self, symbol: str, timeframe: str = '1h', since: int = None, limit: int = 100, params={}) -> List[dict]:
        """
        Fetches historical open interest data for a given symbol.
        :param symbol: The trading symbol (e.g., 'BTC-USDT-SWAP').
        :param timeframe: The timeframe for the data points (e.g., '5m', '1h', '4h', '1d').
        :param since: The starting time in milliseconds since the epoch.
        :param limit: The number of data points to fetch. Max is 100 for this endpoint.
        :param params: Extra parameters to pass to the exchange API call.
        :return: A list of open interest data points, typically newest first.
        """
        if not self.exchange.has['fetchOpenInterestHistory']:
            raise NotSupported(f"The exchange {self.exchange.id} does not support fetching open interest history.")
        if symbol not in self.exchange.markets:
            raise ValueError(f"Symbol '{symbol}' not available on {self.exchange.id}")

        print(f"Fetching {limit} open interest data points for {symbol} on timeframe {timeframe}...")
        open_interest = self.exchange.fetch_open_interest_history(symbol, timeframe, since, limit, params)
        # We REMOVE sorting to respect the exchange's default order, which is required for backward pagination.
        return open_interest



    def fetch_paginated_history(self, fetch_method, symbol: str, since: int, end: int, **kwargs) -> List[dict]:
        """
        Fetches historical data by paginating backwards from the end date.
        This is useful for endpoints with limited history like open interest or funding rates.
        :param fetch_method: The bound method to call for fetching data.
        :param symbol: The trading symbol.
        :param since: The earliest time in milliseconds to fetch data for.
        :param end: The latest time in milliseconds to fetch data for (the starting point).
        :param kwargs: Additional keyword arguments for the fetch_method (e.g., timeframe).
        :return: A list of all data points in the range, sorted chronologically.
        """
        all_data = []
        current_until = end
        print(f"Fetching all data for {symbol} BACKWARDS from {datetime.datetime.fromtimestamp(end/1000)} to {datetime.datetime.fromtimestamp(since/1000)}")

        while current_until > since:
            try:
                # We use the 'params' argument to pass 'until' to ccxt, which maps to OKX's 'before' parameter.
                # We fetch data *before* the current_until timestamp.
                data_chunk = fetch_method(symbol=symbol, since=None, limit=100, params={'until': current_until}, **kwargs)

                if not data_chunk:
                    print("No more data returned from exchange. Stopping backward pagination.")
                    break

                # To prevent getting stuck on APIs that ignore millisecond precision in 'until',
                # we manually filter out any data points we have already collected.
                existing_timestamps = {d['timestamp'] for d in all_data}
                unique_new_data = [d for d in data_chunk if d['timestamp'] not in existing_timestamps]

                if not unique_new_data:
                    print("No new data points returned in the latest chunk. Stopping backward pagination.")
                    break

                all_data.extend(unique_new_data)

                # The next 'until' should be based on the oldest timestamp in the unique new data we just found.
                oldest_ts_in_chunk = min(d['timestamp'] for d in unique_new_data)
                print(f"  Fetched {len(unique_new_data)} new data points, back to {datetime.datetime.fromtimestamp(oldest_ts_in_chunk/1000)}")

                # Subtract 1ms to make the next request exclusive of the last record
                current_until = oldest_ts_in_chunk - 1

                # Be polite to the API
                time.sleep(self.exchange.rateLimit / 1000)

            except ExchangeError as e:
                if '50030' in str(e):
                    print(f"Info: Reached the end of available history for this asset (50030).")
                else:
                    print(f"An exchange error occurred while fetching a chunk of data: {e}")
                break # Stop paginating on exchange errors, but keep the data we have.
            except Exception as e:
                print(f"A general error occurred while fetching a chunk of data: {e}")
                break

        # Final filter and sort chronologically before returning
        final_data = [d for d in all_data if since <= d['timestamp'] <= end]
        print(f"Total data points fetched: {len(final_data)}")
        return sorted(final_data, key=lambda x: x['timestamp'])


# Example usage:
if __name__ == '__main__':
    collector = DataCollector(exchange_id='okx')
    symbol_swap = 'BTC/USDT:USDT'
    # Define a time range for the last 3 days
    end_time = datetime.datetime.now(datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(days=3)
    since_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    print(f"\n--- Testing OHLCV for {symbol_swap} ---")
    try:
        candles_data = collector.fetch_candles(symbol_swap, timeframe='1h', limit=5)
        for candle in candles_data:
            dt_object = datetime.datetime.fromtimestamp(candle[0] / 1000, tz=datetime.timezone.utc)
            print(f"Time: {dt_object}, Open: {candle[1]}, Close: {candle[4]}")
    except (ValueError, NotSupported) as e:
        print(f"An error occurred: {e}")

    print(f"\n--- Testing Funding Rate History for {symbol_swap} ---")
    try:
        fr_data = collector.fetch_paginated_history(
            collector.fetch_funding_rate_history,
            symbol=symbol_swap,
            since=since_ms,
            end=end_ms
        )
        if fr_data:
            print(f"Fetched {len(fr_data)} funding rate entries. First 3:")
            for fr in fr_data[:3]:
                print(fr)
    except (ValueError, NotSupported) as e:
        print(f"An error occurred: {e}")

    print(f"\n--- Testing Open Interest History for {symbol_swap} ---")
    try:
        oi_data = collector.fetch_paginated_history(
            collector.fetch_open_interest_history,
            symbol=symbol_swap,
            since=since_ms,
            end=end_ms,
            timeframe='1h' # timeframe is a required kwarg for this method
        )
        if oi_data:
            print(f"Fetched {len(oi_data)} open interest entries. First 3:")
            for oi in oi_data[:3]:
                print(oi)
    except (ValueError, NotSupported) as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
