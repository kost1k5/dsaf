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


from sqlalchemy import func

class DataCollector:
    def __init__(self, exchange_id: str = 'okx', db_session: Session = None):
        """
        Initializes the DataCollector with a specific exchange and a database session.
        :param exchange_id: The ID of the exchange to use (e.g., 'binance', 'bybit').
        :param db_session: An active SQLAlchemy session.
        """
        self.db = db_session
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'options': {
                    'defaultType': 'swap',  # Changed from 'spot' to 'swap' for perpetuals
                },
                'timeout': 30000,  # 30-second timeout for exchange requests
            })
        except AttributeError:
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.")

        # Load markets to get symbol information
        self.exchange.load_markets()

    def fetch_candles(self, symbol: str, timeframe: str = '1h', since: int = None, limit: int = 100) -> List[list]:
        """
        Fetches historical OHLCV data for a given symbol.
        (A) Includes retry logic for network robustness.
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

        # print(f"Fetching {limit} candles for {symbol} on timeframe {timeframe}...") # Becomes too verbose
        retries = 3
        for i in range(retries):
            try:
                # CCXT returns data in a list of lists format: [timestamp, open, high, low, close, volume]
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                return ohlcv
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                print(f"Attempt {i + 1}/{retries} for fetch_candles failed: {e}. Retrying in 5 seconds...")
                time.sleep(5)

        # If all retries fail, raise an exception to be handled by the calling function.
        raise ExchangeError(f"Failed to fetch candles for {symbol} after {retries} retries.")

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

        # Filter out any candles that might be outside the end date and remove duplicates
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if not df.empty:
            df = df[df['timestamp'] <= end]
            df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)

        final_candles = df.values.tolist()
        print(f"Total unique candles fetched: {len(final_candles)}")
        return final_candles

    def get_latest_candle_timestamp(self, symbol: str, interval: str) -> int:
        """
        Gets the timestamp of the most recent candle for a given symbol and interval.
        """
        if not self.db:
            return None

        latest_candle_time = (
            self.db.query(func.max(Candle.open_time))
            .filter(Candle.symbol == symbol, Candle.interval == interval)
            .scalar()
        )
        if latest_candle_time:
            # Ensure the datetime object is timezone-aware before getting the timestamp
            if latest_candle_time.tzinfo is None:
                latest_candle_time = latest_candle_time.replace(tzinfo=datetime.timezone.utc)
            return int(latest_candle_time.timestamp() * 1000)
        return None


    def save_candles_to_db(self, candles: List[list], symbol: str, interval: str, batch_size: int = 1500) -> int:
        """
        Saves a list of OHLCV candles to the database in batches.
        It ignores duplicates based on the unique constraint (symbol, interval, open_time).
        Returns the number of new rows inserted.
        """
        if not candles:
            return 0
        if not self.db:
            raise Exception("Database session not provided to DataCollector.")

        from sqlalchemy.dialects.sqlite import insert

        unique_candles = {}
        for c in candles:
            unique_candles[c[0]] = c

        candle_dicts = [
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
            for c in unique_candles.values()
        ]

        if not candle_dicts:
            return 0

        stmt = insert(Candle).on_conflict_do_nothing(
            index_elements=['symbol', 'interval', 'open_time']
        )

        total_inserted = 0
        for i in range(0, len(candle_dicts), batch_size):
            chunk = candle_dicts[i:i + batch_size]
            try:
                self.db.execute(stmt, chunk)
                self.db.commit()
                # result.rowcount is not reliable across all DBs/versions,
                # especially with 'ON CONFLICT'. We'll count the attempted rows.
                total_inserted += len(chunk)
            except Exception as e:
                print(f"Error inserting candle batch for {symbol} ({interval}): {e}")
                self.db.rollback()
                raise
        return total_inserted

    def fetch_funding_rate_history(self, symbol: str, since: int = None, limit: int = 100, params={}) -> List[dict]:
        """
        Fetches historical funding rate data for a given symbol.
        (A) Includes retry logic for network robustness.
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

        # print(f"Fetching {limit} funding rates for {symbol}...") # Becomes too verbose
        retries = 3
        for i in range(retries):
            try:
                funding_rates = self.exchange.fetch_funding_rate_history(symbol, since, limit, params)
                return funding_rates
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                print(f"Attempt {i + 1}/{retries} for fetch_funding_rate_history failed: {e}. Retrying in 5 seconds...")
                time.sleep(5)

        raise ExchangeError(f"Failed to fetch funding rates for {symbol} after {retries} retries.")

    def fetch_full_funding_rate_history(self, instrument_family: str, start_date_str: str, end_date_str: str) -> List[dict]:
        """
        Fetches the complete funding rate history for a given instrument family over a date range
        by using the OKX historical market data download endpoint. This is necessary because
        the standard CCXT `fetchFundingRateHistory` endpoint for OKX is limited to 3 months.
        This method handles the API limitations by fetching data in monthly chunks.
        :param instrument_family: The instrument family, e.g., 'BTC-USDT'.
        :param start_date_str: The start date in 'YYYY-MM-DD' format.
        :param end_date_str: The end date in 'YYYY-MM-DD' format.
        :return: A list of funding rate data dictionaries, compatible with CCXT format.
        """
        print(f"Attempting to download full funding rate history for {instrument_family} from {start_date_str} to {end_date_str}...")

        try:
            start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError:
            print("Error: Invalid date format. Please use 'YYYY-MM-DD'.")
            return []

        # --- 1. Get the list of file URLs by iterating month by month ---
        base_url = "https://www.okx.com/api/v5/public/market-data-history"
        download_urls = set() # Use a set to avoid duplicate URLs
        current_start = start_date

        while current_start <= end_date:
            # Set the end of the chunk to the last day of the current month
            next_month = current_start.replace(day=28) + datetime.timedelta(days=4)
            chunk_end = next_month - datetime.timedelta(days=next_month.day)

            if chunk_end > end_date:
                chunk_end = end_date

            # Per documentation, timestamps are treated as UTC+8.
            # We will use simple timestamps and let the server handle timezone conversion.
            begin_ms = int(current_start.timestamp() * 1000)
            end_ms = int(chunk_end.timestamp() * 1000)

            params = {
                "module": "3", # 3 = funding_rate
                "instType": "SWAP",
                "instFamilyList": instrument_family,
                "dateAggrType": "monthly", # Changed from 'daily' to 'monthly'
                "begin": begin_ms,
                "end": end_ms,
            }

            try:
                print(f"Requesting file list for {current_start.strftime('%Y-%m')}...")
                response = requests.get(base_url, params=params, timeout=(10, 30)) # (connect, read)
                response.raise_for_status()
                data = response.json()

                if data.get("code") == "0" and data.get("data"):
                    if data["data"][0].get("details"):
                        for detail in data["data"][0]["details"]:
                            for group in detail.get("groupDetails", []):
                                download_urls.add(group["url"])
                elif data.get("code") == "52000": # Specific code for "No market data found"
                    print(f"Info: No monthly data file found for {current_start.strftime('%Y-%m')}. This is expected for some periods.")
                else:
                    print(f"API Warning/Error for chunk {current_start.strftime('%Y-%m')}: {data.get('msg', 'No data returned')}")

            except requests.exceptions.RequestException as e:
                print(f"Error fetching file list for chunk {current_start.strftime('%Y-%m')}: {e}")

            # Move to the start of the next month robustly
            current_start = (current_start.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
            time.sleep(1) # Be polite

        if not download_urls:
            print("No downloadable files found for the entire period.")
            return []

        print(f"Found {len(download_urls)} unique files. Starting download and processing...")

        # --- 2. Download and process each file ---
        all_data_frames = []
        for url in sorted(list(download_urls)): # Sort to process chronologically
            try:
                print(f"  Downloading: {url.split('/')[-1]}")
                file_response = requests.get(url, stream=True, timeout=(10, 60)) # (connect, read)
                file_response.raise_for_status()

                with zipfile.ZipFile(io.BytesIO(file_response.content)) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as csv_file:
                        df = pd.read_csv(csv_file)
                        all_data_frames.append(df)
            except Exception as e:
                print(f"  Failed to download or process file {url}. Error: {e}")
                continue

        if not all_data_frames:
            print("Data processing failed for all files.")
            return []

        # --- 3. Consolidate and format the data ---
        full_history_df = pd.concat(all_data_frames, ignore_index=True)
        print(f"DEBUG: DataFrame columns are: {full_history_df.columns.tolist()}")

        # (FIX) Rename columns to match the expected format, as CSV headers are different.
        # CSV: 'instrument_name', 'funding_rate', 'funding_time'
        # Expected: 'instId', 'fundingRate', 'fundingTime'
        if not full_history_df.empty:
            full_history_df.rename(columns={
                'instrument_name': 'instId',
                'funding_rate': 'fundingRate',
                'funding_time': 'fundingTime'
            }, inplace=True)

        full_history_df.drop_duplicates(subset=['fundingTime', 'instId'], keep='first', inplace=True)


        # Convert to a list of dicts in a format similar to CCXT
        ccxt_formatted_data = []
        for _, row in full_history_df.iterrows():
            ts = int(row['fundingTime'])
            ccxt_formatted_data.append({
                'symbol': f"{instrument_family}-SWAP",
                'timestamp': ts,
                'datetime': datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc).isoformat(),
                'fundingRate': float(row['fundingRate']),
                'info': row.to_dict()
            })

        ccxt_formatted_data.sort(key=lambda x: x['timestamp'])

        print(f"Successfully processed {len(ccxt_formatted_data)} total funding rate records.")
        return ccxt_formatted_data


    def get_latest_funding_rate_timestamp(self, symbol: str) -> int:
        """
        Gets the timestamp of the most recent funding rate for a given symbol.
        """
        if not self.db:
            return None

        latest_fr_time = (
            self.db.query(func.max(FundingRate.funding_time))
            .filter(FundingRate.symbol == symbol)
            .scalar()
        )
        if latest_fr_time:
            if latest_fr_time.tzinfo is None:
                latest_fr_time = latest_fr_time.replace(tzinfo=datetime.timezone.utc)
            return int(latest_fr_time.timestamp() * 1000)
        return None

    def save_funding_rates_to_db(self, funding_rates: List[dict], symbol: str, batch_size: int = 1500) -> int:
        """
        Saves a list of funding rates to the database in batches.
        Returns the number of new rows inserted.
        """
        if not funding_rates:
            return 0
        if not self.db:
            raise Exception("Database session not provided to DataCollector.")

        from sqlalchemy.dialects.sqlite import insert

        unique_rates = {}
        for fr in funding_rates:
            unique_rates[fr['timestamp']] = fr

        fr_dicts = [
            {
                "symbol": symbol,
                "instrument_type": fr.get('info', {}).get('instType', 'SWAP'),
                "funding_time": datetime.datetime.fromtimestamp(fr['timestamp'] / 1000, tz=datetime.timezone.utc),
                "funding_rate": fr['fundingRate']
            }
            for fr in unique_rates.values()
        ]

        if not fr_dicts:
            return 0

        stmt = insert(FundingRate).on_conflict_do_nothing(
            index_elements=['symbol', 'funding_time']
        )

        total_inserted = 0
        for i in range(0, len(fr_dicts), batch_size):
            chunk = fr_dicts[i:i + batch_size]
            try:
                self.db.execute(stmt, chunk)
                self.db.commit()
                # result.rowcount is not reliable. Count the attempted rows instead.
                total_inserted += len(chunk)
            except Exception as e:
                print(f"Error inserting funding rate batch for {symbol}: {e}")
                self.db.rollback()
                raise
        return total_inserted


    def fetch_open_interest_history(self, symbol: str, timeframe: str = '1h', since: int = None, limit: int = 100, params={}) -> List[dict]:
        """
        Fetches historical open interest data for a given symbol.
        (A) Includes retry logic for network robustness.
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

        # print(f"Fetching {limit} open interest data points for {symbol} on timeframe {timeframe}...") # Becomes too verbose
        retries = 3
        for i in range(retries):
            try:
                open_interest = self.exchange.fetch_open_interest_history(symbol, timeframe, since, limit, params)
                return open_interest
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                print(f"Attempt {i + 1}/{retries} for fetch_open_interest_history failed: {e}. Retrying in 5 seconds...")
                time.sleep(5)

        raise ExchangeError(f"Failed to fetch open interest for {symbol} after {retries} retries.")



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
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=3)
    since_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    print(f"\n--- Testing OHLCV for {symbol_swap} ---")
    try:
        candles_data = collector.fetch_candles(symbol_swap, timeframe='1h', limit=5)
        for candle in candles_data:
            dt_object = datetime.datetime.fromtimestamp(candle[0] / 1000)
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
