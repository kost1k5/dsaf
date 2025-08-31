import sqlite3
import pandas as pd
from datetime import datetime
import os
from typing import Optional

from .collector import DataCollector


class DataCacher:
    """
    Handles caching of historical candle data in a local SQLite database
    to minimize redundant data fetching from the exchange.
    """
    def __init__(self, db_path='data/historical_data.db'):
        """
        Initializes the cacher and ensures the database and table exist.
        :param db_path: Path to the SQLite database file.
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        """
        Creates the 'candles' table if it doesn't already exist.
        The table schema is designed to store OHLCV data for different symbols and timeframes.
        A UNIQUE constraint on symbol, timeframe, and open_time prevents duplicate entries.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(symbol, timeframe, open_time)
            )
        ''')
        self.conn.commit()

    def save_candles(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Saves a DataFrame of candle data to the database.
        'on conflict do nothing' ensures that existing candles are not duplicated.
        """
        if df.empty:
            return

        cursor = self.conn.cursor()
        df_to_save = df.copy()
        df_to_save['symbol'] = symbol
        df_to_save['timeframe'] = timeframe

        # Ensure open_time is a Unix timestamp (integer)
        if pd.api.types.is_datetime64_any_dtype(df_to_save['open_time']):
            df_to_save['open_time'] = (df_to_save['open_time'].astype(int) / 10**9).astype(int)

        tuples_to_insert = [tuple(x) for x in df_to_save[['symbol', 'timeframe', 'open_time', 'open', 'high', 'low', 'close', 'volume']].to_numpy()]

        cursor.executemany('''
            INSERT INTO candles (symbol, timeframe, open_time, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe, open_time) DO NOTHING
        ''', tuples_to_insert)
        self.conn.commit()
        print(f"Saved/updated {len(tuples_to_insert)} candles for {symbol} {timeframe} in cache.")

    def get_candles(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Retrieves cached candle data for a given symbol, timeframe, and date range.
        """
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        query = '''
            SELECT open_time, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND timeframe = ? AND open_time >= ? AND open_time <= ?
            ORDER BY open_time ASC
        '''
        df = pd.read_sql_query(query, self.conn, params=(symbol, timeframe, start_ts, end_ts))

        if not df.empty:
            df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
            df.set_index('open_time', inplace=True)

        return df

    def get_latest_cached_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        """
        Finds the most recent timestamp for a given symbol and timeframe in the cache.
        Returns a Unix timestamp integer or None if no data exists.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MAX(open_time) FROM candles
            WHERE symbol = ? AND timeframe = ?
        ''', (symbol, timeframe))
        result = cursor.fetchone()[0]
        return result

    def fetch_and_cache_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Orchestrates fetching data. It gets as much as possible from the cache,
        then fetches any missing data from the exchange and updates the cache.
        """
        print(f"Fetching data for {symbol} ({timeframe}) from {start_date} to {end_date}")

        # 1. Get the latest timestamp from our cache
        latest_cached_ts = self.get_latest_cached_timestamp(symbol, timeframe)

        # 2. Determine the start time for the new data fetch
        # The 'since' parameter for ccxt fetch_ohlcv is the start of the timeframe, not the end.
        # So if the last candle is for 10:00, we need to fetch from 10:00 onwards.
        fetch_since_ts = None
        if latest_cached_ts:
            fetch_since_ts = latest_cached_ts * 1000 # Collector expects milliseconds
            print(f"Latest cached data found at {datetime.fromtimestamp(latest_cached_ts)}. Fetching new data since then.")
        else:
            # No data in cache, fetch everything from the user-specified start_date
            fetch_since_ts = int(start_date.timestamp() * 1000)
            print("No cached data found. Performing initial fetch.")

        # 3. Fetch new data from the exchange if needed
        collector = DataCollector(exchange_id='okx')

        # Only fetch if the desired end date is after our last cached point
        # or if there is no cached data at all.
        if fetch_since_ts < end_date.timestamp() * 1000:
            print(f"Fetching from exchange, since {datetime.fromtimestamp(fetch_since_ts / 1000)}...")
            new_candles_list = collector.fetch_candles_in_range(
                symbol=symbol,
                timeframe=timeframe,
                since=fetch_since_ts,
                end=int(end_date.timestamp() * 1000)
            )

            if new_candles_list:
                new_candles_df = pd.DataFrame(new_candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
                new_candles_df['open_time'] = pd.to_datetime(new_candles_df['open_time'], unit='ms')

                self.save_candles(new_candles_df, symbol, timeframe)
            else:
                print("No new candles fetched from the exchange.")
        else:
            print("Cache is already up to date for the requested range.")

        # 4. Retrieve the complete, up-to-date data from the cache
        print("Retrieving complete dataset from cache...")
        return self.get_candles(symbol, timeframe, start_date, end_date)

    def close(self):
        """Closes the database connection."""
        self.conn.close()
