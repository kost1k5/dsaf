import requests
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import func
from functools import reduce

from src.database.models import MacroData
from src.core.config import settings

class MacroDataCollector:
    """
    A class to collect macroeconomic data using the FRED API.
    """
    def __init__(self, db_session: Session):
        self.db = db_session
        self.api_key = settings.FRED_API_KEY
        self.api_url = "https://api.stlouisfed.org/fred/series/observations"
        # Mapping of desired symbols to their FRED Series IDs
        self.series_map = {
            'SPY': 'SP500',      # S&P 500 Index
            'VIX': 'VIXCLS',     # VIX Volatility Index
            'DXY': 'DTWEXBGS'  # Trade Weighted U.S. Dollar Index
        }

    def get_latest_macro_timestamp(self) -> datetime:
        """
        Gets the timestamp of the most recent macro data entry in the database.
        """
        latest_macro = self.db.query(func.max(MacroData.date)).scalar()
        return latest_macro

    def save_macro_data_to_db(self, macro_df: pd.DataFrame) -> int:
        """
        Saves macro data to the database, ignoring duplicates.
        Returns the number of new rows inserted.
        """
        if macro_df.empty:
            return 0

        records = []
        for timestamp, row in macro_df.iterrows():
            records.append({
                "date": timestamp.to_pydatetime(),
                "spy_close": row.get('SPY'),
                "vix_close": row.get('VIX'),
                "dxy_close": row.get('DXY')
            })

        if not records:
            return 0

        stmt = insert(MacroData).values(records)
        stmt = stmt.on_conflict_do_nothing(index_elements=['date'])
        self.db.execute(stmt)
        self.db.commit()
        return len(records)

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for SPY, VIX, and DXY from the FRED API.

        :param start_date: The start date in 'YYYY-MM-DD' format.
        :param end_date: The end date in 'YYYY-MM-DD' format.
        :return: A pandas DataFrame with prices for the tickers.
        """
        if not self.api_key:
            print("Warning: FRED_API_KEY is not set. Skipping macroeconomic data collection.")
            return pd.DataFrame()

        all_series_dfs = []
        for symbol, series_id in self.series_map.items():
            print(f"Fetching FRED data for {symbol} (Series ID: {series_id})...")
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
                "sort_order": "asc",
            }
            try:
                response = requests.get(self.api_url, params=params)
                response.raise_for_status()  # Raise an exception for bad status codes
                data = response.json()

                observations = data.get("observations", [])
                if not observations:
                    print(f"  -> No observations found for {symbol}.")
                    continue

                df = pd.DataFrame(observations)
                df = df[['date', 'value']]
                df['date'] = pd.to_datetime(df['date'])
                # FRED data sometimes contains '.' for missing values, convert to NaN
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.rename(columns={'value': symbol}).set_index('date')
                all_series_dfs.append(df)
                print(f"  -> Successfully fetched {len(df)} data points for {symbol}.")

            except requests.exceptions.RequestException as e:
                print(f"  -> An error occurred while fetching data for {symbol}: {e}")
                continue
            except Exception as e:
                print(f"  -> An unexpected error occurred for {symbol}: {e}")
                continue

        if not all_series_dfs:
            return pd.DataFrame()

        # Merge all dataframes on the date index
        # Start with the first dataframe and outer join the rest onto it
        merged_df = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), all_series_dfs)

        # FRED data is daily, so forward-fill to cover weekends/holidays for a consistent dataset
        merged_df = merged_df.ffill()

        print(f"Successfully merged data for all symbols. Final shape: {merged_df.shape}")
        return merged_df

from src.database.session import SessionLocal

if __name__ == '__main__':
    db_session = SessionLocal()
    collector = MacroDataCollector(db_session)

    # Example: Fetch data for the last year
    end_dt = datetime.now()
    start_dt = end_dt.replace(year=end_dt.year - 1)

    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')

    macro_df = collector.fetch_data(start_date=start_str, end_date=end_str)

    if not macro_df.empty:
        print("\n--- Fetched Macro Data (Head) ---")
        print(macro_df.head())
        print("\n--- Data Info ---")
        macro_df.info()
        print("\n--- Check for remaining NaNs ---")
        print(macro_df.isnull().sum())
        print("\n--- Fetched Macro Data (Tail) ---")
        print(macro_df.tail())
