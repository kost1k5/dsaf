import yfinance as yf
import pandas as pd
from datetime import datetime

class MacroDataCollector:
    """
    A class to collect macroeconomic data using the yfinance library.
    """
    def __init__(self):
        # Tickers for S&P 500 (SPY), VIX Index, and US Dollar Index (DXY)
        self.tickers = {
            'SPY': 'SPY',
            'VIX': '^VIX',
            'DXY': 'DX-Y.NYB'
        }

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for SPY, VIX, and DXY.

        :param start_date: The start date in 'YYYY-MM-DD' format.
        :param end_date: The end date in 'YYYY-MM-DD' format.
        :return: A pandas DataFrame with 'Adj Close' prices for the tickers.
        """
        print(f"Fetching macro data for tickers {list(self.tickers.values())} from {start_date} to {end_date}...")
        try:
            # Download data from yfinance
            data = yf.download(list(self.tickers.values()), start=start_date, end=end_date, progress=False)

            if data.empty:
                print("Warning: No data returned from yfinance.")
                return pd.DataFrame()

            # With auto_adjust=True (default), yfinance returns the adjusted close price in the 'Close' column.
            close_prices = data['Close']

            # Rename columns to be more intuitive (e.g., '^VIX' -> 'VIX')
            close_prices = close_prices.rename(columns={v: k for k, v in self.tickers.items()})

            # Forward-fill missing values, which is common for market data (e.g., weekends, holidays)
            close_prices = close_prices.ffill()

            print(f"Successfully fetched {len(close_prices)} data points.")
            return close_prices

        except Exception as e:
            print(f"An error occurred while fetching data from yfinance: {e}")
            return pd.DataFrame()

if __name__ == '__main__':
    collector = MacroDataCollector()

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
