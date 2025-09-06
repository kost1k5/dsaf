import pandas as pd
import pandas_ta as ta
from statsmodels.tsa.stattools import adfuller
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os # (A) Import os for file path handling

class FeatureGenerator:
    """
    A class to generate a wide array of features for the trading model.
    It takes various dataframes as input and produces a single dataframe with all features.
    """
    def __init__(self, ohlcv_df: pd.DataFrame, timeframe: str,
                 ohlcv_df_4h: pd.DataFrame = None, ohlcv_df_1d: pd.DataFrame = None,
                 funding_rate_df: pd.DataFrame = None,
                 macro_df: pd.DataFrame = None, fng_df: pd.DataFrame = None, news_df: pd.DataFrame = None,
                 eth_ohlcv_df: pd.DataFrame = None):
        """
        Initializes the FeatureGenerator with all necessary dataframes.
        The main dataframe `self.df` is based on the ohlcv data.
        """
        # (A) Setup logging
        log_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'indicator_log.txt')
        # Use 'a' to append, which is safer for parallel script execution
        self.log_file = open(log_file_path, 'a', encoding='utf-8')
        self._log("--- Starting Feature Generation ---")

        self.timeframe = timeframe

        # Standardize the main OHLCV dataframe
        self.df = self._standardize_ohlcv(ohlcv_df, "ohlcv_df").copy()
        if self.df is None:
            raise ValueError("Primary OHLCV dataframe is missing or invalid.")

        # Standardize all other optional dataframes
        self.ohlcv_4h = self._standardize_ohlcv(ohlcv_df_4h, "ohlcv_4h")
        self.ohlcv_1d = self._standardize_ohlcv(ohlcv_df_1d, "ohlcv_1d")
        self.eth_ohlcv = self._standardize_ohlcv(eth_ohlcv_df, "eth_ohlcv")
        self.funding_rate = self._standardize_funding_rate(funding_rate_df)
        self.macro = self._standardize_generic(macro_df, "macro_df", ['Date', 'timestamp'])
        self.fng = self._standardize_generic(fng_df, "fng_df", ['timestamp'])
        self.news = self._standardize_generic(news_df, "news_df", ['published_at', 'published', 'timestamp'])

        self.open_interest = None # OI feature remains removed

    def _standardize_generic(self, df, name, time_cols):
        """Standardizes a generic dataframe to have a 'timestamp' DatetimeIndex."""
        if df is None or df.empty:
            self._log(f"  - DataFrame '{name}' is None or empty. Skipping.")
            return None

        df = df.copy()

        # If index is already datetime, just ensure name is correct
        if isinstance(df.index, pd.DatetimeIndex):
            df.index.name = 'timestamp' # Standardize name
            return df.sort_index()

        # Find the time column
        time_col = next((col for col in time_cols if col in df.columns), None)
        if time_col:
            df = df.set_index(time_col)
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = 'timestamp'
            return df.sort_index()

        self._log(f"Warning: Could not find a suitable time column in '{name}'. Columns: {df.columns}. Skipping.")
        return None

    def _standardize_ohlcv(self, df, name):
        """Standardizes an OHLCV dataframe."""
        if df is None or df.empty:
            return None
        # In our pipeline, OHLCV data comes pre-indexed from the DB loader.
        if isinstance(df.index, pd.DatetimeIndex):
            df.index.name = 'timestamp'
            return df.sort_index()
        # Fallback if it's not indexed
        return self._standardize_generic(df, name, ['open_time', 'timestamp'])

    def _standardize_funding_rate(self, df):
        """Helper method to standardize the funding rate dataframe index and columns."""
        if df is None or df.empty:
            self._log("  - DataFrame 'funding_rate' is None or empty. Skipping.")
            return None

        temp_df = self._standardize_generic(df, "funding_rate", ['funding_time', 'timestamp'])
        if temp_df is None:
            return None

        # Standardize the Rate Value Column
        rate_col = None
        possible_rate_cols = ['funding_rate', 'fundingRate', 'rate']
        for col in possible_rate_cols:
            if col in temp_df.columns:
                rate_col = col
                break

        if rate_col:
            # Rename to the standardized internal name 'funding_rate'
            if rate_col != 'funding_rate':
                temp_df.rename(columns={rate_col: 'funding_rate'}, inplace=True)

            # Return the standardized DataFrame, keeping only the necessary column
            return temp_df[['funding_rate']]
        else:
            self._log(f"Error: Could not identify the funding rate value column. Available columns: {temp_df.columns.tolist()}. Skipping funding data.")
            return None

    def _log(self, message: str):
        """Logs a message to the console and the designated log file."""
        print(message)
        self.log_file.write(message + '\n')

    def add_technical_indicators(self):
        """
        Calculates and adds technical indicators using the pandas-ta library.
        (Б, В) This has been expanded to include multiple time windows and new indicators.
        """
        self._log("\n[Step 1/8] Adding Technical Indicators")

        # (В) Add multiple time windows for key indicators
        for length in [7, 14, 21]:
            self._log(f"  - Calculating RSI, ADX, PPO for length: {length}")
            self.df.ta.rsi(length=length, append=True)
            self.df.ta.adx(length=length, append=True)
            self.df.ta.ppo(fast=length, slow=length*2, append=True)

        # (Б.1) Add Volatility and Volume indicators
        self._log("  - Calculating ATR (length: 14)")
        self.df.ta.atr(append=True)

        self._log("  - Calculating Bollinger Bands (length: 20)")
        self.df.ta.bbands(length=20, append=True)

        self._log("  - Calculating On-Balance Volume (OBV)")
        self.df.ta.obv(append=True)

        self._log("  - Calculating Chaikin Money Flow (CMF, length: 20)")
        self.df.ta.cmf(append=True)

        self._log("  - Calculating Volume Weighted Average Price (VWAP)")
        self.df.ta.vwap(append=True)
        # The default column name for vwap is VWAP_D
        if 'VWAP_D' in self.df.columns:
            self.df['dist_from_vwap'] = (self.df['close'] / self.df['VWAP_D']) - 1
            self._log("  - Calculating distance from VWAP")

        self._log(f"  - Generated {len(self.df.columns)} columns so far.")
        return self

    def add_multi_timeframe_features(self):
        """
        Adds technical indicators from higher timeframes (4h, 1d) to the base dataframe.
        """
        self._log("\n[Step 2/8] Adding Multi-Timeframe Features")
        for tf_df, tf_name in [(self.ohlcv_4h, "4h"), (self.ohlcv_1d, "1d")]:
            if tf_df is None or tf_df.empty:
                self._log(f"  - Skipping {tf_name} timeframe, data not available.")
                continue

            self._log(f"  - Calculating indicators for {tf_name} timeframe (RSI, PPO)")
            # Calculate indicators on the higher timeframe dataframe
            tf_df.ta.rsi(append=True)
            tf_df.ta.ppo(append=True)

            # Select and rename the indicator columns to avoid name clashes
            indicators = tf_df[['RSI_14', 'PPO_12_26_9', 'PPOh_12_26_9', 'PPOs_12_26_9']]
            indicators = indicators.add_suffix(f'_{tf_name}')

            # Merge the HTF indicators onto the base dataframe
            self.df = pd.merge_asof(self.df, indicators, left_index=True, right_index=True, direction='backward')
            self._log(f"  - Merged {tf_name} indicators onto base timeframe.")

        return self

    def add_derivative_features(self):
        """
        Calculates and adds features based on open interest and funding rates.
        Uses merge_asof for robust joining of sparse data.
        """
        self._log("\n[Step 3/8] Adding Derivative Features")

        # Check if standardization in __init__ was successful
        if self.funding_rate is None or self.funding_rate.empty:
            self._log("  - Funding rate data not available or standardized. Skipping derivative features.")
            return self

        self._log("  - Merging funding rate data.")

        # Ensure the main DF is sorted (required for merge_asof).
        if not self.df.index.is_monotonic_increasing:
             self.df.sort_index(inplace=True)

        # Replace the original selection/rename logic (Line 145) with this merge.
        # self.funding_rate is already sorted and standardized.
        self.df = pd.merge_asof(
            self.df,
            self.funding_rate, # Already standardized to have 'funding_rate' column
            left_index=True,
            right_index=True,
            direction='backward' # Use the most recent past value
        )

        # Forward-fill is okay, back-filling is not.
        self.df['funding_rate'] = self.df['funding_rate'].ffill()
        self._log("  - Calculating funding rate momentum (3-period diff).")
        self.df['funding_rate_mom'] = self.df['funding_rate'].diff(periods=3)

        return self

    def add_sentiment_features(self):
        """
        Calculates and adds features based on sentiment data.
        """
        self._log("\n[Step 4/8] Adding Sentiment Features")
        # Add Fear & Greed Index
        if self.fng is not None and not self.fng.empty:
            self._log("  - Merging Fear & Greed Index data.")
            self.df = pd.merge_asof(self.df, self.fng['fng_value'], left_index=True, right_index=True, direction='backward')
            self.df['fng_value'] = self.df['fng_value'].ffill()
        else:
            self._log("  - No Fear & Greed data available.")

        # Add VADER sentiment from news headlines
        if self.news is not None and not self.news.empty:
            self._log("  - Calculating VADER sentiment from news headlines.")
            analyzer = SentimentIntensityAnalyzer()
            self.news['title'] = self.news['title'].astype(str)
            self.news['news_sentiment'] = self.news['title'].apply(lambda title: analyzer.polarity_scores(title)['compound'])
            daily_sentiment = self.news[['news_sentiment']].resample('D').mean()
            self.df = pd.merge_asof(self.df, daily_sentiment, left_index=True, right_index=True, direction='backward')
            self.df['news_sentiment'] = self.df['news_sentiment'].ffill()
            self._log("  - Merged daily news sentiment.")
        else:
            self._log("  - No news data available.")

        return self

    def add_cross_market_features(self):
        """
        Calculates and adds features based on cross-market correlations.
        """
        self._log("\n[Step 5/8] Adding Cross-Market Features")
        if self.macro is not None:
            self._log("  - Merging macro data (SPY, VIX, DXY).")
            self.df = pd.merge_asof(self.df, self.macro[['SPY', 'VIX', 'DXY']], left_index=True, right_index=True, direction='backward')
        else:
            self._log("  - No macro data available.")

        if self.eth_ohlcv is not None and not self.eth_ohlcv.empty:
            self._log("  - Calculating 30-day rolling correlation with ETH returns.")
            # Calculate returns for both assets
            eth_returns = self.eth_ohlcv['close'].pct_change().rename('eth_returns')
            asset_returns = self.df['close'].pct_change().rename('asset_returns')

            # Merge ETH returns onto the main df
            merged_returns = pd.merge_asof(asset_returns.to_frame(), eth_returns.to_frame(), left_index=True, right_index=True, direction='backward')

            # Calculate rolling correlation (e.g., 30-day window on 1h data)
            window = 24 * 30
            self.df['corr_eth_30d'] = merged_returns['asset_returns'].rolling(window=window).corr(merged_returns['eth_returns'])
        else:
            self._log("  - No ETH data available for correlation.")

        return self

    def add_time_based_features(self):
        """
        Adds time-based features like hour of day, day of week, and volatility regimes.
        """
        self._log("\n[Step 6/8] Adding Time-Based Features")
        self.df['hour_of_day'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek
        self._log("  - Added 'hour_of_day' and 'day_of_week'.")

        # Add volatility feature
        self._log("  - Calculating 24-period rolling volatility.")
        if 'h' in self.timeframe:
            try:
                hours = int(self.timeframe.replace('h', ''))
                window_24h = 24 // hours
                self.df['volatility_24h'] = self.df['close'].pct_change().rolling(window=window_24h).std()
            except (ValueError, ZeroDivisionError):
                self._log(f"Warning: Could not parse timeframe '{self.timeframe}' to calculate volatility. Skipping.")
        else:
             # Default to a 24 period window if timeframe is not in hours (e.g. '1d')
            self.df['volatility_24h'] = self.df['close'].pct_change().rolling(window=24).std()

        return self

    def add_lag_features(self):
        """
        Adds lagged versions of important features to provide historical context.
        """
        self._log("\n[Step 7/8] Adding Lag Features")
        lags = [3, 6, 12, 24]
        self._log(f"  - Lagging columns with periods: {lags}")

        # Define columns to lag
        self.df['close_pct_change'] = self.df['close'].pct_change()
        cols_to_lag = ['close_pct_change', 'volume', 'funding_rate']

        for col in cols_to_lag:
            if col in self.df.columns:
                self._log(f"    - Lagging '{col}'")
                for lag in lags:
                    self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
            else:
                self._log(f"    - Column '{col}' not found for lagging, skipping.")


        # We can drop the original close_pct_change as it's now represented by its lags
        self.df.drop(columns=['close_pct_change'], inplace=True, errors='ignore')

        return self

    def add_interaction_features(self):
        """
        Creates features by combining or transforming existing ones.
        """
        self._log("\n[Step 8/8] Adding Interaction Features")
        if 'RSI_14' in self.df.columns and 'ATRr_14' in self.df.columns:
            self._log("  - Creating 'rsi_vol_norm' (RSI normalized by ATR).")
            # Example: RSI normalized by volatility
            atr_normalized = self.df['ATRr_14'] / self.df['close']
            # Avoid division by zero
            atr_normalized[atr_normalized == 0] = 1e-6
            self.df['rsi_vol_norm'] = self.df['RSI_14'] * (1 / atr_normalized)
        else:
            self._log("  - Skipping 'rsi_vol_norm' (required columns not found).")

        return self

    def check_and_transform_stationarity(self):
        """
        Checks all feature columns for stationarity and applies .pct_change() if not stationary.
        """
        self._log("\n[Bonus Step] Checking for Feature Stationarity")
        # List of columns to exclude from stationarity check (e.g., identifiers, original prices)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']

        # Iterate over a copy of column names as we might modify the dataframe
        for col in self.df.columns.copy():
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.df[col]):
                # self._log(f"Testing stationarity of: {col}") # This is too verbose
                p_value = self.test_stationarity(self.df[col])
                if p_value > 0.05:
                    self._log(f"  - Column '{col}' is not stationary (p-value: {p_value:.4f}). Applying transformation.")
                    self.df[f'{col}_pct_change'] = self.df[col].pct_change()
                    self.df.drop(columns=[col], inplace=True)
        return self

    def test_stationarity(self, series: pd.Series):
        """
        Performs the Augmented Dickey-Fuller test on a series.
        Returns the p-value.
        """
        series = series.dropna()
        if len(series) < 20: # Not enough data to test
            return 0.0 # Assume stationary if not enough data

        # If the series is constant, ADF test will fail. A constant series is non-stationary.
        if series.nunique() < 2:
            print(f"  -> Series is constant. Marking as non-stationary.")
            return 1.0 # p-value of 1 indicates non-stationarity

        try:
            result = adfuller(series)
            return result[1] # p-value
        except Exception as e:
            print(f"  -> ADF test failed for series. Error: {e}. Marking as non-stationary.")
            return 1.0

    def generate_all_features(self):
        """
        Runs the full feature generation pipeline.
        """
        self.add_technical_indicators()
        self.add_multi_timeframe_features()
        self.add_derivative_features()
        self.add_sentiment_features()
        self.add_cross_market_features()
        self.add_time_based_features()
        self.add_lag_features()
        self.add_interaction_features()

        # Final check for stationarity
        self.check_and_transform_stationarity()

        # Do not drop NaNs. LightGBM can handle them, and this preserves the maximum
        # amount of historical data, especially when some features have a shorter history.
        # self.df.dropna(inplace=True)

        self._log("\n--- Feature Generation Complete ---")
        self.log_file.close() # (A) Close the log file
        print("Feature generation complete.")
        return self.df
