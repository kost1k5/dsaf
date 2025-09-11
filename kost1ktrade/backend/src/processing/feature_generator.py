import pandas as pd
import numpy as np
import talib
from statsmodels.tsa.stattools import adfuller
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

class FeatureGenerator:
    """
    A class to generate a wide array of features for the trading model.
    It takes various dataframes as input and produces a single dataframe with all features.
    """
    def __init__(self, asset: str, ohlcv_df: pd.DataFrame, timeframe: str,
                 ohlcv_df_4h: pd.DataFrame = None, ohlcv_df_1d: pd.DataFrame = None,
                 funding_rate_df: pd.DataFrame = None,
                 macro_df: pd.DataFrame = None, fng_df: pd.DataFrame = None, news_df: pd.DataFrame = None,
                 eth_ohlcv_df: pd.DataFrame = None, economic_calendar_df: pd.DataFrame = None):
        """
        Initializes the FeatureGenerator with all necessary dataframes.
        The main dataframe `self.df` is based on the ohlcv data.
        """
        log_file_path = os.path.join(os.path.dirname(__file__), '..', '..', f'indicator_log_{asset}.txt')
        # Open in 'w' mode to clear the log for this specific asset on each run
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self._log(f"--- Starting Feature Generation for {asset} ---")

        # --- Critical Data Check ---
        if ohlcv_df is None or ohlcv_df.empty:
            self._log("CRITICAL: Primary OHLCV DataFrame is missing or empty. Halting feature generation.")
            raise ValueError("Primary OHLCV DataFrame cannot be empty.")

        self.timeframe = timeframe

        # Standardize all dataframes
        # The check above ensures ohlcv_df is valid, so we can safely copy.
        self.df = self._standardize_ohlcv(ohlcv_df, "ohlcv_df").copy()
        self._log("  - Calculating log returns for price stationarity.")
        self.df['log_returns'] = np.log(self.df['close']).diff()

        self.ohlcv_4h = self._standardize_ohlcv(ohlcv_df_4h, "ohlcv_4h")
        self.ohlcv_1d = self._standardize_ohlcv(ohlcv_df_1d, "ohlcv_1d")
        self.eth_ohlcv = self._standardize_ohlcv(eth_ohlcv_df, "eth_ohlcv")
        self.funding_rate = self._standardize_funding_rate(funding_rate_df)
        self.macro = self._standardize_macro_data(macro_df)
        self.fng = self._standardize_generic(fng_df, "fng_df", ['timestamp', 'date'])
        self.news = self._standardize_generic(news_df, "news_df", ['published_at', 'published', 'timestamp'])
        self.economic_calendar = self._standardize_generic(economic_calendar_df, "economic_calendar", ['timestamp', 'event_datetime'])

        self.open_interest = None

    def _log(self, message: str):
        """Logs a message to the console and the designated log file."""
        print(message)
        self.log_file.write(message + '\n')

    def _standardize_generic(self, df, name, time_cols):
        """Standardizes a generic dataframe to have a 'timestamp' DatetimeIndex."""
        if df is None or df.empty:
            self._log(f"  - DataFrame '{name}' is None or empty. Skipping.")
            return None

        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            df.index.name = 'timestamp'
            return df.sort_index()

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
        return self._standardize_generic(df, name, ['open_time', 'timestamp'])

    def _standardize_funding_rate(self, df):
        """Helper method to standardize the funding rate dataframe index and columns."""
        if df is None or df.empty:
            self._log("  - DataFrame 'funding_rate' is None or empty. Skipping.")
            return None

        temp_df = self._standardize_generic(df, "funding_rate", ['funding_time', 'timestamp'])
        if temp_df is None: return None

        rate_col = next((col for col in ['funding_rate', 'fundingRate', 'rate'] if col in temp_df.columns), None)

        if rate_col:
            if rate_col != 'funding_rate':
                temp_df.rename(columns={rate_col: 'funding_rate'}, inplace=True)
            return temp_df[['funding_rate']]
        else:
            self._log(f"Error: Could not identify the funding rate value column. Available columns: {temp_df.columns.tolist()}. Skipping.")
            return None

    def _standardize_macro_data(self, df):
        """
        Helper method to standardize the macro dataframe.
        Handles both Wide and Long (EAV) formats.
        """
        if df is None or df.empty:
            return None

        temp_df = self._standardize_generic(df, "macro_df", ['date', 'timestamp'])
        if temp_df is None: return None

        expected_metrics = ['SPY', 'VIX', 'DXY']

        # --- Enhanced Wide Format Detection ---
        # New logic to find metrics even with suffixes like '_close'
        rename_map = {}
        for col in temp_df.columns:
            col_upper = str(col).upper()
            for metric in expected_metrics:
                if metric in col_upper:
                    rename_map[col] = metric
                    break # Move to the next column once a match is found

        if rename_map:
             self._log(f"  - Found wide-format macro columns: {list(rename_map.keys())}. Renaming to standard format.")
             temp_df.rename(columns=rename_map, inplace=True)
             # Filter to only keep the expected metrics that were actually found
             available_metrics = [metric for metric in expected_metrics if metric in temp_df.columns]
             return temp_df[available_metrics]


        # --- Fallback to Long (EAV) Format Detection ---
        metric_col = next((col for col in ['metric', 'name', 'symbol', 'ticker', 'indicator'] if col in temp_df.columns), None)
        value_col = next((col for col in ['value', 'price', 'close'] if col in temp_df.columns), None)

        if metric_col and value_col:
            try:
                self._log("  - Attempting to pivot long-format macro data.")
                pivoted_df = temp_df.pivot_table(index='timestamp', columns=metric_col, values=value_col, aggfunc='mean')

                # The pivot might create columns like 'spy' (lowercase), so we do the same rename logic as above
                pivot_rename_map = {}
                for col in pivoted_df.columns:
                    col_upper = str(col).upper()
                    for metric in expected_metrics:
                        if metric in col_upper:
                            pivot_rename_map[col] = metric
                            break

                pivoted_df.rename(columns=pivot_rename_map, inplace=True)

                available_metrics = [m for m in expected_metrics if m in pivoted_df.columns]
                if not available_metrics:
                    self._log("Warning: After pivoting macro data, none of the expected metrics were found. Skipping.")
                    return None

                self._log(f"  - Successfully pivoted macro data. Found metrics: {available_metrics}")
                return pivoted_df[available_metrics]
            except Exception as e:
                self._log(f"Error during pivoting macro data: {e}. Skipping.")
                return None

        self._log(f"Warning: Macro data format not recognized. Columns: {temp_df.columns.tolist()}. Skipping.")
        return None

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Helper to calculate daily VWAP efficiently."""
        if not isinstance(df.index, pd.DatetimeIndex):
            self._log("Error: VWAP calculation requires a DatetimeIndex.")
            return pd.Series(index=df.index, dtype='float64')

        # Group by date
        grouped = df.groupby(df.index.date)

        # Calculate cumulative volume and cumulative volume * price
        cum_vol = grouped['volume'].transform('cumsum')
        cum_vol_price = (df['close'] * df['volume']).groupby(df.index.date).transform('cumsum')

        # Calculate VWAP, handle potential division by zero
        vwap = (cum_vol_price / cum_vol).fillna(0)

        return vwap.rename('VWAP_D')

    def add_technical_indicators(self):
        self._log("\n[Step 1/8] Adding Technical Indicators (using TA-Lib)")
        # Prepare numpy arrays for TA-Lib
        high, low, close, volume = self.df['high'].values, self.df['low'].values, self.df['close'].values, self.df['volume'].values

        for length in [7, 14, 21]:
            self._log(f"  - Calculating RSI, ADX, PPO for length: {length}")
            self.df[f'RSI_{length}'] = talib.RSI(close, timeperiod=length)
            self.df[f'ADX_{length}'] = talib.ADX(high, low, close, timeperiod=length)

            # Correct PPO Calculation
            ppo_line_np = talib.PPO(close, fastperiod=length, slowperiod=length*2, matype=0)
            ppo_signal_np = talib.EMA(ppo_line_np, timeperiod=9)
            ppo_line_series = pd.Series(ppo_line_np, index=self.df.index)
            ppo_signal_series = pd.Series(ppo_signal_np, index=self.df.index)
            ppo_hist_series = ppo_line_series - ppo_signal_series

            self.df[f'PPO_{length}_{length*2}_9'] = ppo_line_series
            self.df[f'PPOs_{length}_{length*2}_9'] = ppo_signal_series
            self.df[f'PPOh_{length}_{length*2}_9'] = ppo_hist_series

        self._log("  - Calculating ATR (length: 14)")
        # Other parts of the code expect 'ATRr_14' from pandas-ta, so we match that name.
        self.df['ATRr_14'] = talib.ATR(high, low, close, timeperiod=14)

        self._log("  - Calculating Bollinger Bands & Width (length: 20)")
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        self.df['BBU_20_2.0'] = upper
        self.df['BBM_20_2.0'] = middle
        self.df['BBL_20_2.0'] = lower
        # Replace infinities with NaN, as BBM can be 0
        bbw = (self.df['BBU_20_2.0'] - self.df['BBL_20_2.0']) / self.df['BBM_20_2.0']
        self.df['BBW_20_2.0'] = bbw.replace([np.inf, -np.inf], np.nan)

        self._log("  - Calculating Stochastic Oscillator (14, 3, 3)")
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        self.df['STOCH_slowk'] = slowk
        self.df['STOCH_slowd'] = slowd

        self._log("  - Calculating MACD (12, 26, 9)")
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.df['MACD_12_26_9'] = macd
        self.df['MACDs_12_26_9'] = macdsignal
        self.df['MACDh_12_26_9'] = macdhist

        self._log("  - Calculating On-Balance Volume (OBV)")
        self.df['OBV'] = talib.OBV(close, volume)

        self._log("  - Calculating Money Flow Index (MFI as CMF substitute, length: 20)")
        self.df['CMF_20'] = talib.MFI(high, low, close, volume, timeperiod=20)

        self._log("  - Calculating Volume Weighted Average Price (VWAP)")
        self.df['VWAP_D'] = self._calculate_vwap(self.df)
        if 'VWAP_D' in self.df.columns:
            self.df['dist_from_vwap'] = (self.df['close'] / self.df['VWAP_D']) - 1
            self._log("  - Calculating distance from VWAP")

        self._log("  - Calculating EMA (length: 200)")
        self.df['EMA_200'] = talib.EMA(close, timeperiod=200)

        self._log(f"  - Generated {len(self.df.columns)} columns so far.")
        return self

    def add_multi_timeframe_features(self):
        self._log("\n[Step 2/8] Adding Multi-Timeframe Features (using TA-Lib)")
        for tf_df, tf_name in [(self.ohlcv_4h, "4h"), (self.ohlcv_1d, "1d")]:
            if tf_df is None or tf_df.empty:
                self._log(f"  - Skipping {tf_name} timeframe, data not available.")
                continue
            self._log(f"  - Calculating indicators for {tf_name} timeframe (RSI, PPO)")

            # Prepare numpy arrays
            tf_close = tf_df['close'].values

            # Calculate indicators
            tf_df['RSI_14'] = talib.RSI(tf_close, timeperiod=14)

            # Correct PPO Calculation
            ppo_line_np = talib.PPO(tf_close, fastperiod=12, slowperiod=26, matype=0)
            ppo_signal_np = talib.EMA(ppo_line_np, timeperiod=9)
            ppo_line_series = pd.Series(ppo_line_np, index=tf_df.index)
            ppo_signal_series = pd.Series(ppo_signal_np, index=tf_df.index)
            ppo_hist_series = ppo_line_series - ppo_signal_series

            tf_df['PPO_12_26_9'] = ppo_line_series
            tf_df['PPOs_12_26_9'] = ppo_signal_series
            tf_df['PPOh_12_26_9'] = ppo_hist_series

            # Select and rename indicators for merging
            indicators = tf_df[['RSI_14', 'PPO_12_26_9', 'PPOh_12_26_9', 'PPOs_12_26_9']].add_suffix(f'_{tf_name}')

            self.df = pd.merge_asof(self.df, indicators, left_index=True, right_index=True, direction='backward')
            self._log(f"  - Merged {tf_name} indicators onto base timeframe.")
        return self

    def add_derivative_features(self):
        self._log("\n[Step 3/8] Adding Derivative Features")
        if self.funding_rate is None or self.funding_rate.empty:
            self._log("  - Funding rate data not available or standardized. Skipping derivative features.")
            return self
        self._log("  - Merging funding rate data.")
        if not self.df.index.is_monotonic_increasing:
             self.df.sort_index(inplace=True)
        self.df = pd.merge_asof(
            self.df,
            self.funding_rate,
            left_index=True,
            right_index=True,
            direction='backward'
        )
        self.df['funding_rate'] = self.df['funding_rate'].ffill()
        self._log("  - Calculating funding rate momentum (3-period diff).")
        self.df['funding_rate_mom'] = self.df['funding_rate'].diff(periods=3)
        return self

    def add_sentiment_features(self):
        self._log("\n[Step 4/8] Adding Sentiment Features")
        if self.fng is not None and not self.fng.empty:
            self._log("  - Merging Fear & Greed Index data.")
            self.df = pd.merge_asof(self.df, self.fng['fng_value'], left_index=True, right_index=True, direction='backward')
            self.df['fng_value'] = self.df['fng_value'].ffill()
        else:
            self._log("  - No Fear & Greed data available.")
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
        self._log("\n[Step 6/8] Adding Cross-Market Features")
        if self.macro is not None and not self.macro.empty:
            self._log(f"  - Merging macro data ({', '.join(self.macro.columns)}).")
            if not self.df.index.is_monotonic_increasing:
                self.df.sort_index(inplace=True)
            self.df = pd.merge_asof(
                self.df,
                self.macro,
                left_index=True,
                right_index=True,
                direction='backward'
            )
        else:
            self._log("  - Macro data not available or standardized. Skipping cross-market features.")
        if self.eth_ohlcv is not None and not self.eth_ohlcv.empty:
            self._log("  - Calculating 30-day rolling correlation with ETH returns.")
            eth_returns = self.eth_ohlcv['close'].pct_change().rename('eth_returns')
            asset_returns = self.df['close'].pct_change().rename('asset_returns')
            merged_returns = pd.merge_asof(asset_returns.to_frame(), eth_returns.to_frame(), left_index=True, right_index=True, direction='backward')
            window = 24 * 30
            self.df['corr_eth_30d'] = merged_returns['asset_returns'].rolling(window=window).corr(merged_returns['eth_returns'])
        else:
            self._log("  - No ETH data available for correlation.")
        return self

    def add_time_based_features(self):
        self._log("\n[Step 7/8] Adding Time-Based Features")
        self.df['hour_of_day'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek
        self._log("  - Added 'hour_of_day' and 'day_of_week'.")
        self._log("  - Calculating 24-period rolling volatility.")
        if 'h' in self.timeframe:
            try:
                hours = int(self.timeframe.replace('h', ''))
                window_24h = 24 // hours
                self.df['volatility_24h'] = self.df['close'].pct_change().rolling(window=window_24h).std()
            except (ValueError, ZeroDivisionError):
                self._log(f"Warning: Could not parse timeframe '{self.timeframe}' to calculate volatility. Skipping.")
        else:
            self.df['volatility_24h'] = self.df['close'].pct_change().rolling(window=24).std()
        return self

    def add_lag_features(self):
        self._log("\n[Step 8/9] Adding Lag Features")
        lags = [3, 6, 12, 24]
        self._log(f"  - Lagging columns with periods: {lags}")
        self.df['close_pct_change'] = self.df['close'].pct_change()
        cols_to_lag = ['close_pct_change', 'volume', 'funding_rate']
        for col in cols_to_lag:
            if col in self.df.columns:
                self._log(f"    - Lagging '{col}'")
                for lag in lags:
                    self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
            else:
                self._log(f"    - Column '{col}' not found for lagging, skipping.")
        self.df.drop(columns=['close_pct_change'], inplace=True, errors='ignore')
        return self

    def add_calendar_features(self):
        self._log("\n[Step 5/8] Adding Economic Calendar Features")
        if self.economic_calendar is None or self.economic_calendar.empty:
            self._log("  - Economic calendar data not available. Skipping.")
            return self

        # Ensure calendar is sorted
        calendar_df = self.economic_calendar.sort_index()

        # Filter for high importance events
        high_impact_events = calendar_df[calendar_df['importance'].str.lower() == 'high']

        if high_impact_events.empty:
            self._log("  - No high-impact events found in the calendar data.")
            self.df['minutes_to_next_event'] = -1 # Sentinel value
            self.df['is_high_impact_event_in_next_24h'] = 0
            return self

        # Use merge_asof to find the next event for each timestamp
        merged_df = pd.merge_asof(
            self.df.sort_index(),
            high_impact_events.add_suffix('_event'),
            left_index=True,
            right_index=True,
            direction='forward' # Find next event
        )

        # Calculate time difference
        time_diff = merged_df['timestamp_event'] - merged_df.index

        # Convert to minutes and handle NaNs for timestamps with no future event
        self.df['minutes_to_next_event'] = time_diff.dt.total_seconds().div(60).fillna(-1)

        # Create binary flag for events within 24 hours
        one_day = pd.Timedelta(days=1)
        self.df['is_high_impact_event_in_next_24h'] = ((time_diff >= pd.Timedelta(0)) & (time_diff <= one_day)).astype(int)

        # --- Create News Filter Feature ---
        # Find the time to the PREVIOUS event as well
        merged_df_prev = pd.merge_asof(
            self.df.sort_index(),
            high_impact_events.add_suffix('_event_prev'),
            left_index=True,
            right_index=True,
            direction='backward' # Find previous event
        )
        time_diff_prev = merged_df_prev.index - merged_df_prev['timestamp_event_prev']

        # A news event is active if we are within 60 mins AFTER it, or 60 mins BEFORE the next one
        one_hour = pd.Timedelta(minutes=60)
        after_event = (time_diff_prev >= pd.Timedelta(0)) & (time_diff_prev <= one_hour)
        before_event = (time_diff >= pd.Timedelta(0)) & (time_diff <= one_hour)

        self.df['news_filter_60min'] = (after_event | before_event).astype(int)

        self._log("  - Added 'minutes_to_next_event', 'is_high_impact_event_in_next_24h', and 'news_filter_60min' features.")
        return self

    def check_and_transform_stationarity(self):
        self._log("\n[Bonus Step] Checking for Feature Stationarity")
        # Exclude columns that are known to be non-stationary or are needed in their raw form later.
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'EMA_200', 'ATRr_14']
        for col in self.df.columns.copy():
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.df[col]):
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
        # For adfuller with maxlag=48, we need nobs > 2 * (maxlag + 1), which is roughly 100.
        # Let's use a safe buffer.
        if len(series) < 101:
            self._log(f"  -> Series has too few observations ({len(series)}) for ADF test with maxlag=48. Skipping.")
            return 0.0
        if series.nunique() < 2:
            self._log(f"  -> Series is constant. Marking as non-stationary.")
            return 1.0
        try:
            result = adfuller(series, maxlag=48)
            return result[1]
        except Exception as e:
            self._log(f"  -> ADF test failed for series. Error: {e}. Marking as non-stationary.")
            return 1.0

    def _cleanup_raw_data(self):
        """Removes raw price columns to prevent lookahead bias or non-stationarity issues."""
        self._log("\n[Bonus Step] Cleaning up raw price data...")
        # Note: 'volume' is kept for indicators like OBV and MFI, and its stationarity
        # is handled by the check_and_transform_stationarity step.
        # We KEEP 'close' as it's needed for the labeling step.
        cols_to_drop = ['open', 'high', 'low']
        self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        self._log(f"  - Dropped columns: {cols_to_drop}")
        return self

    def generate_all_features(self):
        """
        Runs the full feature generation pipeline.
        """
        self.add_technical_indicators()
        self.add_multi_timeframe_features()
        self.add_derivative_features()
        self.add_sentiment_features()
        self.add_calendar_features() # Add new step here
        self.add_cross_market_features()
        self.add_time_based_features()
        self.add_lag_features()

        self._cleanup_raw_data()

        self._log("\n[Bonus Step] Optimizing memory by downcasting float types...")
        for col in self.df.select_dtypes(include=['float64']).columns:
            self.df[col] = self.df[col].astype('float32')
        self._log("  - Downcasting complete.")

        self.check_and_transform_stationarity()
        self._log("\n--- Feature Generation Complete ---")
        self.log_file.close()
        print("Feature generation complete.")
        return self.df
