import pandas as pd
import numpy as np
import talib
from statsmodels.tsa.stattools import adfuller
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import sys

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

        self._log("\n[Step 0/8] Initializing Feature Generator & Standardizing Data...")
        self._log(f"  - Received initial dataframes for asset '{asset}' on timeframe '{timeframe}':")
        for name, df in [
            ('ohlcv_df', ohlcv_df),
            ('ohlcv_df_4h', ohlcv_df_4h),
            ('ohlcv_df_1d', ohlcv_df_1d),
            ('funding_rate_df', funding_rate_df),
            ('macro_df', macro_df),
            ('fng_df', fng_df),
            ('news_df', news_df),
            ('eth_ohlcv_df', eth_ohlcv_df),
            ('economic_calendar_df', economic_calendar_df)
        ]:
            if df is not None and not df.empty:
                self._log(f"    - {name}: present, shape {df.shape}")
            else:
                self._log(f"    - {name}: missing or empty")


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
        sys.stdout.flush()
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def _standardize_generic(self, df, name, time_cols):
        """Standardizes a generic dataframe to have a 'timestamp' DatetimeIndex."""
        if df is None or df.empty:
            return None # Already logged in __init__

        self._log(f"  - Standardizing '{name}'...")
        self._log(f"    - Initial shape: {df.shape}, Initial index type: {type(df.index)}")
        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            self._log(f"    - Index is already a DatetimeIndex. Renaming to 'timestamp'.")
            df.index.name = 'timestamp'
            df = df.sort_index()
            self._log(f"    - Standardization complete for '{name}'. Final shape: {df.shape}")
            return df

        time_col = next((col for col in time_cols if col in df.columns), None)
        if time_col:
            self._log(f"    - Found time column: '{time_col}'. Setting as UTC DatetimeIndex.")
            df = df.set_index(time_col)
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = 'timestamp'
            df = df.sort_index()
            self._log(f"    - Standardization complete for '{name}'. Final shape: {df.shape}")
            return df

        self._log(f"  - WARNING in standardization: Could not find a suitable time column in '{name}'. Tried: {time_cols}. Columns: {df.columns}. Skipping.")
        return None

    def _standardize_ohlcv(self, df, name):
        """Standardizes an OHLCV dataframe."""
        return self._standardize_generic(df, name, ['open_time', 'timestamp'])

    def _standardize_funding_rate(self, df):
        """Helper method to standardize the funding rate dataframe index and columns."""
        if df is None or df.empty:
            return None # Logged in __init__

        temp_df = self._standardize_generic(df, "funding_rate", ['funding_time', 'timestamp'])
        if temp_df is None: return None

        self._log(f"  - Post-standardization check for 'funding_rate' columns...")
        rate_col = next((col for col in ['funding_rate', 'fundingRate', 'rate'] if col in temp_df.columns), None)

        if rate_col:
            self._log(f"    - Found funding rate value column: '{rate_col}'.")
            if rate_col != 'funding_rate':
                self._log(f"    - Renaming '{rate_col}' to 'funding_rate'.")
                temp_df.rename(columns={rate_col: 'funding_rate'}, inplace=True)
            self._log(f"    - Selecting final 'funding_rate' column.")
            return temp_df[['funding_rate']]
        else:
            self._log(f"  - ERROR in standardization: Could not identify the funding rate value column. Available columns: {temp_df.columns.tolist()}. Skipping.")
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

        self._log("  - Post-standardization check for 'macro_df' format...")
        expected_metrics = ['SPY', 'VIX', 'DXY']
        self._log(f"    - Expecting metrics: {expected_metrics}")

        # --- Enhanced Wide Format Detection ---
        self._log("    - Checking for wide-format data...")
        rename_map = {}
        for col in temp_df.columns:
            col_upper = str(col).upper()
            for metric in expected_metrics:
                if metric in col_upper:
                    rename_map[col] = metric
                    break

        if rename_map:
             self._log(f"    - Wide-format data found. Column rename map: {rename_map}.")
             temp_df.rename(columns=rename_map, inplace=True)
             available_metrics = [metric for metric in expected_metrics if metric in temp_df.columns]
             self._log(f"    - Found and selected metrics: {available_metrics}")
             return temp_df[available_metrics]
        self._log("    - No wide-format columns found.")


        # --- Fallback to Long (EAV) Format Detection ---
        self._log("    - Checking for long-format (EAV) data...")
        metric_col = next((col for col in ['metric', 'name', 'symbol', 'ticker', 'indicator'] if col in temp_df.columns), None)
        value_col = next((col for col in ['value', 'price', 'close'] if col in temp_df.columns), None)

        if metric_col and value_col:
            self._log(f"    - Long-format data found. Using metric column '{metric_col}' and value column '{value_col}'.")
            try:
                self._log(f"    - Pivoting data from long to wide format...")
                self._log(f"      - Shape before pivot: {temp_df.shape}")
                pivoted_df = temp_df.pivot_table(index='timestamp', columns=metric_col, values=value_col, aggfunc='mean')
                self._log(f"      - Shape after pivot: {pivoted_df.shape}")

                # The pivot might create columns like 'spy' (lowercase), so we do the same rename logic as above
                pivot_rename_map = {}
                for col in pivoted_df.columns:
                    col_upper = str(col).upper()
                    for metric in expected_metrics:
                        if metric in col_upper:
                            pivot_rename_map[col] = metric
                            break

                if pivot_rename_map:
                    self._log(f"    - Renaming pivoted columns. Map: {pivot_rename_map}")
                    pivoted_df.rename(columns=pivot_rename_map, inplace=True)

                available_metrics = [m for m in expected_metrics if m in pivoted_df.columns]
                if not available_metrics:
                    self._log("    - WARNING: After pivoting, none of the expected metrics were found. Skipping.")
                    return None

                self._log(f"    - Successfully pivoted. Found and selected metrics: {available_metrics}")
                return pivoted_df[available_metrics]
            except Exception as e:
                self._log(f"    - ERROR: Failed to pivot macro data: {e}. Skipping.")
                return None

        self._log(f"  - WARNING in standardization: Macro data format not recognized. Columns: {temp_df.columns.tolist()}. Skipping.")
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
        initial_cols = set(self.df.columns)
        self._log(f"  - Initial number of columns: {len(initial_cols)}")

        # Prepare numpy arrays for TA-Lib
        self._log("  - Preparing numpy arrays from dataframe columns (high, low, close, volume)...")
        high, low, close, volume = self.df['high'].values, self.df['low'].values, self.df['close'].values, self.df['volume'].values

        # Multi-length indicators
        for length in [7, 14, 21]:
            self._log(f"  - Group: Calculating indicators for length: {length}")

            self._log(f"    - RSI (length={length})")
            self.df[f'RSI_{length}'] = talib.RSI(close, timeperiod=length)

            self._log(f"    - ADX (length={length})")
            self.df[f'ADX_{length}'] = talib.ADX(high, low, close, timeperiod=length)

            self._log(f"    - PPO (fast={length}, slow={length*2}, signal=9)")
            ppo_line_np = talib.PPO(close, fastperiod=length, slowperiod=length*2, matype=0)
            ppo_signal_np = talib.EMA(ppo_line_np, timeperiod=9)
            ppo_line_series = pd.Series(ppo_line_np, index=self.df.index)
            ppo_signal_series = pd.Series(ppo_signal_np, index=self.df.index)
            ppo_hist_series = ppo_line_series - ppo_signal_series
            self.df[f'PPO_{length}_{length*2}_9'] = ppo_line_series
            self.df[f'PPOs_{length}_{length*2}_9'] = ppo_signal_series
            self.df[f'PPOh_{length}_{length*2}_9'] = ppo_hist_series

        # Single-length indicators
        self._log("  - Group: Calculating single-length indicators...")

        self._log("    - ATR (length=14)")
        self.df['ATR'] = talib.ATR(high, low, close, timeperiod=14)

        self._log("    - Bollinger Bands (length=20, stddev=2.0)")
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        self.df['BBU_20_2.0'] = upper
        self.df['BBM_20_2.0'] = middle
        self.df['BBL_20_2.0'] = lower
        self._log("    - Bollinger Band Width (length=20)")
        bbw = (self.df['BBU_20_2.0'] - self.df['BBL_20_2.0']) / self.df['BBM_20_2.0']
        self.df['BBW_20_2.0'] = bbw.replace([np.inf, -np.inf], np.nan)

        self._log("    - Stochastic Oscillator (fastk=14, slowk=3, slowd=3)")
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        self.df['STOCH_slowk'] = slowk
        self.df['STOCH_slowd'] = slowd

        self._log("    - MACD (fast=12, slow=26, signal=9)")
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.df['MACD_12_26_9'] = macd
        self.df['MACDs_12_26_9'] = macdsignal
        self.df['MACDh_12_26_9'] = macdhist

        self._log("    - On-Balance Volume (OBV)")
        self.df['OBV'] = talib.OBV(close, volume)

        self._log("    - Money Flow Index (MFI) (length=20)")
        self.df['CMF_20'] = talib.MFI(high, low, close, volume, timeperiod=20)

        self._log("    - Daily Volume Weighted Average Price (VWAP)")
        self.df['VWAP_D'] = self._calculate_vwap(self.df)
        if 'VWAP_D' in self.df.columns and self.df['VWAP_D'].notna().any():
            self._log("      - Calculating distance from VWAP")
            self.df['dist_from_vwap'] = (self.df['close'] / self.df['VWAP_D']) - 1
        else:
            self._log("      - VWAP calculation resulted in empty series. Skipping distance calculation.")

        self._log("    - Rolling VWAP (length=20)")
        vwap_period = 20  # As seen in .env: INDICATORS__VWAP_PERIOD=20
        typical_price_x_volume = (self.df['close'] * self.df['volume'])
        sum_of_pv = typical_price_x_volume.rolling(window=vwap_period).sum()
        sum_of_vol = self.df['volume'].rolling(window=vwap_period).sum()
        self.df[f'VWAP_{vwap_period}'] = sum_of_pv / sum_of_vol
        self.df[f'dist_from_vwap_{vwap_period}'] = (self.df['close'] / self.df[f'VWAP_{vwap_period}']) - 1


        self._log("    - EMA (length=200)")
        self.df['EMA_200'] = talib.EMA(close, timeperiod=200)

        final_cols = set(self.df.columns)
        new_cols = final_cols - initial_cols
        self._log(f"\n  - Step 1 Complete. Added {len(new_cols)} new indicator columns.")
        self._log(f"  - Total columns now: {len(final_cols)}.")
        return self

    def add_multi_timeframe_features(self):
        self._log("\n[Step 2/8] Adding Multi-Timeframe Features (using TA-Lib)")
        for tf_df, tf_name in [(self.ohlcv_4h, "4h"), (self.ohlcv_1d, "1d")]:
            if tf_df is None or tf_df.empty:
                self._log(f"  - Skipping {tf_name} timeframe, data not available.")
                continue
            self._log(f"  - Processing {tf_name} timeframe data...")

            # Prepare numpy arrays
            tf_close = tf_df['close'].values

            # Calculate indicators
            self._log(f"    - Calculating RSI_14 for {tf_name}")
            tf_df['RSI_14'] = talib.RSI(tf_close, timeperiod=14)

            self._log(f"    - Calculating PPO_12_26_9 for {tf_name}")
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
            self._log(f"    - Merging {tf_name} indicators: {list(indicators.columns)}")
            self._log(f"    - Merge strategy: pd.merge_asof (backward on index)")
            self._log(f"    - Shape before merge: {self.df.shape}")

            # Ensure both dataframes are sorted before merge_asof
            self.df.sort_index(inplace=True)
            indicators.sort_index(inplace=True)

            self.df = pd.merge_asof(self.df, indicators, left_index=True, right_index=True, direction='backward')
            self._log(f"    - Shape after merge: {self.df.shape}")
        self._log(f"  - Multi-timeframe feature step complete.")
        return self

    def add_derivative_features(self):
        self._log("\n[Step 3/8] Adding Derivative Features")
        if self.funding_rate is None or self.funding_rate.empty:
            self._log("  - Funding rate data not available or standardized. Skipping derivative features.")
            return self

        self._log("  - Merging funding rate data...")
        self._log(f"    - Merge strategy: pd.merge_asof (backward on index)")
        self._log(f"    - Shape before merge: {self.df.shape}")
        if not self.df.index.is_monotonic_increasing:
             self._log("    - Sorting main dataframe index before merge.")
             self.df.sort_index(inplace=True)

        self.df = pd.merge_asof(
            self.df,
            self.funding_rate,
            left_index=True,
            right_index=True,
            direction='backward'
        )
        self._log(f"    - Shape after merge: {self.df.shape}")

        self._log("  - Forward-filling missing funding rate values...")
        self.df['funding_rate'] = self.df['funding_rate'].ffill()

        self._log("  - Calculating funding rate momentum (3-period diff).")
        self.df['funding_rate_mom'] = self.df['funding_rate'].diff(periods=3)
        self._log(f"  - Derivative feature step complete.")
        return self

    def add_sentiment_features(self):
        self._log("\n[Step 4/8] Adding Sentiment Features")
        if self.fng is not None and not self.fng.empty:
            self._log("  - Merging Fear & Greed Index data...")
            self._log(f"    - Merge strategy: pd.merge_asof (backward on index)")
            self._log(f"    - Shape before merge: {self.df.shape}")
            self.df = pd.merge_asof(self.df.sort_index(), self.fng['fng_value'].sort_index(), left_index=True, right_index=True, direction='backward')
            self.df['fng_value'] = self.df['fng_value'].ffill()
            self._log(f"    - Shape after merge: {self.df.shape}")
        else:
            self._log("  - No Fear & Greed data available. Skipping F&G merge.")

        if self.news is not None and not self.news.empty:
            self._log("  - Calculating VADER sentiment from news headlines...")
            analyzer = SentimentIntensityAnalyzer()
            self.news['title'] = self.news['title'].astype(str)
            # This can be slow, so let's add a progress indicator if possible. For now, a simple log.
            self._log(f"    - Processing {len(self.news)} news headlines...")
            self.news['news_sentiment'] = self.news['title'].apply(lambda title: analyzer.polarity_scores(title)['compound'])
            self._log("    - Resampling news sentiment to daily average.")
            daily_sentiment = self.news[['news_sentiment']].resample('D').mean()

            self._log("  - Merging daily news sentiment...")
            self._log(f"    - Merge strategy: pd.merge_asof (backward on index)")
            self._log(f"    - Shape before merge: {self.df.shape}")
            self.df = pd.merge_asof(self.df.sort_index(), daily_sentiment.sort_index(), left_index=True, right_index=True, direction='backward')

            # Check if the merge resulted in any valid sentiment data
            if 'news_sentiment' in self.df.columns and self.df['news_sentiment'].notna().any():
                self.df['news_sentiment'] = self.df['news_sentiment'].ffill()
                self._log(f"    - Merge successful. Shape after merge: {self.df.shape}")
            else:
                self.df['news_sentiment'] = 0
                self._log("    - No matching news sentiment data for the given timeframe. Set to 0.")
        else:
            self._log("  - No news data available. Creating a neutral 'news_sentiment' column with value 0.")
            self.df['news_sentiment'] = 0
        self._log(f"  - Sentiment feature step complete.")
        return self

    def add_cross_market_features(self):
        self._log("\n[Step 6/8] Adding Cross-Market Features")
        if self.macro is not None and not self.macro.empty:
            self._log(f"  - Merging macro data ({', '.join(self.macro.columns)})...")
            self._log(f"    - Merge strategy: pd.merge_asof (backward on index)")
            self._log(f"    - Shape before merge: {self.df.shape}")
            if not self.df.index.is_monotonic_increasing:
                self._log("    - Sorting main dataframe index before merge.")
                self.df.sort_index(inplace=True)
            self.df = pd.merge_asof(
                self.df,
                self.macro.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'
            )
            self._log(f"    - Shape after merge: {self.df.shape}")
        else:
            self._log("  - Macro data not available. Skipping macro merge.")

        if self.eth_ohlcv is not None and not self.eth_ohlcv.empty:
            self._log("  - Calculating 30-day rolling correlation with ETH returns...")
            eth_returns = self.eth_ohlcv['close'].pct_change().rename('eth_returns')
            asset_returns = self.df['close'].pct_change().rename('asset_returns')

            self._log("    - Merging asset and ETH returns for correlation calculation...")
            merged_returns = pd.merge_asof(asset_returns.to_frame(), eth_returns.to_frame(), left_index=True, right_index=True, direction='backward')

            window = 24 * 30
            self._log(f"    - Calculating rolling correlation with window={window} periods...")
            self.df['corr_eth_30d'] = merged_returns['asset_returns'].rolling(window=window).corr(merged_returns['eth_returns'])
            self._log(f"    - Added 'corr_eth_30d' column.")
        else:
            self._log("  - No ETH data available for correlation. Skipping.")
        self._log(f"  - Cross-market feature step complete.")
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

        self._log("  - Calculating realized volatility over multiple periods (std of log returns)...")
        for period in [7, 21, 60]:
            self.df[f'realized_vol_{period}'] = self.df['log_returns'].rolling(window=period).std()
        self._log("  - Added realized volatility features.")

        return self

    def add_lag_features(self):
        self._log("\n[Step 8/8] Adding Lag Features")
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

        self._log("  - Processing economic calendar events...")
        calendar_df = self.economic_calendar.sort_index()
        self._log(f"    - Found {len(calendar_df)} total events.")

        # Filter for high importance events
        high_impact_events = calendar_df[calendar_df['importance'].str.lower() == 'high']
        self._log(f"    - Found {len(high_impact_events)} high-impact events.")

        if high_impact_events.empty:
            self._log("  - No high-impact events found. Setting default calendar features.")
            self.df['minutes_to_next_event'] = -1
            self.df['is_high_impact_event_in_next_24h'] = 0
            self.df['news_filter_60min'] = 0
            return self

        # --- Time to Next Event ---
        self._log("  - Calculating time to next high-impact event...")
        self._log(f"    - Merge strategy: pd.merge_asof (forward on index)")
        merged_df = pd.merge_asof(
            self.df.sort_index(),
            high_impact_events.add_suffix('_event'),
            left_index=True,
            right_index=True,
            direction='forward'
        )
        time_diff = merged_df['timestamp_event'] - merged_df.index
        self.df['minutes_to_next_event'] = time_diff.dt.total_seconds().div(60).fillna(-1)
        self._log("    - Added 'minutes_to_next_event' feature.")

        # --- Event within 24h flag ---
        self._log("  - Calculating flag for high-impact events in the next 24 hours...")
        one_day = pd.Timedelta(days=1)
        self.df['is_high_impact_event_in_next_24h'] = ((time_diff >= pd.Timedelta(0)) & (time_diff <= one_day)).astype(int)
        self._log("    - Added 'is_high_impact_event_in_next_24h' feature.")

        # --- News Filter based on event proximity ---
        self._log("  - Calculating 60-min news filter around high-impact events...")
        merged_df_prev = pd.merge_asof(
            self.df.sort_index(),
            high_impact_events.add_suffix('_event_prev'),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        time_diff_prev = merged_df_prev.index - merged_df_prev['timestamp_event_prev']
        one_hour = pd.Timedelta(minutes=60)
        after_event = (time_diff_prev >= pd.Timedelta(0)) & (time_diff_prev <= one_hour)
        before_event = (time_diff >= pd.Timedelta(0)) & (time_diff <= one_hour)
        self.df['news_filter_60min'] = (after_event | before_event).astype(int)
        self._log("    - Added 'news_filter_60min' feature.")

        self._log("  - Calendar feature step complete.")
        return self

    def check_and_transform_stationarity(self):
        self._log("\n[Bonus Step] Checking for Feature Stationarity")
        # Exclude columns that are known to be non-stationary or are needed in their raw form later.
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'EMA_200', 'ATRr_14']
        self._log(f"  - Excluding columns from check: {exclude_cols}")

        columns_to_check = [c for c in self.df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(self.df[c])]
        self._log(f"  - Found {len(columns_to_check)} numeric columns to test for stationarity.")

        for col in columns_to_check:
            self._log(f"\n  --- Testing column: '{col}' ---")
            p_value = self.test_stationarity(self.df[col])
            if p_value > 0.05:
                self._log(f"  - Transformation: Column '{col}' is non-stationary, applying '.pct_change()'.")
                self.df[f'{col}_pct_change'] = self.df[col].pct_change()
                self.df.drop(columns=[col], inplace=True)
                self._log(f"  - Dropped original column '{col}' and added '{col}_pct_change'.")
            else:
                self._log(f"  - Transformation: Column '{col}' is stationary, no transformation needed.")

        self._log("\n  - Stationarity check and transformation complete.")
        return self

    def test_stationarity(self, series: pd.Series):
        """
        Performs the Augmented Dickey-Fuller test on a series.
        Returns the p-value.
        """
        series = series.dropna()
        # A minimum number of observations is required for the ADF test. 20 is a safe floor.
        if len(series) < 20:
            self._log(f"  -> Series has too few observations ({len(series)}) for ADF test. Marking as non-stationary by default.")
            return 1.0
        if series.nunique() < 2:
            self._log(f"  -> Series is constant. Marking as stationary to prevent transformation.")
            return 0.0
        try:
            # Use AIC to automatically select the optimal lag.
            self._log(f"  -> Running ADF test with autolag='AIC' on {len(series)} observations...")
            result = adfuller(series, autolag='AIC')
            p_value = result[1]
            test_statistic = result[0]
            critical_values = result[4]

            self._log(f"  -> ADF test complete. Lags used: {result[2]}")
            self._log(f"     - Test Statistic: {test_statistic:.4f}")
            self._log(f"     - P-value: {p_value:.4f}")
            self._log(f"     - Critical Values: { {k: f'{v:.4f}' for k, v in critical_values.items()} }")

            if p_value > 0.05:
                self._log(f"     - Result: Series is likely NON-STATIONARY (p > 0.05)")
            else:
                self._log(f"     - Result: Series is likely STATIONARY (p <= 0.05)")

            return p_value
        except Exception as e:
            self._log(f"  -> ADF test failed for series. Error: {e}. Marking as non-stationary.")
            return 1.0

    def _cleanup_raw_data(self):
        """Removes raw price columns to prevent lookahead bias or non-stationarity issues."""
        self._log("\n[Bonus Step] Cleaning up raw price data...")

        cols_before = self.df.columns.tolist()
        self._log(f"  - Columns before cleanup: {len(cols_before)}")

        # Note: 'volume' is kept for indicators like OBV and MFI, and its stationarity
        # is handled by the check_and_transform_stationarity step.
        # We KEEP 'close' as it's needed for the labeling step.
        cols_to_drop = ['open', 'high', 'low']

        # We only want to drop columns that actually exist in the dataframe
        existing_cols_to_drop = [col for col in cols_to_drop if col in self.df.columns]

        self._log(f"  - Attempting to drop columns: {cols_to_drop}")

        if not existing_cols_to_drop:
            self._log("  - No columns to drop were found in the dataframe. Skipping.")
            return self

        self.df.drop(columns=existing_cols_to_drop, inplace=True, errors='ignore')

        cols_after = self.df.columns.tolist()
        self._log(f"  - Successfully dropped: {existing_cols_to_drop}")
        self._log(f"  - Columns after cleanup: {len(cols_after)}")

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
