import pandas as pd
import pandas_ta as ta
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
                 eth_ohlcv_df: pd.DataFrame = None):
        """
        Initializes the FeatureGenerator with all necessary dataframes.
        The main dataframe `self.df` is based on the ohlcv data.
        """
        log_file_path = os.path.join(os.path.dirname(__file__), '..', '..', f'indicator_log_{asset}.txt')
        # Open in 'w' mode to clear the log for this specific asset on each run
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self._log(f"--- Starting Feature Generation for {asset} ---")

        self.timeframe = timeframe

        # Standardize all dataframes
        self.df = self._standardize_ohlcv(ohlcv_df, "ohlcv_df").copy()
        if self.df is None:
            raise ValueError("Primary OHLCV dataframe is missing or invalid.")

        self.ohlcv_4h = self._standardize_ohlcv(ohlcv_df_4h, "ohlcv_4h")
        self.ohlcv_1d = self._standardize_ohlcv(ohlcv_df_1d, "ohlcv_1d")
        self.eth_ohlcv = self._standardize_ohlcv(eth_ohlcv_df, "eth_ohlcv")
        self.funding_rate = self._standardize_funding_rate(funding_rate_df)
        self.macro = self._standardize_macro_data(macro_df)
        self.fng = self._standardize_generic(fng_df, "fng_df", ['timestamp', 'date'])
        self.news = self._standardize_generic(news_df, "news_df", ['published_at', 'published', 'timestamp'])

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

    def add_technical_indicators(self):
        self._log("\n[Step 1/8] Adding Technical Indicators")
        for length in [7, 14, 21]:
            self._log(f"  - Calculating RSI, ADX, PPO for length: {length}")
            self.df.ta.rsi(length=length, append=True)
            self.df.ta.adx(length=length, append=True)
            self.df.ta.ppo(fast=length, slow=length*2, append=True)
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
        if 'VWAP_D' in self.df.columns:
            self.df['dist_from_vwap'] = (self.df['close'] / self.df['VWAP_D']) - 1
            self._log("  - Calculating distance from VWAP")
        self._log(f"  - Generated {len(self.df.columns)} columns so far.")
        return self

    def add_multi_timeframe_features(self):
        self._log("\n[Step 2/8] Adding Multi-Timeframe Features")
        for tf_df, tf_name in [(self.ohlcv_4h, "4h"), (self.ohlcv_1d, "1d")]:
            if tf_df is None or tf_df.empty:
                self._log(f"  - Skipping {tf_name} timeframe, data not available.")
                continue
            self._log(f"  - Calculating indicators for {tf_name} timeframe (RSI, PPO)")
            tf_df.ta.rsi(append=True)
            tf_df.ta.ppo(append=True)
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
        self._log("\n[Step 5/8] Adding Cross-Market Features")
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
        self._log("\n[Step 6/8] Adding Time-Based Features")
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
        self._log("\n[Step 7/8] Adding Lag Features")
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

    def add_interaction_features(self):
        self._log("\n[Step 8/8] Adding Interaction Features")
        if 'RSI_14' in self.df.columns and 'ATRr_14' in self.df.columns:
            self._log("  - Creating 'rsi_vol_norm' (RSI normalized by ATR).")
            atr_normalized = self.df['ATRr_14'] / self.df['close']
            atr_normalized[atr_normalized == 0] = 1e-6
            self.df['rsi_vol_norm'] = self.df['RSI_14'] * (1 / atr_normalized)
        else:
            self._log("  - Skipping 'rsi_vol_norm' (required columns not found).")
        return self

    def check_and_transform_stationarity(self):
        self._log("\n[Bonus Step] Checking for Feature Stationarity")
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
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
        if len(series) < 20:
            return 0.0
        if series.nunique() < 2:
            self._log(f"  -> Series is constant. Marking as non-stationary.")
            return 1.0
        try:
            result = adfuller(series)
            return result[1]
        except Exception as e:
            self._log(f"  -> ADF test failed for series. Error: {e}. Marking as non-stationary.")
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
        self.check_and_transform_stationarity()
        self._log("\n--- Feature Generation Complete ---")
        self.log_file.close()
        print("Feature generation complete.")
        return self.df
