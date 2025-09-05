import pandas as pd
import pandas_ta as ta
from statsmodels.tsa.stattools import adfuller

class FeatureGenerator:
    """
    A class to generate a wide array of features for the trading model.
    It takes various dataframes as input and produces a single dataframe with all features.
    """
    def __init__(self, ohlcv_df: pd.DataFrame, timeframe: str,
                 ohlcv_df_4h: pd.DataFrame = None, ohlcv_df_1d: pd.DataFrame = None,
                 open_interest_df: pd.DataFrame = None, funding_rate_df: pd.DataFrame = None,
                 macro_df: pd.DataFrame = None, fng_df: pd.DataFrame = None,
                 eth_ohlcv_df: pd.DataFrame = None):
        """
        Initializes the FeatureGenerator with all necessary dataframes.
        The main dataframe `self.df` is based on the ohlcv data.
        """
        self.timeframe = timeframe
        # Ensure dataframes are sorted by time and have a timestamp index
        self.df = ohlcv_df.set_index('timestamp').sort_index().copy()
        self.ohlcv_4h = ohlcv_df_4h.set_index('timestamp').sort_index() if ohlcv_df_4h is not None else None
        self.ohlcv_1d = ohlcv_df_1d.set_index('timestamp').sort_index() if ohlcv_df_1d is not None else None
        self.eth_ohlcv = eth_ohlcv_df.set_index('timestamp').sort_index() if eth_ohlcv_df is not None else None
        self.open_interest = open_interest_df.set_index('timestamp').sort_index() if open_interest_df is not None else None
        self.funding_rate = funding_rate_df.set_index('timestamp').sort_index() if funding_rate_df is not None else None
        self.macro = macro_df.set_index(pd.to_datetime(macro_df['Date'])).sort_index() if macro_df is not None else None
        self.fng = fng_df.set_index(pd.to_datetime(fng_df.index)).sort_index() if fng_df is not None else None

    def add_technical_indicators(self):
        """
        Calculates and adds technical indicators using the pandas-ta library.
        (Б, В) This has been expanded to include multiple time windows and new indicators.
        """
        print("Adding technical indicators with multiple time windows...")

        # (В) Add multiple time windows for key indicators
        for length in [7, 14, 21]:
            self.df.ta.rsi(length=length, append=True)
            self.df.ta.adx(length=length, append=True)
            self.df.ta.ppo(fast=length, slow=length*2, append=True)

        # (Б.1) Add Volatility and Volume indicators
        # ATR (already uses a default length of 14, which is fine)
        self.df.ta.atr(append=True)

        # Bollinger Bands (BBB is Band Width)
        self.df.ta.bbands(length=20, append=True)

        # On-Balance Volume
        self.df.ta.obv(append=True)

        # CMF
        self.df.ta.cmf(append=True)

        # VWAP and distance from it
        self.df.ta.vwap(append=True)
        # The default column name for vwap is VWAP_D
        if 'VWAP_D' in self.df.columns:
            self.df['dist_from_vwap'] = (self.df['close'] / self.df['VWAP_D']) - 1

        print(f"  - Generated {len(self.df.columns)} columns so far.")
        return self

    def add_multi_timeframe_features(self):
        """
        Adds technical indicators from higher timeframes (4h, 1d) to the base dataframe.
        """
        print("Adding multi-timeframe features...")
        for tf_df, tf_name in [(self.ohlcv_4h, "4h"), (self.ohlcv_1d, "1d")]:
            if tf_df is None or tf_df.empty:
                print(f"  - Skipping {tf_name} timeframe, data not available.")
                continue

            # Calculate indicators on the higher timeframe dataframe
            tf_df.ta.rsi(append=True)
            tf_df.ta.ppo(append=True)

            # Select and rename the indicator columns to avoid name clashes
            indicators = tf_df[['RSI_14', 'PPO_12_26_9', 'PPOh_12_26_9', 'PPOs_12_26_9']]
            indicators = indicators.add_suffix(f'_{tf_name}')

            # Merge the HTF indicators onto the base dataframe
            self.df = pd.merge_asof(self.df, indicators, left_index=True, right_index=True, direction='backward')
            print(f"  - Merged {tf_name} indicators.")

        return self

    def add_derivative_features(self):
        """
        Calculates and adds features based on open interest and funding rates.
        Uses merge_asof for robust joining of sparse data.
        """
        print("Adding derivative features...")
        if self.open_interest is not None and not self.open_interest.empty and 'oi_value' in self.open_interest.columns:
            # The 'oi_value' column is already standardized by the collection script.
            self.df = pd.merge_asof(self.df, self.open_interest[['oi_value']], left_index=True, right_index=True, direction='backward')
            # Forward-fill is okay to propagate last known value, but back-filling introduces lookahead bias.
            # The model will learn to handle NaNs for periods where no data was available.
            self.df['oi_value'] = self.df['oi_value'].ffill()
            self.df['oi_pct_change'] = self.df['oi_value'].pct_change()

        if self.funding_rate is not None and not self.funding_rate.empty:
            fr_series = self.funding_rate[['fundingRate']].rename(columns={'fundingRate': 'funding_rate'})
            self.df = pd.merge_asof(self.df, fr_series, left_index=True, right_index=True, direction='backward')
            # Forward-fill is okay, back-filling is not.
            self.df['funding_rate'] = self.df['funding_rate'].ffill()
            self.df['funding_rate_mom'] = self.df['funding_rate'].diff(periods=3)

        return self

    def add_sentiment_features(self):
        """
        Calculates and adds features based on sentiment data.
        """
        print("Adding sentiment features...")
        # Add Fear & Greed Index
        if self.fng is not None and not self.fng.empty:
            self.df = pd.merge_asof(self.df, self.fng['fng_value'], left_index=True, right_index=True, direction='backward')
            self.df['fng_value'] = self.df['fng_value'].ffill()

        return self

    def add_cross_market_features(self):
        """
        Calculates and adds features based on cross-market correlations.
        """
        print("Adding cross-market features...")
        if self.macro is not None:
            # Join macro data, forward-filling for weekends/holidays
            self.df = pd.merge_asof(self.df, self.macro[['SPY', 'VIX', 'DXY']], left_index=True, right_index=True, direction='backward')

        if self.eth_ohlcv is not None and not self.eth_ohlcv.empty:
            print("  - Calculating ETH correlation...")
            # Calculate returns for both assets
            eth_returns = self.eth_ohlcv['close'].pct_change().rename('eth_returns')
            asset_returns = self.df['close'].pct_change().rename('asset_returns')

            # Merge ETH returns onto the main df
            merged_returns = pd.merge_asof(asset_returns.to_frame(), eth_returns.to_frame(), left_index=True, right_index=True, direction='backward')

            # Calculate rolling correlation (e.g., 30-day window on 1h data)
            window = 24 * 30
            self.df['corr_eth_30d'] = merged_returns['asset_returns'].rolling(window=window).corr(merged_returns['eth_returns'])

        return self

    def add_time_based_features(self):
        """
        Adds time-based features like hour of day, day of week, and volatility regimes.
        """
        print("Adding time-based features...")
        self.df['hour_of_day'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek

        # Add volatility feature
        # We need to parse the timeframe to make the window dynamic
        # For simplicity, we assume 'h' for hours. A more robust parser would be needed for 'm', 'd', etc.
        if 'h' in self.timeframe:
            try:
                hours = int(self.timeframe.replace('h', ''))
                window_24h = 24 // hours
                self.df['volatility_24h'] = self.df['close'].pct_change().rolling(window=window_24h).std()
            except ValueError:
                print(f"Warning: Could not parse timeframe '{self.timeframe}' to calculate volatility. Skipping.")
        else:
             # Default to a 24 period window if timeframe is not in hours (e.g. '1d')
            self.df['volatility_24h'] = self.df['close'].pct_change().rolling(window=24).std()

        return self

    def add_lag_features(self):
        """
        Adds lagged versions of important features to provide historical context.
        """
        print("Adding lag features...")
        lags = [3, 6, 12, 24] # e.g., for 1h timeframe, this is 3h, 6h, 12h, 24h ago

        # Define columns to lag
        # We lag the percentage change of price, not the price itself, to maintain stationarity.
        self.df['close_pct_change'] = self.df['close'].pct_change()
        cols_to_lag = ['close_pct_change', 'volume', 'oi_pct_change', 'funding_rate']

        for col in cols_to_lag:
            if col in self.df.columns:
                for lag in lags:
                    self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)

        # We can drop the original close_pct_change as it's now represented by its lags
        self.df.drop(columns=['close_pct_change'], inplace=True, errors='ignore')

        return self

    def add_interaction_features(self):
        """
        Creates features by combining or transforming existing ones.
        """
        print("Adding interaction features...")
        if 'RSI_14' in self.df.columns and 'ATRr_14' in self.df.columns:
            # Example: RSI normalized by volatility
            atr_normalized = self.df['ATRr_14'] / self.df['close']
            # Avoid division by zero
            atr_normalized[atr_normalized == 0] = 1e-6
            self.df['rsi_vol_norm'] = self.df['RSI_14'] * (1 / atr_normalized)

        return self

    def check_and_transform_stationarity(self):
        """
        Checks all feature columns for stationarity and applies .pct_change() if not stationary.
        """
        print("Checking for stationarity...")
        # List of columns to exclude from stationarity check (e.g., identifiers, original prices)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']

        # Iterate over a copy of column names as we might modify the dataframe
        for col in self.df.columns.copy():
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.df[col]):
                # A constant series will cause the ADF test to fail and should not be transformed.
                # We skip it here entirely.
                if self.df[col].nunique() < 2:
                    print(f"  -> WARNING: Column '{col}' is constant. Skipping stationarity test and transformation.")
                    continue

                print(f"Testing stationarity of: {col}")
                p_value = self.test_stationarity(self.df[col].dropna())
                if p_value > 0.05:
                    print(f"  -> Column '{col}' is not stationary (p-value: {p_value:.4f}). Applying transformation.")
                    self.df[f'{col}_pct_change'] = self.df[col].pct_change()
                    self.df.drop(columns=[col], inplace=True)
        return self

    def test_stationarity(self, series: pd.Series):
        """
        Performs the Augmented Dickey-Fuller test on a series.
        Assumes NaNs and constant series have already been handled.
        Returns the p-value.
        """
        if len(series) < 20: # Not enough data to test
            return 0.0 # Assume stationary if not enough data

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

        print("Feature generation complete.")
        return self.df
