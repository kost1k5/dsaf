import pandas as pd
import pandas_ta as ta
from statsmodels.tsa.stattools import adfuller
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class FeatureGenerator:
    """
    A class to generate a wide array of features for the trading model.
    It takes various dataframes as input and produces a single dataframe with all features.
    """
    def __init__(self, ohlcv_df: pd.DataFrame, timeframe: str, open_interest_df: pd.DataFrame = None,
                 funding_rate_df: pd.DataFrame = None, macro_df: pd.DataFrame = None,
                 fng_df: pd.DataFrame = None, news_df: pd.DataFrame = None):
        """
        Initializes the FeatureGenerator with all necessary dataframes.
        The main dataframe `self.df` is based on the ohlcv data.
        """
        self.timeframe = timeframe
        # Ensure dataframes are sorted by time
        self.df = ohlcv_df.set_index('timestamp').sort_index().copy()
        self.open_interest = open_interest_df.set_index('timestamp').sort_index() if open_interest_df is not None else None
        self.funding_rate = funding_rate_df.set_index('timestamp').sort_index() if funding_rate_df is not None else None
        self.macro = macro_df.set_index(pd.to_datetime(macro_df['Date'])).sort_index() if macro_df is not None else None
        self.fng = fng_df.set_index(pd.to_datetime(fng_df.index)).sort_index() if fng_df is not None else None
        self.news = news_df.set_index(pd.to_datetime(news_df['published'])).sort_index() if news_df is not None else None

    def add_technical_indicators(self):
        """
        Calculates and adds technical indicators using the pandas-ta library.
        """
        print("Adding technical indicators...")
        # Use the main dataframe's OHLCV data
        self.df.ta.atr(append=True)
        self.df.ta.ppo(append=True)
        self.df.ta.rsi(append=True)
        self.df.ta.vwap(append=True)
        self.df.ta.cmf(append=True)
        self.df.ta.adx(append=True)

        return self

    def add_derivative_features(self):
        """
        Calculates and adds features based on open interest and funding rates.
        Uses merge_asof for robust joining of sparse data.
        """
        print("Adding derivative features...")
        if self.open_interest is not None and not self.open_interest.empty:
            oi_series = self.open_interest[['openInterestValue']].rename(columns={'openInterestValue': 'oi_value'})
            self.df = pd.merge_asof(self.df, oi_series, left_index=True, right_index=True, direction='backward')
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
            self.df['fng_value'] = self.df['fng_value'].ffill().bfill()

        # Add VADER sentiment from news headlines
        if self.news is not None and not self.news.empty:
            analyzer = SentimentIntensityAnalyzer()
            self.news['title'] = self.news['title'].astype(str)
            self.news['news_sentiment'] = self.news['title'].apply(lambda title: analyzer.polarity_scores(title)['compound'])
            daily_sentiment = self.news[['news_sentiment']].resample('D').mean()
            self.df = pd.merge_asof(self.df, daily_sentiment, left_index=True, right_index=True, direction='backward')
            # Propagate last known sentiment. This assumes sentiment persists until new news arrives.
            self.df['news_sentiment'] = self.df['news_sentiment'].ffill().bfill()

        return self

    def add_cross_market_features(self):
        """
        Calculates and adds features based on cross-market correlations.
        """
        print("Adding cross-market features...")
        if self.macro is not None:
            # Join macro data, forward-filling for weekends/holidays
            self.df = pd.merge_asof(self.df, self.macro[['SPY', 'VIX', 'DXY']], left_index=True, right_index=True, direction='backward')

            # Calculate rolling correlations - DISABLED due to large lookback window causing issues with data size.
            # self.df['corr_spy_30d'] = self.df['close'].rolling(window=24*30).corr(self.df['SPY'])
            # self.df['corr_vix_30d'] = self.df['close'].rolling(window=24*30).corr(self.df['VIX'])

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
                print(f"Testing stationarity of: {col}")
                p_value = self.test_stationarity(self.df[col])
                if p_value > 0.05:
                    print(f"  -> Column '{col}' is not stationary (p-value: {p_value:.4f}). Applying transformation.")
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
        self.add_derivative_features()
        self.add_sentiment_features()
        self.add_cross_market_features()
        self.add_time_based_features()
        self.add_interaction_features()

        # Final check for stationarity
        self.check_and_transform_stationarity()

        # Drop any remaining rows with NaNs
        self.df.dropna(inplace=True)

        print("Feature generation complete.")
        return self.df
