import pandas as pd
import numpy as np

def manual_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ADX manually."""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr))

    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

# Define thresholds for ADX interpretation
ADX_TREND_STRONG = 25
ADX_TREND_WEAK = 20

def get_market_state(candles_df: pd.DataFrame, adx_period: int = 14) -> (str, float):
    """
    Analyzes the market state using the Average Directional Index (ADX).
    """
    if not all(col in candles_df.columns for col in ['high', 'low', 'close']):
        raise ValueError("Candles DataFrame must contain 'high', 'low', and 'close' columns.")

    if len(candles_df) < adx_period * 2:
        return "Not Enough Data", 0.0

    # Calculate ADX manually
    adx_series = manual_adx(candles_df['high'], candles_df['low'], candles_df['close'], period=adx_period)
    latest_adx = adx_series.iloc[-1]

    if pd.isna(latest_adx):
        return "Not Enough Data", 0.0

    # Interpret the ADX value
    if latest_adx > ADX_TREND_STRONG:
        market_state = "Trending"
    elif latest_adx < ADX_TREND_WEAK:
        market_state = "Ranging"
    else:
        market_state = "Weak Trend"

    return market_state, round(latest_adx, 2)
