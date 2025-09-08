"""
Cointegrated Pairs Finder

This script is a standalone tool to find statistically cointegrated pairs of assets,
which are potential candidates for a pairs trading (statistical arbitrage) strategy.

The script prints the results to the console but does NOT automatically configure
or start the Pairs Trading Bot.

Workflow:
1. Run this script to discover cointegrated pairs (e.g., `pdm run python scripts/find_cointegrated_pairs.py`).
2. Note the promising pairs from the output (e.g., ('ETH/USDT', 'LTC/USDT')).
3. Manually use the API endpoint `/api/pairs-bot/start` with the chosen pair to start the bot.
"""
import pandas as pd
import numpy as np
import argparse
import sys
import os
from itertools import combinations
from statsmodels.tsa.stattools import coint

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collector.data_cacher import DataCacher
from datetime import datetime, timedelta

def find_cointegrated_pairs(symbols: list, timeframe: str, start_date: str, end_date: str, corr_threshold: float = 0.9):
    """
    Finds cointegrated pairs of assets from a given list of symbols.

    1. Fetches historical data for all symbols.
    2. Calculates the correlation matrix.
    3. For highly correlated pairs, performs the Engle-Granger cointegration test.
    4. Prints the pairs that are statistically cointegrated.
    """
    print("--- Starting Cointegration Analysis ---")

    # 1. Fetch data for all symbols
    cacher = DataCacher(db_path='data/historical_data.db')
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    all_prices = {}
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        df = cacher.fetch_and_cache_data(symbol, timeframe, start_dt, end_dt)
        if not df.empty:
            all_prices[symbol] = df['close']

    cacher.close()

    if len(all_prices) < 2:
        print("Need at least two symbols with data to find pairs.")
        return

    # 2. Create a single DataFrame with all close prices
    price_df = pd.DataFrame(all_prices).dropna()
    print(f"\nCreated price matrix with shape: {price_df.shape}")

    # 3. Calculate correlation matrix
    corr_matrix = price_df.corr()
    print("\nCorrelation Matrix (Top 5):")
    print(corr_matrix.head())

    # 4. Find highly correlated pairs
    # Get the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    highly_correlated_pairs = [
        column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)
    ]

    potential_pairs = []
    for col in highly_correlated_pairs:
        potential_pairs.extend(
            [(col, row) for row in upper_tri.index[upper_tri[col] > corr_threshold]]
        )

    if not potential_pairs:
        print(f"\nNo pairs found with correlation > {corr_threshold}. Try a lower threshold.")
        return

    print(f"\nFound {len(potential_pairs)} potential pairs with correlation > {corr_threshold}. Testing for cointegration...")

    # 5. Test for cointegration
    cointegrated_pairs = []
    for pair in potential_pairs:
        symbol1, symbol2 = pair
        series1 = price_df[symbol1]
        series2 = price_df[symbol2]

        # The result contains: t-statistic, p-value, critical values
        score, pvalue, _ = coint(series1, series2)

        if pvalue < 0.05:
            cointegrated_pairs.append({
                "pair": (symbol1, symbol2),
                "p_value": pvalue,
                "correlation": corr_matrix.loc[symbol1, symbol2]
            })

    # 6. Report results
    print("\n--- Cointegration Test Results ---")
    if cointegrated_pairs:
        print(f"Found {len(cointegrated_pairs)} cointegrated pairs (p-value < 0.05):")
        # Sort by p-value for relevance
        sorted_pairs = sorted(cointegrated_pairs, key=lambda x: x['p_value'])
        for item in sorted_pairs:
            print(f"  - Pair: {item['pair'][0]} / {item['pair'][1]}, P-value: {item['p_value']:.4f}, Correlation: {item['correlation']:.2f}")
    else:
        print("No statistically significant cointegrated pairs were found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find cointegrated pairs for statistical arbitrage.")

    # A default list of major assets to check
    default_symbols = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT"
    ]

    end_date_default = datetime.now().strftime('%Y-%m-%d')
    start_date_default = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    parser.add_argument("--symbols", nargs='+', default=default_symbols, help="List of trading symbols to test.")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe for candles.")
    parser.add_argument("--start_date", type=str, default=start_date_default, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default=end_date_default, help="End date (YYYY-MM-DD).")
    parser.add_argument("--corr", type=float, default=0.9, help="Minimum correlation threshold to test for cointegration.")

    args = parser.parse_args()

    find_cointegrated_pairs(
        symbols=args.symbols,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        corr_threshold=args.corr
    )
