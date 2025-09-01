import pandas as pd
import fear_greed_index
import os
from datetime import datetime, timezone

def get_fear_and_greed_index(limit: int = 365 * 2) -> pd.DataFrame:
    """
    Fetches historical Fear & Greed Index data.

    Args:
        limit (int): Number of days to fetch.

    Returns:
        pd.DataFrame: DataFrame with 'date' and 'fng_value'.
    """
    print(f"Fetching last {limit} days of Fear & Greed Index...")
    try:
        fng_data = fear_greed_index.get(limit, as_json=True)
        df = pd.DataFrame(fng_data)
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
        df = df.rename(columns={'value': 'fng_value'})
        return df[['date', 'fng_value']]
    except Exception as e:
        print(f"Could not fetch Fear & Greed Index data: {e}")
        return pd.DataFrame(columns=['date', 'fng_value'])

def get_onchain_metrics(days_back: int = 365 * 2) -> pd.DataFrame:
    """
    Placeholder function to fetch on-chain metrics.
    A real implementation would require an API key from a provider like Glassnode,
    CryptoQuant, or Santiment.

    Args:
        days_back (int): How many days of on-chain data to fetch.

    Returns:
        pd.DataFrame: DataFrame with 'date' and on-chain metric columns.
    """
    api_key = os.getenv('ONCHAIN_API_KEY') # Example: GLASSNODE_API_KEY
    if not api_key:
        print("Warning: ONCHAIN_API_KEY not found. Skipping on-chain metrics. "
              "This is a placeholder function. You need to implement it with your own data provider.")
        return pd.DataFrame(columns=['date', 'net_exchange_flow', 'sopr', 'mvrv'])

    #
    # --- Placeholder for actual API call ---
    # Example:
    # client = OnChainProviderClient(api_key)
    # data = client.get_metrics(['net_exchange_flow', 'sopr', 'mvrv'], days=days_back)
    # df = pd.DataFrame(data)
    # df['date'] = pd.to_datetime(df['timestamp']).dt.date
    # return df
    #
    print("Note: `get_onchain_metrics` is a placeholder and returned no data.")
    return pd.DataFrame(columns=['date', 'net_exchange_flow', 'sopr', 'mvrv'])


if __name__ == '__main__':
    # Example usage:
    fng_df = get_fear_and_greed_index(limit=30)
    print("\nFear & Greed Data:")
    print(fng_df.head())

    onchain_df = get_onchain_metrics()
    print("\nOn-chain Data (Placeholder):")
    print(onchain_df.head())
