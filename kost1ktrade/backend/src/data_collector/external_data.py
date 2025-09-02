import pandas as pd
import os
import requests
import json
from datetime import datetime, timezone

def get_fear_and_greed_index(limit: int = 365 * 2) -> pd.DataFrame:
    """
    Fetches historical Fear & Greed Index data from the alternative.me API.

    Args:
        limit (int): Number of days to fetch.

    Returns:
        pd.DataFrame: DataFrame with 'date' and 'fng_value'.
    """
    print(f"Fetching last {limit} days of Fear & Greed Index...")
    try:
        url = f"https://api.alternative.me/fng/?limit={limit}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        df = pd.DataFrame(data['data'])
        df['fng_value'] = pd.to_numeric(df['value'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
        return df[['date', 'fng_value']]

    except requests.exceptions.RequestException as e:
        print(f"Could not fetch Fear & Greed Index data due to a network error: {e}")
        return pd.DataFrame(columns=['date', 'fng_value'])
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Could not parse Fear & Greed Index data: {e}")
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
