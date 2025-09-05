"""
A collection of shared utility functions used across the application.
"""

def sanitize_symbol(symbol: str) -> str:
    """Converts a symbol like 'BTC/USDT' to 'BTC_USDT' for filenames."""
    return symbol.replace('/', '_')

def parse_asset_from_symbol(symbol: str) -> str:
    """Converts 'BTC/USDT' or 'BTC-USDT' to 'BTC'."""
    return symbol.split('/')[0].split('-')[0]
