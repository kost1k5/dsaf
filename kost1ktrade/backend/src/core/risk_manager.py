"""
Risk Management Module
This module provides functions for calculating position sizes based on risk parameters.
"""

def calculate_position_size(
    capital: float,
    risk_per_trade_pct: float,
    atr_value: float,
    atr_multiplier: float,
    price: float
) -> float:
    """
    Calculates the position size in the base currency based on volatility (ATR).

    This method normalizes the risk of each trade to a fixed percentage of the total capital,
    regardless of market volatility.

    Args:
        capital (float): The total trading capital available.
        risk_per_trade_pct (float): The percentage of capital to risk on a single trade (e.g., 1 for 1%).
        atr_value (float): The current Average True Range (ATR) value for the asset.
        atr_multiplier (float): The multiplier for ATR to determine the stop-loss distance.
        price (float): The current price of the asset, used to convert the position size from quote to base currency.

    Returns:
        float: The calculated position size in the number of contracts/shares.
    """
    if atr_value <= 0 or atr_multiplier <= 0 or price <= 0:
        return 0.0

    # Convert percentage to a decimal
    risk_decimal = risk_per_trade_pct / 100.0

    # 1. Calculate the total risk amount in the quote currency (e.g., USD)
    risk_amount_in_quote = capital * risk_decimal

    # 2. Calculate the stop-loss distance in the quote currency
    stop_loss_distance = atr_value * atr_multiplier

    # 3. Calculate the position size in the quote currency
    # This is how much of the quote currency we should hold.
    # For crypto futures, this is often the value we need.
    # However, for spot or other instruments, we need the size in the base asset.
    position_size_quote = risk_amount_in_quote / stop_loss_distance

    # 4. Convert the position size from quote currency to base currency (e.g., BTC, ETH)
    position_size_base = position_size_quote / price

    return position_size_base
