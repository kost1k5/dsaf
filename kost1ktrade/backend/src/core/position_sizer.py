import math

def calculate_position_size(
    account_equity: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float,
    contract_size: float,
    instrument_type: str = 'SWAP'
) -> int | None:
    """
    Calculates the number of contracts for a trade based on a fixed-fractional risk model.

    Args:
        account_equity: The total current equity of the trading account.
        risk_percent: The fraction of account equity to risk on this trade (e.g., 0.01 for 1%).
        entry_price: The expected entry price of the trade.
        stop_loss_price: The price at which the stop-loss order will be placed.
        contract_size: The value of a single contract in the base currency (e.g., 0.001 for BTC-USDT-SWAP).
        instrument_type: The type of instrument, e.g., 'SWAP' or 'SPOT'. Currently only SWAP is handled.

    Returns:
        The calculated number of contracts to trade as an integer, or None if inputs are invalid.
    """
    if risk_percent <= 0 or risk_percent > 1:
        print("Error: Risk percent must be between 0 and 1.")
        return None

    if entry_price == stop_loss_price:
        print("Error: Entry price cannot be the same as stop-loss price.")
        return None

    # 1. Calculate the total risk in USD for this trade
    risk_in_usd = account_equity * risk_percent

    # 2. Calculate the price distance for the stop-loss
    price_risk_per_unit = abs(entry_price - stop_loss_price)

    # 3. Calculate the position size in the base currency (e.g., in BTC)
    # This tells us how much of the asset we can buy/sell given our risk tolerance
    position_size_in_base = risk_in_usd / price_risk_per_unit

    # 4. Convert the position size from base currency to number of contracts
    if instrument_type == 'SWAP':
        # For swaps/futures, the order size is in number of contracts
        number_of_contracts = position_size_in_base / contract_size
    else:
        # For SPOT, the order size might be in the base currency itself.
        # This logic can be expanded if needed.
        print(f"Unsupported instrument type for sizing: {instrument_type}")
        return None

    # Return as an integer, as we can only trade whole contracts
    return math.floor(number_of_contracts)

if __name__ == '__main__':
    print("--- Testing Position Sizer ---")

    # Using the example from the user's documentation
    test_equity = 10000.0  # USD
    test_risk_pct = 0.01   # 1%
    test_entry = 60000.0   # USD
    test_stop = 59500.0    # USD
    test_contract_size = 0.001 # BTC per contract for BTC-USDT-SWAP

    print(f"\nTest Case 1: Long Position")
    print(f"Account Equity: ${test_equity:,.2f}")
    print(f"Risk: {test_risk_pct * 100}%")
    print(f"Entry: ${test_entry:,.2f}, Stop: ${test_stop:,.2f}")
    print(f"Contract Size: {test_contract_size} BTC")

    calculated_size = calculate_position_size(
        account_equity=test_equity,
        risk_percent=test_risk_pct,
        entry_price=test_entry,
        stop_loss_price=test_stop,
        contract_size=test_contract_size
    )

    if calculated_size is not None:
        risk_in_usd = test_equity * test_risk_pct
        price_risk = abs(test_entry - test_stop)
        position_in_btc = (calculated_size * test_contract_size)
        total_risk_of_trade = position_in_btc * price_risk

        print(f"\nCalculated Number of Contracts: {calculated_size}")
        print(f"This corresponds to a position size of {position_in_btc:.4f} BTC.")
        print(f"Max risk per trade was set to: ${risk_in_usd:,.2f}")
        print(f"Actual risk for this trade size is: ${total_risk_of_trade:,.2f}")
        assert calculated_size == 200, "Calculation does not match example!"
        print("\nAssertion passed: Calculation matches the example in the documentation.")
    else:
        print("\nPosition size calculation failed.")
