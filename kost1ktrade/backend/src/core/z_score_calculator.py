import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.database.session import SessionLocal
from src.database.models import EconomicCalendarEvent

def calculate_z_score(event_name: str, actual: float, forecast: float, history_lookback: int = 12) -> float | None:
    """
    Calculates the Z-score for a given economic event surprise.

    The Z-score measures how many standard deviations an event's "surprise"
    (actual - forecast) is from the mean of historical surprises for that same event.

    Args:
        event_name: The name of the economic event (e.g., "Non-Farm Payrolls").
        actual: The actual released value of the indicator.
        forecast: The forecasted value of the indicator.
        history_lookback: The number of past events to use for the historical sample.

    Returns:
        The calculated Z-score as a float, or None if there is insufficient
        historical data or the standard deviation is zero.
    """
    db: Session = SessionLocal()
    try:
        # Fetch the last 'history_lookback' events of the same name that have both actual and forecast values
        historical_events = (
            db.query(EconomicCalendarEvent)
            .filter(
                EconomicCalendarEvent.event_name == event_name,
                EconomicCalendarEvent.actual.isnot(None),
                EconomicCalendarEvent.forecast.isnot(None)
            )
            .order_by(desc(EconomicCalendarEvent.event_datetime))
            .limit(history_lookback)
            .all()
        )

        # We need at least 2 data points to calculate a standard deviation
        if len(historical_events) < 2:
            print(f"Insufficient historical data for '{event_name}' (found {len(historical_events)} records). Cannot calculate Z-score.")
            return None

        # Calculate the historical "surprises"
        surprises = [float(e.actual) - float(e.forecast) for e in historical_events]

        # Calculate the standard deviation of the historical surprises
        std_dev = np.std(surprises)

        if std_dev == 0:
            print(f"Standard deviation of surprises for '{event_name}' is zero. Cannot calculate Z-score.")
            return None

        # Calculate the current surprise
        current_surprise = actual - forecast

        # Calculate the Z-score
        z_score = current_surprise / std_dev

        print(f"Calculated Z-score for '{event_name}': {z_score:.2f} (Surprise: {current_surprise}, Hist. StdDev: {std_dev:.4f})")
        return z_score

    finally:
        db.close()

if __name__ == '__main__':
    # This is a test block to demonstrate the function's usage.
    # It requires a populated database to work correctly.
    print("--- Testing Z-Score Calculator ---")

    # Example: Let's assume a test event is in the DB.
    # In a real scenario, you'd get these values from the WebSocket feed.

    # This test will likely return None unless you have seeded your database
    # with historical "US ISM Manufacturing PMI" data.
    test_event_name = "US ISM Manufacturing PMI"
    test_actual = 52.5
    test_forecast = 50.2

    print(f"\nTesting with hypothetical event: {test_event_name}")
    print(f"Actual: {test_actual}, Forecast: {test_forecast}")

    z_score_result = calculate_z_score(test_event_name, test_actual, test_forecast)

    if z_score_result is not None:
        print(f"\nFinal Z-Score: {z_score_result:.4f}")
        if abs(z_score_result) > 2.0:
            print("Z-score exceeds threshold of |2.0|. This would trigger a trade.")
        else:
            print("Z-score is within threshold of |2.0|. No trade would be triggered.")
    else:
        print("\nCould not calculate Z-score due to reasons mentioned above.")
