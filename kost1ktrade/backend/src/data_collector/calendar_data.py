import requests
import time
from datetime import datetime
from sqlalchemy.orm import Session

from src.database.session import SessionLocal
from src.database.models import EconomicCalendarEvent
# The DataCollector import is no longer needed for this function
# from src.data_collector.collector import DataCollector

def fetch_and_store_economic_calendar(countries: list = None):
    """
    Fetches economic calendar data from the OKX v5 API using the requests
    library and stores it in the database.
    """
    print("--- Fetching Economic Calendar Data from OKX API ---")

    if countries is None:
        # Default countries can be modified here if needed
        countries = ['US', 'CN', 'EU', 'GB', 'JP']

    base_url = "https://www.okx.com"
    endpoint = "/api/v5/public/economic-calendar"

    # OKX API uses 'country' for a comma-separated list, not 'region'
    params = {
        "country": ",".join(countries),
    }

    all_events = []
    try:
        response = requests.get(base_url + endpoint, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()

        # According to OKX docs, code "0" means success
        if data.get("code") == "0":
            # The data is nested in a list within the 'data' key
            event_list = data.get("data", [])
            if event_list:
                all_events = event_list[0]
                print(f"Successfully fetched {len(all_events)} events from OKX.")
        else:
            print(f"API returned an error: {data.get('msg')}")
            return

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the network request: {e}")
        return
    except ValueError: # Catches JSON decoding errors
        print(f"Failed to decode JSON from response. Response text: {response.text}")
        return

    if not all_events:
        print("No economic events found for the specified criteria.")
        return

    # --- Data Processing and Storage (largely unchanged) ---
    db: Session = SessionLocal()
    try:
        print("Storing economic events in the database...")
        new_events_count = 0
        for event in all_events:
            event_id = event.get('id')
            if not event_id:
                continue

            exists = db.query(EconomicCalendarEvent).filter_by(event_id=event_id).first()
            if not exists:
                ts = event.get('dateTimestamp')
                if not ts:
                    continue

                importance_map = {'1': 'low', '2': 'medium', '3': 'high'}
                importance_str = importance_map.get(str(event.get('importance')), 'low')

                db_event = EconomicCalendarEvent(
                    event_id=event_id,
                    event_datetime=datetime.utcfromtimestamp(int(ts) / 1000),
                    country=event.get('country'),
                    importance=importance_str,
                    event_name=event.get('event'),
                    actual=event.get('actual'),
                    forecast=event.get('forecast'),
                    previous=event.get('previous')
                )
                db.add(db_event)
                new_events_count += 1

        if new_events_count > 0:
            db.commit()
            print(f"Finished storing {new_events_count} new economic events.")
        else:
            print("No new events to store.")

    except Exception as e:
        print(f"An error occurred during DB operations: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == '__main__':
    # This function can now be run standalone for testing
    print("Running standalone test for fetch_and_store_economic_calendar...")
    fetch_and_store_economic_calendar()
