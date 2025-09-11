import requests
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session

from src.database.session import SessionLocal
from src.database.models import EconomicCalendarEvent

def fetch_and_store_economic_calendar(countries: list = None):
    """
    Fetches economic calendar data from the OKX v5 API and stores it.
    """
    print("--- Fetching Economic Calendar Data from OKX API ---")

    if countries is None:
        countries = ['US', 'CN', 'EU', 'GB', 'JP'] # Default countries

    country_str = ",".join(countries)
    url = f"https://www.okx.com/api/v5/public/economic-calendar?country={country_str}"

    all_events = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get('code') == '0':
            events = data.get('data', [])
            if events:
                all_events.extend(events[0]) # The data is nested in a list
            print(f"Successfully fetched {len(all_events)} events from OKX.")
        else:
            print(f"Error from OKX API: {data.get('msg')}")
            return

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data from OKX: {e}")
        return

    if not all_events:
        print("No economic events found for the specified criteria.")
        return

    # --- Data Processing and Storage ---
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
                # Convert timestamp from milliseconds to a datetime object
                ts = event.get('dateTimestamp')
                if not ts:
                    continue

                # Importance mapping: OKX uses 1,2,3 for low,med,high
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
    fetch_and_store_economic_calendar()
