import requests
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session

from src.database.session import SessionLocal
from src.database.models import EconomicCalendarEvent
from src.data_collector.collector import DataCollector # Import the main collector

def fetch_and_store_economic_calendar(data_collector: DataCollector, countries: list = None):
    """
    Fetches economic calendar data from the OKX v5 API using the provided
    authenticated DataCollector instance and stores it.
    """
    print("--- Fetching Economic Calendar Data from OKX API ---")

    if countries is None:
        countries = ['US', 'CN', 'EU', 'GB', 'JP'] # Default countries

    country_str = ",".join(countries)

    # Use the generic 'request' method for endpoints not explicitly defined in ccxt
    path = 'public/economic-calendar'
    params = {'country': country_str}

    all_events = []
    try:
        # This will automatically handle signing and headers via the collector's ccxt instance
        data = data_collector.exchange.request(path, 'public', 'GET', params)

        # The OKX API nests the actual data inside a 'data' key and another list
        events = data.get('data', [])
        if events:
            all_events.extend(events[0])
        print(f"Successfully fetched {len(all_events)} events from OKX.")

    except Exception as e:
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
    # This script is not meant to be run standalone anymore,
    # as it requires an initialized DataCollector.
    print("This script should be called from the main data collection pipeline.")
