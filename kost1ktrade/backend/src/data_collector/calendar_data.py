import os
import pandas as pd
from datetime import datetime, timedelta
from apify_client import ApifyClient

from src.core.config import settings
from src.database.session import SessionLocal
from src.database.models import EconomicCalendarEvent
from sqlalchemy.orm import Session

def fetch_and_store_economic_calendar(
    start_date: str,
    end_date: str,
    countries: list = None,
    importances: list = None
):
    """
    Fetches economic calendar data from Apify and stores it in the database.

    :param start_date: The start date in 'dd/mm/yyyy' format.
    :param end_date: The end date in 'dd/mm/yyyy' format.
    :param countries: A list of countries to fetch data for.
    :param importances: A list of importance levels ('high', 'medium', 'low').
    """
    api_token = settings.APIFY_API_TOKEN
    if not api_token:
        print("Warning: APIFY_API_TOKEN is not set in the environment. Skipping economic calendar data collection.")
        return

    print("--- Fetching Economic Calendar Data from Apify ---")

    if countries is None:
        countries = ["united states", "china", "germany", "united kingdom", "japan"]
    if importances is None:
        importances = ["high", "medium"]

    client = ApifyClient(api_token)
    all_events = []

    for country in countries:
        # Loop through each importance level and make a separate API call
        for importance in importances:
            print(f"  - Fetching calendar data for {country} (importance: {importance})...")
            run_input = {
                "country": country,
                "importances": importance, # Pass one importance level at a time
                "fromDate": start_date,
                "toDate": end_date,
            }

            try:
                run = client.actor("pintostudio/economic-calendar-data-investing-com").call(run_input=run_input)
                for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                    all_events.append(item)
            except Exception as e:
                print(f"  - An error occurred while fetching calendar data for {country} (importance: {importance}): {e}")
                continue

    if not all_events:
        print("No economic events found for the specified criteria across all countries.")
        return

    df = pd.DataFrame(all_events)
    print(f"Successfully fetched a total of {len(df)} economic events from Apify.")

    # --- Data Processing and Storage ---
    try:
        df['event_datetime_utc'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%m/%Y %H:%M', utc=True)

        db: Session = SessionLocal()
        try:
            print("Storing economic events in the database...")
            for _, row in df.iterrows():
                # Check if event already exists to prevent duplicates
                exists = db.query(EconomicCalendarEvent).filter_by(event_id=row['id']).first()
                if not exists:
                    db_event = EconomicCalendarEvent(
                        event_id=row['id'],
                        event_datetime=row['event_datetime_utc'],
                        country=row['zone'],
                        importance=row['importance'],
                        event_name=row['event'],
                        actual=row.get('actual'),
                        forecast=row.get('forecast'),
                        previous=row.get('previous')
                    )
                    db.add(db_event)
            db.commit()
            print("Finished storing economic events.")
        finally:
            db.close()
    except Exception as e:
        print(f"An error occurred while processing or storing economic calendar data: {e}")

if __name__ == '__main__':
    # Example usage
    today = datetime.now()
    start = (today - timedelta(days=365)).strftime('%d/%m/%Y')
    end = (today + timedelta(days=30)).strftime('%d/%m/%Y')

    fetch_and_store_economic_calendar(start_date=start, end_date=end)
