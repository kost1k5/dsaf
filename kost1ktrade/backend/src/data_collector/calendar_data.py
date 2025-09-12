import requests
import time
import hmac
import base64
import json
from datetime import datetime
from sqlalchemy.orm import Session

from src.database.session import SessionLocal
from src.database.models import EconomicCalendarEvent

def get_auth_headers(api_key: str, secret_key: str, passphrase: str, method: str, request_path: str) -> dict:
    """Generates the required authentication headers for an OKX API request."""
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    string_to_sign = timestamp + method.upper() + request_path

    mac = hmac.new(bytes(secret_key, 'utf-8'), bytes(string_to_sign, 'utf-8'), digestmod='sha256')
    signature = base64.b64encode(mac.digest()).decode('utf-8')

    headers = {
        'OK-ACCESS-KEY': api_key,
        'OK-ACCESS-SIGN': signature,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': passphrase,
        'Content-Type': 'application/json'
    }
    return headers

def fetch_and_store_economic_calendar(api_key: str, secret_key: str, passphrase: str, countries: list = None):
    """
    Fetches economic calendar data from the OKX v5 API using manual request signing
    and stores it in the database.
    """
    print("--- Fetching Economic Calendar Data from OKX API (with manual auth) ---")

    if countries is None:
        countries = ['US', 'CN', 'EU', 'GB', 'JP']

    base_url = "https://www.okx.com"
    request_path = f"/api/v5/public/economic-calendar?country={','.join(countries)}"

    try:
        headers = get_auth_headers(api_key, secret_key, passphrase, 'GET', request_path)
        response = requests.get(base_url + request_path, headers=headers)
        response.raise_for_status()
        data = response.json()

        all_events = []
        if data.get("code") == "0":
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
    except ValueError:
        print(f"Failed to decode JSON from response. Response text: {response.text}")
        return

    if not all_events:
        print("No economic events found.")
        return

    db: Session = SessionLocal()
    try:
        # (The database storage logic remains the same as before)
        new_events_count = 0
        for event in all_events:
            event_id = event.get('id')
            if not event_id: continue
            if not db.query(EconomicCalendarEvent).filter_by(event_id=event_id).first():
                ts = event.get('dateTimestamp')
                if not ts: continue
                importance_map = {'1': 'low', '2': 'medium', '3': 'high'}
                db.add(EconomicCalendarEvent(
                    event_id=event_id,
                    event_datetime=datetime.utcfromtimestamp(int(ts) / 1000),
                    country=event.get('country'),
                    importance=importance_map.get(str(event.get('importance')), 'low'),
                    event_name=event.get('event'),
                    actual=event.get('actual'),
                    forecast=event.get('forecast'),
                    previous=event.get('previous')
                ))
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
    # This test requires API keys to be set as environment variables
    import os
    api_key = os.getenv("OKX_APIKEY")
    secret_key = os.getenv("OKX_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")
    if not all([api_key, secret_key, passphrase]):
        print("Please set OKX_APIKEY, OKX_SECRET, and OKX_PASSPHRASE environment variables to run the test.")
    else:
        fetch_and_store_economic_calendar(api_key, secret_key, passphrase)
