import pandas as pd
import feedparser
import requests
from datetime import datetime, timezone
import time
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import func

from src.database.models import FearGreedIndex, NewsHeadline
from src.database.session import SessionLocal

class SentimentCollector:
    """
    A class to collect sentiment data from various sources.
    """
    def __init__(self, db_session: Session):
        self.db = db_session
        self.rss_feeds = {
            'Cointelegraph': 'https://cointelegraph.com/rss',
            'CoinDesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/'
        }

    def get_latest_fng_timestamp(self) -> datetime:
        """
        Gets the timestamp of the most recent Fear & Greed index entry in the database.
        """
        latest_fng = self.db.query(func.max(FearGreedIndex.timestamp)).scalar()
        return latest_fng

    def save_fng_data_to_db(self, fng_df: pd.DataFrame) -> int:
        """
        Saves Fear & Greed data to the database, ignoring duplicates.
        Returns the number of new rows inserted.
        """
        if fng_df.empty:
            return 0

        fng_records = []
        for timestamp, row in fng_df.iterrows():
            fng_records.append({
                "timestamp": timestamp.to_pydatetime(),
                "value": int(row['fng_value']),
                "classification": row['fng_classification']
            })

        if not fng_records:
            return 0

        stmt = insert(FearGreedIndex).values(fng_records)
        stmt = stmt.on_conflict_do_nothing(index_elements=['timestamp'])
        result = self.db.execute(stmt)
        self.db.commit()
        return result.rowcount


    def fetch_fear_greed_data(self, limit: int = 0) -> pd.DataFrame:
        """
        Fetches historical Fear & Greed Index data from alternative.me API.
        (A) Includes retry logic for network robustness.

        :param limit: The number of days to fetch data for. 0 means all available data.
        :return: A pandas DataFrame with historical F&G data.
        """
        print(f"Fetching Fear & Greed Index data (limit={limit})...")
        retries = 3
        for i in range(retries):
            try:
                url = f"https://api.alternative.me/fng/?limit={limit}"
                response = requests.get(url, timeout=15)  # Increased timeout
                response.raise_for_status()  # Raise an exception for bad status codes

                json_data = response.json()
                df = pd.DataFrame(json_data['data'])

                # Convert 'timestamp' to datetime and set as index
                df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')
                df = df.set_index('timestamp').sort_index()

                # Convert 'value' to numeric
                df['value'] = pd.to_numeric(df['value'])

                # Rename columns for clarity
                df = df.rename(columns={'value': 'fng_value', 'value_classification': 'fng_classification'})

                # Drop the 'time_until_update' column as it's not needed for historical analysis
                if 'time_until_update' in df.columns:
                    df = df.drop(columns=['time_until_update'])

                print(f"Successfully fetched {len(df)} F&G data points.")
                return df

            except requests.exceptions.RequestException as e:
                print(f"Attempt {i + 1}/{retries} failed: An error occurred during API request to alternative.me: {e}")
                if i < retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("All retries failed for Fear & Greed Index.")
                    return pd.DataFrame()
            except Exception as e:
                print(f"A non-recoverable error occurred while processing Fear & Greed data: {e}")
                # This is for JSON errors, etc. which are not worth retrying.
                return pd.DataFrame()

        return pd.DataFrame() # Should only be reached if the loop finishes, which it shouldn't

    def get_latest_news_timestamp(self) -> datetime:
        """
        Gets the timestamp of the most recent news headline in the database.
        """
        latest_news = self.db.query(func.max(NewsHeadline.published_at)).scalar()
        return latest_news

    def save_news_to_db(self, news_items: list) -> int:
        """
        Saves news items to the database, ignoring duplicates based on the link.
        Returns the number of new rows inserted.
        """
        if not news_items:
            return 0

        records = []
        for item in news_items:
            # Ensure there is a datetime object to save
            if item.get('published') and isinstance(item['published'], datetime):
                records.append({
                    "source": item['source'],
                    "title": item['title'],
                    "link": item['link'],
                    "published_at": item['published']
                })

        if not records:
            return 0

        stmt = insert(NewsHeadline).values(records)
        stmt = stmt.on_conflict_do_nothing(index_elements=['link'])
        result = self.db.execute(stmt)
        self.db.commit()
        return result.rowcount

    def fetch_rss_news(self) -> list:
        """
        Fetches news headlines from a list of RSS feeds.

        :return: A list of dictionaries, where each dict is a news item.
        """
        print(f"Fetching news from RSS feeds: {list(self.rss_feeds.keys())}")
        all_news = []
        for source, url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    # Convert 'published_parsed' time.struct_time to a datetime object
                    published_dt = None
                    if 'published_parsed' in entry and entry.published_parsed:
                        published_dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))

                    all_news.append({
                        'source': source,
                        'title': entry.title,
                        'link': entry.link,
                        'published': published_dt
                    })
            except Exception as e:
                print(f"Could not fetch or parse RSS feed from {source} ({url}): {e}")

        print(f"Successfully fetched {len(all_news)} news items.")
        return all_news

if __name__ == '__main__':
    db_session = SessionLocal()
    collector = SentimentCollector(db_session)

    print("\n--- Testing Fear & Greed Index Collector ---")
    # Fetch all historical data
    fng_df = collector.fetch_fear_greed_data(limit=0)
    if not fng_df.empty:
        print(fng_df.head())
        print(fng_df.tail())
        fng_df.info()

    print("\n--- Testing RSS News Collector ---")
    news_items = collector.fetch_rss_news()
    if news_items:
        print(f"Fetched {len(news_items)} total items. Showing first 5:")
        for item in news_items[:5]:
            print(f"  - [{item['source']} @ {item['published']}] {item['title']}")
