import pandas as pd
import feedparser
import requests
from datetime import datetime
import time

class SentimentCollector:
    """
    A class to collect sentiment data from various sources.
    """
    def __init__(self):
        self.rss_feeds = {
            'Cointelegraph': 'https://cointelegraph.com/rss',
            'CoinDesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/'
        }

    def fetch_fear_greed_data(self, limit: int = 0) -> pd.DataFrame:
        """
        Fetches historical Fear & Greed Index data from alternative.me API.

        :param limit: The number of days to fetch data for. 0 means all available data.
        :return: A pandas DataFrame with historical F&G data.
        """
        print(f"Fetching Fear & Greed Index data (limit={limit})...")
        try:
            url = f"https://api.alternative.me/fng/?limit={limit}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes

            json_data = response.json()
            df = pd.DataFrame(json_data['data'])

            # Convert 'timestamp' to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
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
            print(f"An error occurred during API request to alternative.me: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An error occurred while processing Fear & Greed data: {e}")
            return pd.DataFrame()

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
    collector = SentimentCollector()

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
