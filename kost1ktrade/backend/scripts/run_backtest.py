import sys
import os
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collector.collector import DataCollector
from src.strategies.sma_crossover import SmaCrossoverStrategy
from src.database.session import SessionLocal
from src.database.models import Candle

def load_data_from_db(db_session, symbol, interval) -> pd.DataFrame:
    """Loads candle data from the database for a given symbol and interval."""
    print(f"Loading data for {symbol} ({interval}) from database...")
    try:
        query = db_session.query(Candle).filter(Candle.symbol == symbol, Candle.interval == interval).order_by(Candle.open_time)
        df = pd.read_sql(query.statement, db_session.bind)
        if df.empty:
            print("No data found in the database.")
        else:
            print(f"Loaded {len(df)} candles from database.")
        return df
    except Exception as e:
        print(f"Could not load from DB: {e}")
        return pd.DataFrame()


def run_backtest(strategy, data: pd.DataFrame, initial_cash=10000.0):
    """Runs a simple backtest on the given data and strategy."""
    if data.empty:
        print("Data is empty, cannot run backtest.")
        return

    print("\nRunning backtest...")
    df = strategy.generate_signals(data)

    cash = initial_cash
    position = 0.0 # Holds the amount of the asset we own

    for i, row in df.iterrows():
        # Using .loc for safer access
        close_price = df.loc[i, 'close']
        signal = df.loc[i, 'signal']

        if signal == 'BUY' and cash > 0:
            position = cash / close_price
            print(f"{df.loc[i, 'open_time'].date()} | BUY at {close_price:.2f} | Portfolio: ${cash:.2f}")
            cash = 0
        elif signal == 'SELL' and position > 0:
            cash = position * close_price
            print(f"{df.loc[i, 'open_time'].date()} | SELL at {close_price:.2f} | Portfolio: ${cash:.2f}")
            position = 0

    # Calculate final portfolio value
    final_value = cash if cash > 0 else position * df.iloc[-1]['close']

    print("\n--- Backtest Report ---")
    print(f"Initial Portfolio Value: ${initial_cash:.2f}")
    print(f"Final Portfolio Value:   ${final_value:.2f}")
    pnl = final_value - initial_cash
    pnl_percent = (pnl / initial_cash) * 100
    print(f"Profit/Loss:             ${pnl:.2f} ({pnl_percent:.2f}%)")
    print("-----------------------")


if __name__ == '__main__':
    SYMBOL = 'BTC/USD'
    TIMEFRAME = '1d' # Daily timeframe for a longer-term backtest

    # 1. Try to load data from DB
    db = SessionLocal()
    candles_df = load_data_from_db(db, SYMBOL, TIMEFRAME)
    db.close()

    # 2. If DB is empty, fetch from the exchange
    if candles_df.empty:
        print(f"Fetching new data for {SYMBOL} since database is empty...")
        collector = DataCollector(exchange_id='kraken')
        # Fetch last 365 days of data
        since = collector.exchange.parse8601('2023-01-01T00:00:00Z')
        candles_list = collector.fetch_candles(SYMBOL, timeframe=TIMEFRAME, since=since, limit=365)

        if candles_list:
            candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
            # Convert timestamp to datetime
            candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms', utc=True)
        else:
            print("Failed to fetch new data. Exiting.")
            candles_df = pd.DataFrame()


    # 3. Initialize and run the backtest
    if not candles_df.empty:
        sma_strategy = SmaCrossoverStrategy(short_window=40, long_window=100)
        run_backtest(sma_strategy, candles_df)
    else:
        print("No data available to run the backtest.")
