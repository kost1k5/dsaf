import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collector.collector import DataCollector
from src.strategies.sma_crossover import SmaCrossoverStrategy
from src.database.session import SessionLocal
from src.database.models import Candle, BacktestResult

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


def save_results_to_db(results: dict):
    """Saves the backtest results to the database."""
    db = SessionLocal()
    try:
        # Serialize datetime objects in trades list before saving
        if 'trades' in results and results['trades']:
            for trade in results['trades']:
                if 'entry_time' in trade:
                    trade['entry_time'] = trade['entry_time'].isoformat()
                if 'exit_time' in trade:
                    trade['exit_time'] = trade['exit_time'].isoformat()

        backtest_entry = BacktestResult(**results)
        db.add(backtest_entry)
        db.commit()
        print("\nBacktest results saved to database.")
    except Exception as e:
        print(f"\nCould not save results to DB: {e}")
        db.rollback()
    finally:
        db.close()


def run_backtest(strategy, data: pd.DataFrame, initial_cash=10000.0, commission_pct=0.001, slippage_pct=0.0005):
    """
    Runs a detailed backtest, including commission and slippage, and calculates performance metrics.
    """
    if data.empty:
        print("Data is empty, cannot run backtest.")
        return

    print(f"\nRunning backtest... (Commission: {commission_pct*100}%, Slippage: {slippage_pct*100}%)")
    df = strategy.generate_signals(data)

    cash = initial_cash
    position = 0.0
    portfolio_values = [initial_cash]
    trades = []
    entry_price = 0

    for i, row in df.iterrows():
        close_price = df.loc[i, 'close']
        signal = df.loc[i, 'signal']
        trade_time = df.loc[i, 'open_time']

        # Manage trades, including costs
        if signal == 'BUY' and cash > 0:
            buy_price = close_price * (1 + slippage_pct)
            position_before_commission = cash / buy_price
            commission = position_before_commission * commission_pct
            position = position_before_commission - commission

            entry_price = buy_price
            entry_time = trade_time
            cash = 0
            print(f"{trade_time.date()} | BUY at ~{buy_price:.2f} | Portfolio: ${initial_cash:.2f}")

        elif signal == 'SELL' and position > 0:
            sell_price = close_price * (1 - slippage_pct)
            cash_before_commission = position * sell_price
            commission = cash_before_commission * commission_pct
            cash = cash_before_commission - commission

            trades.append({
                'entry_price': entry_price,
                'exit_price': sell_price,
                'entry_time': entry_time,
                'exit_time': trade_time
            })
            position = 0
            entry_price = 0
            print(f"{trade_time.date()} | SELL at ~{sell_price:.2f} | Portfolio: ${cash:.2f}")

        # Record portfolio value at each step
        current_value = cash if cash > 0 else position * close_price
        portfolio_values.append(current_value)

    # --- Metrics Calculation ---
    final_value = portfolio_values[-1]
    total_pnl = final_value - initial_cash
    total_return_percent = (total_pnl / initial_cash) * 100

    # Win Rate
    wins = sum(1 for trade in trades if trade['exit_price'] > trade['entry_price'])
    win_rate = (wins / len(trades)) * 100 if trades else 0

    # Max Drawdown
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0

    # Sharpe Ratio (assuming daily data and 0 risk-free rate)
    returns = portfolio_series.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365) if len(returns) > 1 else 0

    # --- Prepare results for saving and printing ---
    results = {
        "strategy_name": strategy.__class__.__name__,
        "symbol": data.attrs.get('symbol', 'N/A'),
        "timeframe": data.attrs.get('timeframe', 'N/A'),
        "initial_balance": float(initial_cash),
        "final_balance": float(final_value),
        "pnl_usd": float(total_pnl),
        "pnl_percent": float(total_return_percent),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),
        "total_trades": len(trades),
        "start_date": df.iloc[0]['open_time'].date(),
        "end_date": df.iloc[-1]['open_time'].date(),
        "trades": trades,
    }

    print("\n--- Backtest Report ---")
    print(f"Period: {results['start_date']} to {results['end_date']}")
    print(f"Strategy: {results['strategy_name']}")
    print(f"Symbol: {results['symbol']}, Timeframe: {results['timeframe']}")
    print(f"Initial Portfolio Value: ${results['initial_balance']:,.2f}")
    print(f"Final Portfolio Value:   ${results['final_balance']:,.2f}")
    print(f"Total Profit/Loss:       ${results['pnl_usd']:,.2f} ({results['pnl_percent']:.2f}%)")
    print(f"Total Trades:            {results['total_trades']}")
    print(f"Win Rate:                {results['win_rate']:.2f}%")
    print(f"Max Drawdown:            {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio (ann.):     {results['sharpe_ratio']:.2f}")
    print("-----------------------")

    return results


if __name__ == '__main__':
    SYMBOL = 'BTC/USDT'
    TIMEFRAME = '1d' # Daily timeframe for a longer-term backtest

    # 1. Try to load data from DB
    db = SessionLocal()
    candles_df = load_data_from_db(db, SYMBOL, TIMEFRAME)
    db.close()

    # 2. If DB is empty, fetch from the exchange
    if candles_df.empty:
        print(f"Fetching new data for {SYMBOL} since database is empty...")
        collector = DataCollector(exchange_id='okx')
        # Fetch last 365 days of data
        since = collector.exchange.parse8601('2023-01-01T00:00:00Z')
        candles_list = collector.fetch_candles(SYMBOL, timeframe=TIMEFRAME, since=since, limit=365)

        if candles_list:
            candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
            # Convert timestamp to datetime
            candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms', utc=True)
            # Store metadata in the DataFrame
            candles_df.attrs = {'symbol': SYMBOL, 'timeframe': TIMEFRAME}
        else:
            print("Failed to fetch new data. Exiting.")
            candles_df = pd.DataFrame()


    # 3. Initialize and run the backtest
    if not candles_df.empty:
        from src.core.config import settings
        sma_strategy = SmaCrossoverStrategy(short_window=40, long_window=100)
        results = run_backtest(
            strategy=sma_strategy,
            data=candles_df,
            commission_pct=settings.BACKTEST_COMMISSION_PCT,
            slippage_pct=settings.BACKTEST_SLIPPAGE_PCT
        )
        if results:
            save_results_to_db(results)
    else:
        print("No data available to run the backtest.")
