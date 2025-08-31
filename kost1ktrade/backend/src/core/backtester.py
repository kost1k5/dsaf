import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, strategy, candles_df, initial_balance=10000, commission_pct=0.001):
        self.strategy = strategy
        self.candles_df = candles_df
        self.initial_balance = initial_balance
        self.commission_pct = commission_pct
        self.results = None

    def run(self):
        """
        Runs the backtest simulation.
        """
        df = self.strategy.generate_signals(self.candles_df)

        # Ensure signal column exists and fill NaNs
        if 'signal' not in df.columns:
            raise ValueError("The 'signal' column is missing from the strategy output.")
        df['signal'] = df['signal'].fillna('HOLD')

        balance = self.initial_balance
        position = 0.0  # Represents the amount of the base asset held
        equity_curve = [self.initial_balance]
        trades = []

        for i, row in df.iterrows():
            signal = row['signal']
            close_price = row['close']

            # --- Trade Execution Logic ---
            if signal == 'BUY' and balance > 10: # If we have quote currency to buy
                investment = balance * (1 - self.commission_pct)
                position = investment / close_price
                balance = 0.0
                trades.append({'type': 'BUY', 'price': close_price, 'row': i})

            elif signal == 'SELL' and position > 0: # If we have base currency to sell
                sale_value = position * close_price
                balance = sale_value * (1 - self.commission_pct)
                position = 0.0
                trades.append({'type': 'SELL', 'price': close_price, 'row': i})

            # Update equity for this time step
            current_equity = balance + (position * close_price)
            equity_curve.append(current_equity)

        self.results = self._calculate_metrics(equity_curve, trades)
        return self.results

    def _calculate_metrics(self, equity_curve, trades):
        """
        Calculates performance metrics from the simulation results.
        """
        if not equity_curve:
            return {}

        total_pnl_pct = ((equity_curve[-1] / self.initial_balance) - 1) * 100

        # Max Drawdown
        equity_series = pd.Series(equity_curve)
        cumulative_max = equity_series.cummax()
        drawdown = (equity_series - cumulative_max) / cumulative_max
        max_drawdown_pct = abs(drawdown.min()) * 100 if not drawdown.empty else 0

        # Profit Factor & Win Rate
        if not trades:
            return {
                "total_pnl_pct": total_pnl_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "profit_factor": 0,
                "win_rate_pct": 0,
                "total_trades": 0,
                "equity_curve": equity_curve
            }

        gross_profit = 0
        gross_loss = 0
        wins = 0

        for i in range(len(trades)):
            if trades[i]['type'] == 'BUY':
                # Find the subsequent SELL trade
                for j in range(i + 1, len(trades)):
                    if trades[j]['type'] == 'SELL':
                        profit = (trades[j]['price'] - trades[i]['price']) / trades[i]['price']
                        if profit > 0:
                            gross_profit += profit
                            wins += 1
                        else:
                            gross_loss += abs(profit)
                        break # Move to the next BUY trade

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else "inf"
        total_buy_trades = len([t for t in trades if t['type'] == 'BUY'])
        win_rate_pct = (wins / total_buy_trades) * 100 if total_buy_trades > 0 else 0

        return {
            "total_pnl_pct": round(total_pnl_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "profit_factor": round(profit_factor, 2) if isinstance(profit_factor, float) else profit_factor,
            "win_rate_pct": round(win_rate_pct, 2),
            "total_trades": total_buy_trades,
            "equity_curve": equity_curve
        }
