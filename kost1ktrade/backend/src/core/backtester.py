import pandas as pd
import numpy as np
import talib
from src.core.risk_manager import calculate_position_size

class Backtester:
    def __init__(self, strategy, candles_df, initial_balance=10000, commission_pct=0.001, risk_per_trade_pct=1.0, atr_multiplier=2.0):
        self.strategy = strategy
        self.candles_df = candles_df.copy() # Use a copy to avoid modifying original df
        self.initial_balance = initial_balance
        self.commission_pct = commission_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.atr_multiplier = atr_multiplier
        self.results = None

    def run(self):
        """
        Runs the backtest simulation with integrated risk management.
        """
        # 1. Generate strategy signals
        df = self.strategy.generate_signals(self.candles_df)

        # 2. Add ATR for risk management
        atr_period = 14
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
        df.dropna(inplace=True) # Drop rows where indicators are not yet calculated

        if 'signal' not in df.columns:
            raise ValueError("The 'signal' column is missing from the strategy output.")
        df['signal'] = df['signal'].fillna('HOLD')

        # 3. Run simulation loop
        balance = self.initial_balance
        position = 0.0  # Represents the amount of the base asset held
        equity_curve = [self.initial_balance]
        trades = []

        for i, row in df.iterrows():
            signal = row['signal']
            close_price = row['close']
            atr_value = row['atr']
            current_equity = balance + (position * close_price)

            # --- Trade Execution Logic ---
            if signal == 'BUY' and position == 0: # Can only buy if we have no open position

                # Use risk manager to calculate position size
                sized_position = calculate_position_size(
                    capital=current_equity,
                    risk_per_trade_pct=self.risk_per_trade_pct,
                    atr_value=atr_value,
                    atr_multiplier=self.atr_multiplier,
                    price=close_price
                )

                investment_cost = sized_position * close_price
                commission = investment_cost * self.commission_pct

                if balance >= (investment_cost + commission):
                    balance -= (investment_cost + commission)
                    position = sized_position
                    trades.append({'type': 'BUY', 'price': close_price, 'row': i})

            elif signal == 'SELL' and position > 0: # If we have base currency to sell
                sale_value = position * close_price
                commission = sale_value * self.commission_pct
                balance += (sale_value - commission)
                position = 0.0
                trades.append({'type': 'SELL', 'price': close_price, 'row': i})

            # Update equity for this time step
            equity_curve.append(balance + (position * close_price))

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
