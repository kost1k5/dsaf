import asyncio
import threading
import pandas as pd
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, JobQueue

from src.core.config import settings
from src.core.bot_state import bot_state
from scripts.run_backtest import run_backtest
from src.optimization.optimizer import Optimizer
from src.strategies.sma_crossover import SmaCrossoverStrategy # Example strategy
from src.data_collector.collector import DataCollector

# --- Standalone Notifier Function ---

async def send_telegram_notification(message: str):
    """A simple, standalone function to send a notification."""
    if not settings.TELEGRAM_TOKEN or not settings.TELEGRAM_CHAT_ID:
        # Silently fail if not configured, as this is a non-critical feature.
        print(f"Telegram not configured. Message not sent: {message[:50]}...")
        return

    try:
        bot = telegram.Bot(token=settings.TELEGRAM_TOKEN)
        await bot.send_message(
            chat_id=settings.TELEGRAM_CHAT_ID,
            text=message,
            parse_mode='Markdown'
        )
        print("Sent Telegram notification.")
    except Exception as e:
        print(f"Failed to send Telegram notification: {e}")


# --- Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to Kost1kTrade Bot! Use /status, /backtest, or /optimize.")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_text = (
        f"Signal Bot: *{bot_state.signal_bot_mode}*\n"
        f"Grid Bot: *{bot_state.grid_bot_mode}*"
    )
    await update.message.reply_text(status_text, parse_mode='Markdown')

def _run_backtest_and_report(context: ContextTypes.DEFAULT_TYPE, strategy_class, symbol, timeframe):
    """Helper function to run backtest and send report."""
    try:
        # 1. Get Data
        collector = DataCollector()
        candles_list = collector.fetch_candles(symbol, timeframe, limit=365)
        if not candles_list:
            asyncio.run(context.bot.send_message(context.job.chat_id, "Failed to fetch data for backtest."))
            return

        candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
        candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms', utc=True)
        candles_df.attrs = {'symbol': symbol, 'timeframe': timeframe}

        # 2. Run Backtest
        strategy = strategy_class()
        results = run_backtest(
            strategy=strategy,
            data=candles_df,
            commission_pct=settings.BACKTEST_COMMISSION_PCT,
            slippage_pct=settings.BACKTEST_SLIPPAGE_PCT
        )

        # 3. Format and Send Report
        report_lines = [
            f"*Backtest Report for {strategy_class.__name__}*",
            f"`{symbol} ({timeframe})`",
            "",
            f"Sharpe Ratio: *{results['sharpe_ratio']:.2f}*",
            f"P/L: *{results['pnl_percent']:.2f}%*",
            f"Max Drawdown: *{results['max_drawdown']:.2f}%*",
            f"Win Rate: *{results['win_rate']:.2f}%*",
            f"Total Trades: *{results['total_trades']}*",
        ]

        # Add last 5 trades to the report
        if results['trades']:
            report_lines.append("\n*Recent Trades:*")
            for trade in results['trades'][-5:]:
                pnl = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100
                emoji = "✅" if pnl > 0 else "❌"
                report_lines.append(
                    f"{emoji} `Buy: {trade['entry_price']:.2f}, Sell: {trade['exit_price']:.2f}, PnL: {pnl:.2f}%`"
                )

        report = "\n".join(report_lines)
        asyncio.run(context.bot.send_message(context.job.chat_id, report, parse_mode='Markdown'))

    except Exception as e:
        asyncio.run(context.bot.send_message(context.job.chat_id, f"An error occurred during backtest: {e}"))


async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Starting backtest for SmaCrossoverStrategy... This may take a moment.")
    context.job_queue.run_once(
        lambda ctx: _run_backtest_and_report(ctx, SmaCrossoverStrategy, 'BTC/USDT', '1d'),
        0
    )

def _run_optimization_and_report(context: ContextTypes.DEFAULT_TYPE, strategy_class, symbol, timeframe):
    """Helper function to run optimization and send report."""
    try:
        # 1. Get Data
        collector = DataCollector()
        candles_list = collector.fetch_candles(symbol, timeframe, limit=365)
        if not candles_list:
            asyncio.run(context.bot.send_message(context.job.chat_id, "Failed to fetch data for optimization."))
            return

        candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
        candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms', utc=True)
        candles_df.attrs = {'symbol': symbol, 'timeframe': timeframe}

        # 2. Setup and run optimizer
        optimizer = Optimizer(strategy_class=strategy_class, data=candles_df)
        optimizer.set_params(short_window=range(20, 51, 10), long_window=range(60, 101, 20))
        best_params, best_metrics = optimizer.run(optimize_for="sharpe_ratio")

        # 3. Format and Send Report
        if best_params:
            report = (
                f"*Optimization Report for {strategy_class.__name__}*\n"
                f"`{symbol} ({timeframe})`\n\n"
                f"*Best Parameters Found:*\n`{best_params}`\n\n"
                f"*Resulting Metrics:*\n"
                f"Sharpe Ratio: *{best_metrics['sharpe_ratio']:.2f}*\n"
                f"P/L: *{best_metrics['pnl_percent']:.2f}%*\n"
                f"Max Drawdown: *{best_metrics['max_drawdown']:.2f}%*"
            )
        else:
            report = "Optimization finished with no valid results."

        asyncio.run(context.bot.send_message(context.job.chat_id, report, parse_mode='Markdown'))

    except Exception as e:
        asyncio.run(context.bot.send_message(context.job.chat_id, f"An error occurred during optimization: {e}"))


async def optimize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Starting optimization for SmaCrossoverStrategy... This will take several minutes.")
    context.job_queue.run_once(
        lambda ctx: _run_optimization_and_report(ctx, SmaCrossoverStrategy, 'BTC/USDT', '1d'),
        0
    )

def _run_walk_forward_and_report(context: ContextTypes.DEFAULT_TYPE, strategy_class, symbol, timeframe):
    """Helper function to run walk-forward analysis and send report."""
    try:
        # 1. Get Data
        collector = DataCollector()
        candles_list = collector.fetch_candles(symbol, timeframe, limit=500) # Need more data for WF
        if not candles_list:
            asyncio.run(context.bot.send_message(context.job.chat_id, "Failed to fetch data for Walk-Forward analysis."))
            return

        candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
        candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms', utc=True)
        candles_df.attrs = {'symbol': symbol, 'timeframe': timeframe}

        # 2. Setup and run optimizer
        optimizer = Optimizer(strategy_class=strategy_class, data=candles_df)
        optimizer.set_params(short_window=range(10, 31, 10), long_window=range(40, 71, 15))

        wf_results = optimizer.run_walk_forward(
            in_sample_len=180, out_of_sample_len=60, step_size=60, optimize_for="sharpe_ratio"
        )

        # 3. Format and Send Report
        if wf_results:
            # Re-calculating final metrics based on the chained performance
            final_balance = 10000.0
            for r in wf_results:
                final_balance *= r['final_balance'] / r['initial_balance']
            total_pnl_percent = (final_balance - 10000.0) / 10000.0 * 100

            report = (
                f"*Walk-Forward Analysis Report for {strategy_class.__name__}*\n"
                f"`{symbol} ({timeframe})`\n\n"
                f"Total Out-of-Sample Periods: *{len(wf_results)}*\n"
                f"Chained P/L: *{total_pnl_percent:.2f}%*"
            )
        else:
            report = "Walk-Forward analysis finished with no valid results."

        asyncio.run(context.bot.send_message(context.job.chat_id, report, parse_mode='Markdown'))

    except Exception as e:
        asyncio.run(context.bot.send_message(context.job.chat_id, f"An error occurred during Walk-Forward analysis: {e}"))


async def walkforward_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Starting Walk-Forward analysis for SmaCrossoverStrategy... This will take a long time.")
    context.job_queue.run_once(
        lambda ctx: _run_walk_forward_and_report(ctx, SmaCrossoverStrategy, 'BTC/USDT', '1d'),
        0
    )

# --- Bot Setup ---

def run_bot():
    """Runs the Telegram bot to listen for commands."""
    if not settings.TELEGRAM_TOKEN:
        print("Telegram bot is not configured. Skipping.")
        return

    print("Starting Telegram bot...")
    job_queue = JobQueue()
    application = Application.builder().token(settings.TELEGRAM_TOKEN).job_queue(job_queue).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("backtest", backtest_command))
    application.add_handler(CommandHandler("optimize", optimize_command))
    application.add_handler(CommandHandler("walkforward", walkforward_command))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == '__main__':
    # This allows running the bot directly for testing.
    run_bot()
