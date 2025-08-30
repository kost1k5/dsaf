import asyncio
import pandas as pd
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, JobQueue

from src.core.config import settings
from src.core.bot_state import bot_state
from scripts.run_backtest import run_backtest
from src.optimization.optimizer import Optimizer
from src.strategies.sma_crossover import SmaCrossoverStrategy
from src.data_collector.collector import DataCollector

# --- Standalone Notifier Function ---

async def send_telegram_notification(message: str):
    """A simple, standalone function to send a notification."""
    if not settings.TELEGRAM_TOKEN or not settings.TELEGRAM_CHAT_ID:
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

async def _run_backtest_and_report(context: ContextTypes.DEFAULT_TYPE):
    """Helper coroutine to run backtest and send report."""
    job = context.job
    strategy_class, symbol, timeframe = job.data
    try:
        collector = DataCollector()
        candles_list = collector.fetch_candles(symbol, timeframe, limit=365)
        if not candles_list:
            await context.bot.send_message(job.chat_id, "Failed to fetch data for backtest.")
            return

        candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
        candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms', utc=True)
        candles_df.attrs = {'symbol': symbol, 'timeframe': timeframe}

        strategy = strategy_class()
        results = run_backtest(
            strategy=strategy, data=candles_df,
            commission_pct=settings.BACKTEST_COMMISSION_PCT,
            slippage_pct=settings.BACKTEST_SLIPPAGE_PCT
        )

        report_lines = [f"*Backtest Report for {strategy_class.__name__}*..."] # Shortened for brevity
        if results['trades']:
            report_lines.append("\n*Recent Trades:*")
            for trade in results['trades'][-5:]:
                pnl = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100
                emoji = "✅" if pnl > 0 else "❌"
                report_lines.append(f"{emoji} `PnL: {pnl:.2f}%`")

        report = "\n".join(report_lines)
        await context.bot.send_message(job.chat_id, report, parse_mode='Markdown')
    except Exception as e:
        await context.bot.send_message(job.chat_id, f"An error occurred during backtest: {e}")

async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Starting backtest for SmaCrossoverStrategy...")
    context.job_queue.run_once(_run_backtest_and_report, 0, data=(SmaCrossoverStrategy, 'BTC/USDT', '1d'), name=f"backtest_{update.effective_chat.id}")

async def _run_optimization_and_report(context: ContextTypes.DEFAULT_TYPE):
    """Helper coroutine to run optimization and send report."""
    job = context.job
    strategy_class, symbol, timeframe = job.data
    try:
        collector = DataCollector()
        candles_list = collector.fetch_candles(symbol, timeframe, limit=365)
        if not candles_list:
            await context.bot.send_message(job.chat_id, "Failed to fetch data for optimization.")
            return

        candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
        candles_df['open_time'] = pd.to_datetime(candles_df['open_time'], unit='ms', utc=True)
        candles_df.attrs = {'symbol': symbol, 'timeframe': timeframe}

        optimizer = Optimizer(strategy_class=strategy_class, data=candles_df)
        optimizer.set_params(short_window=range(20, 51, 10), long_window=range(60, 101, 20))
        best_params, best_metrics = optimizer.run_single(optimize_for="sharpe_ratio")

        if best_params:
            report = f"*Optimization Report for {strategy_class.__name__}*\n...Best Params: `{best_params}`"
        else:
            report = "Optimization finished with no valid results."
        await context.bot.send_message(job.chat_id, report, parse_mode='Markdown')
    except Exception as e:
        await context.bot.send_message(job.chat_id, f"An error occurred during optimization: {e}")

async def optimize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Starting optimization for SmaCrossoverStrategy...")
    context.job_queue.run_once(_run_optimization_and_report, 0, data=(SmaCrossoverStrategy, 'BTC/USDT', '1d'), name=f"optimize_{update.effective_chat.id}")

def run_bot():
    """Runs the Telegram bot to listen for commands."""
    if not settings.TELEGRAM_TOKEN:
        print("Telegram bot is not configured. Skipping.")
        return

    print("Starting Telegram bot...")
    job_queue = JobQueue()
    application = Application.builder().token(settings.TELEGRAM_TOKEN).job_queue(job_queue).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("backtest", backtest_command))
    application.add_handler(CommandHandler("optimize", optimize_command))
    application.add_handler(CommandHandler("walkforward", walkforward_command))

    application.run_polling()

if __name__ == '__main__':
    run_bot()
