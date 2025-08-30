import enum
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    Enum,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class Candle(Base):
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    interval = Column(String, nullable=False, index=True)
    open_time = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint('symbol', 'interval', 'open_time', name='_symbol_interval_open_time_uc'),
    )

    def __repr__(self):
        return f"<Candle(symbol='{self.symbol}', interval='{self.interval}', open_time='{self.open_time}')>"


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String)
    is_active = Column(Boolean, default=True, nullable=False)

    trades = relationship("Trade", back_populates="strategy")

    def __repr__(self):
        return f"<Strategy(name='{self.name}', is_active={self.is_active})>"


class TradeSide(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    symbol = Column(String, nullable=False, index=True)
    exchange_order_id = Column(String, nullable=False, unique=True)
    side = Column(Enum(TradeSide), nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())

    strategy = relationship("Strategy", back_populates="trades")

    def __repr__(self):
        return f"<Trade(symbol='{self.symbol}', side='{self.side.name}', price={self.price}, quantity={self.quantity})>"


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True)
    strategy_name = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    initial_balance = Column(Float, nullable=False)
    final_balance = Column(Float, nullable=False)
    pnl_usd = Column(Float, nullable=False)
    pnl_percent = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    run_at = Column(DateTime, server_default=func.now())
    trades = Column(JSON) # To store the list of trade dicts

    def __repr__(self):
        return f"<BacktestResult(strategy='{self.strategy_name}', symbol='{self.symbol}', pnl_percent={self.pnl_percent:.2f}%)>"
