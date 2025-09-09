from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

# --- Nested Models for Organization ---

class DBSettings(BaseModel):
    # USER: str = "postgres"
    # PASS: str = "Kostya1790"
    # HOST: str = "localhost"
    # PORT: int = 5432
    # NAME: str = "trading_bot_data"

    @property
    def DATABASE_URL(self) -> str:
        """Constructs the full database URL."""
        # return f"postgresql+psycopg2://{self.USER}:{self.PASS}@{self.HOST}:{self.PORT}/{self.NAME}"
        return "sqlite:///data/local_database.db"

class OKXKeys(BaseModel):
    API_KEY: str
    SECRET_KEY: str
    PASSPHRASE: str

class RiskManagementSettings(BaseModel):
    MAX_DAILY_DRAWDOWN_PCT: float = 0.03
    MAX_CONCURRENT_POSITIONS: int = 4
    RISK_PER_TRADE: float = 0.03
    MIN_RISK_PER_TRADE: float = 0.005
    VOLATILITY_RISK_ADJUSTMENT_FACTOR: float = 1.0
    MAX_DAILY_LOSS_PCT: float = 0.05
    # New settings for ATR-based position sizing
    RISK_PER_TRADE_PCT: float = 1.0  # e.g., 1.0 for 1%
    ATR_MULTIPLIER: float = 2.0      # e.g., 2.0 for a 2x ATR stop loss

class TradeManagementSettings(BaseModel):
    ATR_MULTIPLIER_FOR_SL: float = 1.2
    ATR_MULTIPLIER_FOR_TP1: float = 1.5
    ATR_MULTIPLIER_FOR_TP2: float = 3.0
    CORRELATION_RISK_REDUCTION_FACTOR: float = 0.7
    VOLATILITY_SPIKE_FACTOR: float = 2.5
    CORRELATION_THRESHOLD: float = 0.7

class MLModelSettings(BaseModel):
    CONFIDENCE_THRESHOLD: float = 0.4
    ATR_LABEL_THRESHOLD: float = 0.4

class IndicatorSettings(BaseModel):
    RSI_PERIOD: int = 14
    EMA_FAST_PERIOD: int = 12
    EMA_SLOW_PERIOD: int = 26
    BB_PERIOD: int = 20
    BB_STD_DEV: int = 2
    ATR_PERIOD: int = 14
    ADX_PERIOD: int = 14
    VWAP_PERIOD: int = 20
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    SMA_PERIOD: int = 20
    SMA_LONG_PERIOD: int = 50
    STOCH_K_PERIOD: int = 14
    STOCH_D_PERIOD: int = 3
    AO_FAST_PERIOD: int = 5
    AO_SLOW_PERIOD: int = 34
    PSAR_ACCELERATION: float = 0.02
    PSAR_MAXIMUM: float = 0.2
    KC_LENGTH: int = 20
    KC_MULTIPLIER: float = 2.0
    KC_ATR_LENGTH: int = 14
    IC_TENKAN: int = 9
    IC_KIJUN: int = 26
    IC_SENKOU_B: int = 52

# --- Main Settings Class ---

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=True,
        env_nested_delimiter='__'
    )

    # --- Core Connections ---
    DB: DBSettings = Field(default_factory=DBSettings)
    OKX_REAL: Optional[OKXKeys] = None
    OKX_DEMO: Optional[OKXKeys] = None

    # --- APIs & Services ---
    TELEGRAM_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    APIFY_API_TOKEN: Optional[str] = None

    # --- General Bot Settings ---
    # Raw comma-separated string for symbols from .env
    SYMBOLS_RAW: str = "BTC-USDT,ETH-USDT,SOL-USDT,LINK-USDT"

    @computed_field
    @property
    def SYMBOLS(self) -> List[str]:
        """Returns a list of symbols from the raw string."""
        return [item.strip() for item in self.SYMBOLS_RAW.split(',')]

    OKX_WS_URL: str = "wss://ws.okx.com:8443/ws/v5/public"
    MAX_CANDLES: int = 5000
    TIMEFRAME: str = "4h"
    COMMANDER_SYMBOL: str = "BTC-USDT-SWAP"

    # --- Backtest Settings ---
    BACKTEST: RiskManagementSettings = Field(default_factory=RiskManagementSettings)
    BACKTEST_COMMISSION_PCT: float = 0.001
    BACKTEST_SLIPPAGE_PCT: float = 0.0005

    # --- Live Trading Settings ---
    RISK: RiskManagementSettings = Field(default_factory=RiskManagementSettings)
    TRADE: TradeManagementSettings = Field(default_factory=TradeManagementSettings)

    # --- ML & Indicator Settings ---
    ML: MLModelSettings = Field(default_factory=MLModelSettings)
    INDICATORS: IndicatorSettings = Field(default_factory=IndicatorSettings)

# Instantiate the settings
settings = Settings()

# Example of how to access a nested setting:
# from src.core.config import settings
# print(settings.DB.DATABASE_URL)
# print(settings.OKX_REAL.API_KEY)
# print(settings.INDICATORS.RSI_PERIOD)
