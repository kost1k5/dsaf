from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

# --- Nested Models for Organization ---

class DBSettings(BaseModel):
    USER: str
    PASS: str
    HOST: str
    PORT: int
    NAME: str

    @property
    def DATABASE_URL(self) -> str:
        """Constructs the full database URL for PostgreSQL."""
        return f"postgresql+psycopg2://{self.USER}:{self.PASS}@{self.HOST}:{self.PORT}/{self.NAME}"

class OKXKeys(BaseModel):
    API_KEY: str
    SECRET_KEY: str
    PASSPHRASE: str

class MasterControllerSettings(BaseModel):
    CHECK_INTERVAL_SECONDS: int
    ADX_TREND_THRESHOLD: int
    ADX_RANGE_THRESHOLD: int
    BOT_STOP_WAIT_SECONDS: int

class BotControllerSettings(BaseModel):
    CANDLE_LIMIT: int
    MIN_CAPITAL_FOR_TRADE: int
    MIN_ORDER_SIZE_USD: int
    LOOP_SLEEP_SECONDS: int

class EvaluationSettings(BaseModel):
    INITIAL_CAPITAL: float
    RISK_PER_TRADE: float
    COMMISSION_RATE: float
    SLIPPAGE_RATE: float
    TP_ATR_MULT: float
    SL_ATR_MULT: float

class RiskManagementSettings(BaseModel):
    MAX_DAILY_DRAWDOWN_PCT: float
    MAX_CONCURRENT_POSITIONS: int
    RISK_PER_TRADE: float
    MIN_RISK_PER_TRADE: float
    VOLATILITY_RISK_ADJUSTMENT_FACTOR: float
    MAX_DAILY_LOSS_PCT: float
    RISK_PER_TRADE_PCT: float # This seems to be missing from the user's .env, will need a default
    ATR_MULTIPLIER: float

class TradeManagementSettings(BaseModel):
    ATR_MULTIPLIER_FOR_SL: float
    ATR_MULTIPLIER_FOR_TP1: float
    ATR_MULTIPLIER_FOR_TP2: float
    CORRELATION_RISK_REDUCTION_FACTOR: float
    VOLATILITY_SPIKE_FACTOR: float
    CORRELATION_THRESHOLD: float

class MLModelSettings(BaseModel):
    CONFIDENCE_THRESHOLD: float
    HYBRID_CONFIDENCE_THRESHOLD: float = 0.6 # Add default based on my memory
    ATR_LABEL_THRESHOLD: float
    MIN_TRADES_FOR_EVAL: int
    SHAP_THRESHOLD: float
    CORR_THRESHOLD: float
    OPTUNA_TRIALS: int
    MIN_TRAIN_SAMPLES: int

class BacktestStrategySettings(BaseModel):
    TP_ATR_MULT: float
    SL_ATR_MULT: float
    CONFIDENCE_THRESHOLD: float
    MAX_HOLDING_PERIOD: int
    HIGH_CONFIDENCE_THRESHOLD: float
    HIGH_RISK_PCT: float
    MED_RISK_PCT: float
    LOW_RISK_PCT: float
    LOSS_STREAK_THRESHOLD: int
    TRADING_PAUSE_HOURS: int

class IndicatorSettings(BaseModel):
    RSI_PERIOD: int
    EMA_FAST_PERIOD: int
    EMA_SLOW_PERIOD: int
    BB_PERIOD: int
    BB_STD_DEV: int
    ATR_PERIOD: int
    ADX_PERIOD: int
    VWAP_PERIOD: int
    MACD_FAST: int
    MACD_SLOW: int
    MACD_SIGNAL: int
    SMA_PERIOD: int
    SMA_LONG_PERIOD: int
    STOCH_K_PERIOD: int
    STOCH_D_PERIOD: int
    AO_FAST_PERIOD: int
    AO_SLOW_PERIOD: int
    PSAR_ACCELERATION: float
    PSAR_MAXIMUM: float
    KC_LENGTH: int
    KC_MULTIPLIER: float
    KC_ATR_LENGTH: int
    IC_TENKAN: int
    IC_KIJUN: int
    IC_SENKOU_B: int

# --- Main Settings Class ---

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False, # Set to False to match case-insensitivity of env vars
        env_nested_delimiter='__'
    )

    # Core Connections
    DB: DBSettings
    OKX_REAL: OKXKeys
    OKX_DEMO: OKXKeys

    # APIs & Services
    TELEGRAM_TOKEN: str
    TELEGRAM_CHAT_ID: str
    NEWS_API_KEY: str
    GEMINI_API_KEY: str
    FRED_API_KEY: str

    # General Bot Settings
    SYMBOLS_RAW: str
    OKX_WS_URL: str
    MAX_CANDLES: int
    TIMEFRAME: str
    COMMANDER_SYMBOL: str
    DATA_HISTORY_DAYS: int

    # Backtest Settings
    BACKTEST_COMMISSION_PCT: float
    BACKTEST_SLIPPAGE_PCT: float

    # Nested Settings Models
    STRATEGY: "StrategySettings" # Forward ref for future use if needed
    RISK: RiskManagementSettings
    TRADE: TradeManagementSettings
    ML: MLModelSettings
    INDICATORS: IndicatorSettings
    EVAL: EvaluationSettings
    BACKTEST_STRATEGY: BacktestStrategySettings

    # These are now optional, with defaults provided by their models
    MASTER_CONTROLLER: MasterControllerSettings = Field(default_factory=MasterControllerSettings)
    BOT_CONTROLLER: BotControllerSettings = Field(default_factory=BotControllerSettings)

    @property
    def SYMBOLS(self) -> List[str]:
        return [item.strip() for item in self.SYMBOLS_RAW.split(',')]

class StrategySettings(BaseModel):
    # This is defined after Settings to avoid circular dependency issues with forward refs
    # if they were ever needed, though not strictly necessary here.
    pass

# Instantiate the settings
settings = Settings()
