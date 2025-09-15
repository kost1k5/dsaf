import os
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from dotenv import load_dotenv

# --- Nested Models for Organization ---
# (These are unchanged)
class DBSettings(BaseModel):
    @property
    def DATABASE_URL(self) -> str:
        """Constructs the full database URL."""
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(backend_dir, 'data', 'local_database.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return f"sqlite:///{db_path}"

class OKXKeys(BaseModel):
    API_KEY: str
    SECRET_KEY: str
    PASSPHRASE: str

class RiskManagementSettings(BaseModel):
    MAX_DAILY_DRAWDOWN_PCT: float
    MAX_CONCURRENT_POSITIONS: int
    RISK_PER_TRADE: float
    MIN_RISK_PER_TRADE: float
    VOLATILITY_RISK_ADJUSTMENT_FACTOR: float
    MAX_DAILY_LOSS_PCT: float
    RISK_PER_TRADE_PCT: float
    ATR_MULTIPLIER: float

class TradeManagementSettings(BaseModel):
    ATR_MULTIPLIER_FOR_SL: float
    ATR_MULTIPLIER_FOR_TP1: float
    ATR_MULTIPLIER_FOR_TP2: float
    CORRELATION_RISK_REDUCTION_FACTOR: float
    VOLATILITY_SPIKE_FACTOR: float
    CORRELATION_THRESHOLD: float

class MLModelSettings(BaseModel):
    HYBRID_CONFIDENCE_THRESHOLD: float
    CONFIDENCE_THRESHOLD: float
    ATR_LABEL_THRESHOLD: float
    MIN_TRADES_FOR_EVAL: int
    SHAP_THRESHOLD: float
    CORR_THRESHOLD: float
    OPTUNA_TRIALS: int
    MIN_TRAIN_SAMPLES: int

class EvaluationSettings(BaseModel):
    INITIAL_CAPITAL: float
    RISK_PER_TRADE: float
    COMMISSION_RATE: float
    SLIPPAGE_RATE: float
    TP_ATR_MULT: float
    SL_ATR_MULT: float

class StrategySettings(BaseModel):
    EMA_FAST: int
    EMA_SLOW: int
    RSI_PERIOD: int
    RSI_ENTRY_LEVEL: int
    OBV_SMA_PERIOD: int
    ATR_PERIOD: int
    ATR_SMA_PERIOD: int
    ADX_PERIOD: int
    ADX_TREND_THRESHOLD: int
    RISK_SL_ATR_MULT: float
    RISK_TP_ATR_MULT: float

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
    # Pydantic will now read from the environment, not the file directly.
    model_config = SettingsConfigDict(
        extra='ignore',
        case_sensitive=True,
        env_nested_delimiter='__'
    )

    # --- Core Connections ---
    DB: DBSettings = Field(default_factory=DBSettings)
    OKX_REAL: Optional[OKXKeys] = None
    OKX_DEMO: Optional[OKXKeys] = None

    # --- APIs & Services (Secrets) ---
    TELEGRAM_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    FRED_API_KEY: Optional[str] = None

    # --- General Bot Settings (from .env) ---
    SYMBOLS_RAW: str
    OKX_WS_URL: str
    MAX_CANDLES: int
    TIMEFRAME: str
    COMMANDER_SYMBOL: str
    DATA_HISTORY_DAYS: int

    @computed_field
    @property
    def SYMBOLS(self) -> List[str]:
        """Returns a list of symbols from the raw string."""
        return [item.strip() for item in self.SYMBOLS_RAW.split(',')]

    # --- Backtest Settings (from .env) ---
    BACKTEST_COMMISSION_PCT: float
    BACKTEST_SLIPPAGE_PCT: float

    # --- Nested Settings Models (populated from .env via parent__child syntax) ---
    STRATEGY: StrategySettings
    RISK: RiskManagementSettings
    TRADE: TradeManagementSettings
    ML: MLModelSettings
    INDICATORS: IndicatorSettings
    EVAL: EvaluationSettings
    BACKTEST_STRATEGY: BacktestStrategySettings

# --- Manual .env Loading ---

def load_settings() -> Settings:
    """
    Manually loads the .env file and then instantiates the settings model.
    This provides a robust way to ensure the .env file is loaded correctly.
    """
    # Construct the path to the .env file relative to this config file
    # config.py is in /backend/src/core, so we need to go up three levels to get to /backend
    config_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(config_dir)
    backend_dir = os.path.dirname(src_dir)
    dotenv_path = os.path.join(backend_dir, '.env')

    if os.path.exists(dotenv_path):
        # Load the .env file into the environment
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Manually loaded .env file from: {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Settings will rely on environment variables.")

    # Now, instantiate the Settings class. It will pick up the loaded variables.
    return Settings()

# Instantiate the settings using the manual loader
settings = load_settings()
