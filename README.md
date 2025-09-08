# Kost1kTrade: A Multi-Strategy Algorithmic Trading Framework

Kost1kTrade is a comprehensive, event-driven framework for developing, backtesting, and deploying algorithmic trading strategies. It features a Python backend powered by FastAPI and a sophisticated quantitative analysis pipeline, along with a modern JavaScript frontend for control and visualization.

## Key Features

*   **Multi-Strategy Support**: Natively supports multiple types of trading strategies:
    *   **Signal-based**: Classic strategies based on technical indicators (RSI, MACD, etc.).
    *   **Hybrid**: Consensus-based strategies that combine signals from multiple indicators for higher conviction trades.
    *   **Pairs Trading**: A statistical arbitrage strategy based on cointegrated asset pairs.
    *   **Grid Trading**: Automated grid bots for range-bound markets.

*   **"Commander" Mode**: An intelligent master controller that analyzes the overall market regime using the ADX indicator. It automatically activates strategies suitable for the current market condition (e.g., trend-following strategies in a trending market, mean-reversion strategies in a ranging market).

*   **Dynamic Risk Management**: Integrated position sizing based on the Average True Range (ATR). This normalizes risk across all trades, ensuring that the capital at risk for any given trade is a fixed, predictable percentage of the total portfolio.

*   **Advanced Quantitative Pipeline**: A suite of command-line scripts for professional-grade quantitative research:
    *   **Walk-Forward Optimization**: Test and optimize strategy parameters on out-of-sample data to prevent overfitting and get a realistic measure of performance.
    *   **Pairs Discovery**: Automatically scan a universe of assets to find statistically significant pairs for the arbitrage strategy.
    *   **Advanced ML Model Training**: Train and evaluate sophisticated prediction models (LightGBM and XGBoost) with feature selection and SHAP analysis.

*   **Modern API & Frontend**:
    *   A robust **FastAPI** backend provides endpoints for controlling all bot types, managing strategies, and running simulations.
    *   A **React/TypeScript** frontend (The "Nebula Command Bridge") provides a user-friendly interface for interacting with the system.

## Getting Started

### Prerequisites
- Python 3.12+
- `pdm` for backend dependency management
- `npm` for frontend dependency management

### Backend Setup
1.  Navigate to the `kost1ktrade/backend` directory.
2.  Install dependencies: `pdm install`
3.  Initialize the database: `pdm run python scripts/create_tables.py`
4.  Run the API server: `pdm run uvicorn src.api.main:app --reload`
    *   The API will be available at `http://127.0.0.1:8000/docs`.

### Frontend Setup
1.  Navigate to the `kost1ktrade/frontend` directory.
2.  Install dependencies: `npm install`
3.  Run the development server: `npm run dev`
    *   The frontend will be available at `http://127.0.0.1:5173`.
