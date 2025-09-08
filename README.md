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

## Typical Workflow

This section describes a common end-to-end workflow for using the system.

### 1. (Optional) Train a Prediction Model
The system can use an ML model for additional signal confirmation. Before running the main bot, you can train a model using the provided script.

```bash
# Navigate to the backend directory
cd kost1ktrade/backend

# Run the training script for a specific symbol
pdm run python scripts/train_model.py --symbols BTC/USDT
```

### 2. Start Backend and Frontend Servers
Follow the instructions in the "Getting Started" section to launch both the API server and the web interface.

### 3. Run the Main "Commander" Bot
The primary way to use the system is to run the Commander, which automates strategy selection.

1.  Open your browser to the API documentation (usually `http://127.0.0.1:8000/docs`).
2.  Navigate to the **Master Bot Control** section.
3.  Use the `POST /api/master-bot/start` endpoint to start the Commander.
4.  The Commander will now run in the background, analyzing the market and activating/deactivating a signal-based bot as the market regime changes.

### 4. (Alternative) Manual Bot Operation
You can also run specific bots manually for targeted tasks.

#### Example: Running a Pairs Trading Bot
1.  **Discover Pairs**: First, run the discovery script to find suitable pairs.
    ```bash
    cd kost1ktrade/backend
    pdm run python scripts/find_cointegrated_pairs.py
    ```
    Note a promising pair from the output (e.g., `('BTC/USDT', 'ETH/USDT')`).

2.  **Start the Bot**: Use the API to start the pairs bot with your chosen pair.
    *   Go to the API docs (`http://127.0.0.1:8000/docs`).
    *   Find the **Pairs Bot Control** section.
    *   Use the `POST /api/pairs-bot/start` endpoint, providing the pair in the request body.
