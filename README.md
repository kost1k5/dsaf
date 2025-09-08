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

## End-to-End Workflow

This project is designed for both quantitative research and live deployment. Here are the two primary workflows.

### Workflow 1: Quantitative Research Pipeline

This pipeline allows you to test new features, strategies, and models systematically. The goal is to generate an evaluation report on a model's out-of-sample performance.

*All commands should be run from the `kost1ktrade/backend` directory.*

**Step 1: Data Collection & Feature Engineering**
```bash
# 1. Fetch historical data for your asset
pdm run python scripts/collect_all_data.py --days 730

# 2. Generate features from the raw data
pdm run python scripts/process_features.py --asset BTC --timeframe 1h

# 3. Apply labels using the Triple-Barrier Method
pdm run python scripts/apply_labels.py --asset BTC --timeframe 1h
```

**Step 2: Feature Selection**
```bash
# 4. Select the most predictive features
pdm run python scripts/select_features.py --asset BTC --timeframe 1h
```

**Step 3: Model Training, Backtesting, and Evaluation**
This step uses walk-forward validation to train models on historical data and test them on future "out-of-sample" data to get a realistic performance estimate.

```bash
# 5. Run the walk-forward backtest
pdm run python scripts/run_backtest.py --asset BTC --timeframe 1h

# 6. Evaluate the results from the backtest
pdm run python scripts/evaluate_model.py --asset BTC --timeframe 1h
```
After this, you will have a full performance report in the `reports/` directory.

---

### Workflow 2: Production Deployment

Once your research is complete and you are satisfied with a model's performance, you can deploy it for live or demo trading.

**Step 1: Train the Production Model**
Run the training script to create the final model artifact that the live system will use. The `train_model.py` script is designed for this purpose.

```bash
# Navigate to the backend directory
cd kost1ktrade/backend

# Run the training script for the desired symbol
pdm run python scripts/train_model.py --symbols BTC/USDT
```
This will save a `lgbm_classifier_BTC_USDT.joblib` file in `src/ml/models/`, which the live predictor service can then load.

**Step 2: Start the System**
Follow the instructions in the "Getting Started" section to launch both the backend API server and the frontend web interface.

**Step 3: Start the Commander**
The primary way to run the system is to start the Commander, which automates strategy selection based on the market regime.

1.  Open your browser to the API documentation (usually `http://127.0.0.1:8000/docs`).
2.  Navigate to the **Master Bot Control** section.
3.  Use the `POST /api/master-bot/start` endpoint to start the Commander.
4.  The Commander will now run in the background, managing a live trading bot.
