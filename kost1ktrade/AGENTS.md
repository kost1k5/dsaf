# Agent Instructions

This document provides instructions for developing the Kost1kTrade project.

---

## Core Data Pipeline (As of September 2025)

This section provides a high-level overview of the main data pipeline, which was significantly overhauled to improve reliability and observability.

-   **Main Entry Point:** The entire pipeline is executed via `kost1ktrade/backend/scripts/run_full_pipeline.py`. This script is responsible for data collection and then processing each asset in parallel.
-   **Subprocess Execution:** The pipeline uses a custom `run_command` function that leverages multithreading to stream `stdout` and `stderr` in real-time. This is critical for debugging and prevents the application from hanging.
-   **Data Sources:**
    -   **Macroeconomic Data (SPY, VIX, DXY):** Sourced from the **FRED API**. The implementation is in `kost1ktrade/backend/src/data_collector/macro_collector.py`. Requires a `FRED_API_KEY` in the `.env` file.
    -   **Economic Calendar:** Sourced from the **native OKX v5 API**. The implementation is in `kost1ktrade/backend/src/data_collector/calendar_data.py`. This is a public endpoint and does not require an API key.
    -   **Feature Generation:** The core logic for feature engineering is in `kost1ktrade/backend/src/processing/feature_generator.py`. This file contains extensive logging for each step.
-   **Dependencies:** All dependencies are managed by `pdm` in the `kost1ktrade/backend/pyproject.toml` file. Always use `pdm install` after pulling changes to this file.

---

## General
- The project follows a monorepo structure with a Python backend and a JavaScript/TypeScript frontend.
- The user prefers detailed explanations of progress and clear commit messages.

## Backend
- The backend is a Python project using `pdm` for dependency management, with its configuration in `pyproject.toml`. The Python version is 3.12.
- All backend source code should be placed within the `kost1ktrade/backend/src` directory, following the existing modular structure (`api`, `database`, `core`, etc.).
- When adding new dependencies, use `pdm add <package_name>`.
- To run commands within the project's virtual environment, use `pdm run <command>`.
- The database is SQLite, with the database file located at `kost1ktrade/backend/data/local_database.db`. All database models are defined using SQLAlchemy in `kost1ktrade/backend/src/database/models.py`.
- The API is built with FastAPI. New endpoints should be added logically within the `kost1ktrade/backend/src/api` module.
- Notifications are handled in the `src/notifications` module.
- The project now supports two types of bots, managed by separate controllers:
  - `src/core/bot_controller.py` for signal-based strategies.
  - `src/core/grid_bot_controller.py` for grid trading logic.
- Signal-based strategies should be created in `src/strategies` and inherit from `BaseStrategy`. Grid strategies are self-contained.
- The `src/optimization` module contains the `Optimizer` class and the `walk_forward.py` splitter for advanced backtesting.

## Data Collection
- The `DataCollector` class in `kost1ktrade/backend/src/data_collector/collector.py` is responsible for fetching data from exchanges.
- A new method `fetch_funding_rate_history_from_archive` has been implemented to collect historical funding rate data from OKX.
- This method fetches a list of ZIP archives from the `/api/v5/public/market-data-history` endpoint, downloads them, extracts the CSV files, and parses the data.
- **Important Note:** As of September 2025, the OKX API is not returning any data for funding rate history requests via this endpoint. The implementation is in place for when the API starts providing the data.

## Frontend
- The frontend will be a modern JavaScript application (e.g., React, Vue).
- All frontend code resides in the `kost1ktrade/frontend` directory.
- The visual style should adhere to a "cosmic" or "space" theme.

## Running the Application

- **Backend:** From the `kost1ktrade/backend` directory, run `pdm run uvicorn src.api.main:app --reload`.
- **Frontend:** From the `kost1ktrade/frontend` directory, run `npm run dev`.
- **Database initialization:** From the `kost1ktrade/backend` directory, run `pdm run python scripts/create_tables.py`.

## Workflow
1.  Always clarify requirements if they are ambiguous.
2.  Propose a clear, step-by-step plan before starting work.
3.  Verify each step after completion (e.g., by reading files, running commands).
4.  Run tests before submitting work. If no automated tests exist, perform manual verification of the functionality.
5.  Request a code review before submitting the final changes.
6.  Use descriptive branch names (e.g., `feature/add-new-strategy`) and commit messages.

---

## System Overhaul (September 2025)

This update represents a major architectural evolution, turning the system into a more robust and versatile framework.

### 1. The "Commander" & Dynamic Risk Management
- **Concept**: The `master_controller.py` now acts as a central "Commander". Its sole purpose is to determine the market regime and authorize strategies, not to run them directly.
- **Market Regime Filter**:
    - The Commander uses the **ADX indicator** on a primary market symbol (defined in `config.py` as `COMMANDER_SYMBOL`) to classify the market as `"trend"` (ADX > 25) or `"range"` (ADX < 20).
    - It then updates the `bot_state.active_strategies` dictionary, enabling only those strategies whose `type` in `strategy_params.json` matches the current regime.
- **Bot Lifecycle Management**: The Commander is also responsible for managing the lifecycle of a single signal-based bot. It ensures that if a bot is running, its strategy is appropriate for the current regime. If not, it will stop the current bot and start a new, suitable one.
- **Dynamic Position Sizing**:
    - A new `risk_manager.py` module has been introduced.
    - The `bot_controller.py` and `backtester.py` now use this module to calculate position size dynamically based on **ATR (Average True Range)**.
    - This normalizes risk per trade to a fixed percentage of capital. The core parameters (`RISK_PER_TRADE_PCT`, `ATR_MULTIPLIER`) are now configurable in `config.py`.

### 2. New Strategy Types
- **Hybrid Strategies**:
    - A framework for creating consensus-based strategies has been added in `strategies/hybrid_base.py`.
    - An example, `rsi_macd_hybrid.py`, demonstrates how to create a strategy that requires signals from both RSI and MACD to fire.
    - To use a hybrid strategy, it must be added to `strategy_params.json` with a `type` field.
- **Pairs Trading (Statistical Arbitrage)**:
    - The system now supports pairs trading.
    - A new `PairsTradingStrategy` (`strategies/pairs_trading_strategy.py`) trades based on the Z-score of the spread between two assets.
    - A dedicated controller (`core/pairs_bot_controller.py`) manages the two-legged trades.
    - New API endpoints (`/api/pairs-bot/start`, `/api/pairs-bot/stop`) have been added to control the pairs bot.

### 3. Expanded Quantitative & ML Pipeline
- **Walk-Forward Optimization**: The new `scripts/run_wfo.py` script allows for robust WFO of strategy parameters. This is a powerful tool for validating strategy robustness.
- **Pairs Discovery**: The `scripts/find_cointegrated_pairs.py` script automates the process of finding correlated and cointegrated asset pairs suitable for the new pairs trading strategy.
- **XGBoost Model Training**: The `scripts/train_xgboost_model.py` script has been added as an alternative to the LightGBM trainer, allowing for experimentation with different ML algorithms.
- **Enhanced Feature Engineering**: The `ml/feature_generator.py` module has been updated with new features, including `High-Low` volatility, `Close-Open` impulse, and additional SMAs and standard deviation metrics. All new price-based features are normalized for stationarity.

---

## Advanced ML Methodologies (September 2025)

This update focuses on improving the methodological robustness of the machine learning pipeline to ensure more reliable model evaluation and feature generation.

### 1. Purged Time-Series Cross-Validation
- **Problem**: Standard cross-validation methods are unsuitable for time series data. Even `TimeSeriesSplit` can suffer from data leakage when labels are generated using future information (as in the Triple-Barrier Method).
- **Solution**: A custom `PurgedTimeSeriesSplit` has been implemented in `kost1ktrade/backend/src/ml/validation.py`.
- **Mechanism**:
    - It uses `TimeSeriesSplit` to create initial folds.
    - It then "purges" the end of each training set, removing any samples whose labels were created using data from the corresponding test set.
    - This requires an `event_end_time` for each sample, which is now generated by `apply_triple_barrier` and stored in the labeled data files.
- **Impact**: Provides a more realistic and reliable estimate of the model's out-of-sample performance during hyperparameter tuning.

### 2. Economic Calendar Integration
- **Goal**: To incorporate fundamental data into the ML model.
- **Implementation**:
    - A new data collector, `kost1ktrade/backend/src/data_collector/calendar_data.py`, fetches economic calendar data from Investing.com via the Apify platform.
    - **Requires an `APIFY_API_TOKEN`** to be set in the `.env` file.
    - A new database model, `EconomicCalendarEvent`, stores the data.
    - `FeatureGenerator` has been updated with an `add_calendar_features` method to create features like `minutes_to_next_event` and `is_high_impact_event_in_next_24h`.

### 3. Two-Stage Feature Selection
- **Goal**: To select a feature set that is both predictive and non-redundant.
- **Mechanism**: The `scripts/select_features.py` script now performs a two-stage selection process:
    1. **SHAP-based Filtering**: It first selects features with a SHAP importance score above a certain threshold.
    2. **Correlation Pruning**: It then calculates the correlation matrix of the selected features and removes highly correlated features, keeping only the one with the higher SHAP importance from each correlated pair.
- **Impact**: Leads to a more parsimonious and robust model by reducing multicollinearity.

## Quantitative Analysis Pipeline (September 2025)

The quantitative research pipeline has been expanded. All scripts are located in `kost1ktrade/backend/scripts/` and should be run from the `kost1ktrade/backend/` directory using `pdm run python ...`.

### **Data & Feature Pipeline**
1.  `collect_all_data.py`: Fetches raw OHLCV data.
2.  `process_features.py`: Generates a full feature set from raw data.
3.  `apply_labels.py`: Applies Triple-Barrier Method labeling to a feature set.

### **ML Model Pipeline**
4.  `train_model.py`: Trains the primary **LightGBM** model. Includes feature selection, hyperparameter tuning (Optuna), and SHAP analysis.
5.  `train_xgboost_model.py`: **(New)** Trains an alternative **XGBoost** model for comparison.
6.  `run_backtest.py`: Executes a full walk-forward validation of the ML model's predictions.
7.  `evaluate_model.py`: Evaluates the out-of-sample predictions to generate final performance metrics.

### **Strategy & Optimization Tools**
8. `run_wfo.py`: **(New)** Performs Walk-Forward Optimization for classic (non-ML) strategies to find robust parameters.
9. `find_cointegrated_pairs.py`: **(New)** Scans multiple symbols to find statistically significant pairs for the pairs trading strategy.

---

## Future Roadmap: The "Nebula Command Bridge" Vision

This document outlines the grand strategic vision for the trading bot. It is divided into phases and sectors. Completed items should be removed or marked as complete.

### **Phase 1: The Simulation Deck Engine (In Progress)**
- **Goal:** Build the backend core required to power the "Simulation Deck" frontend.
- **Key Features:**
    - **1. Implement Backend for Simulation Deck:** Create a new `Backtester` module and a `POST /api/simulation/run` endpoint that can simultaneously backtest up to 10 strategies and return detailed performance reports.
    - **2. Expand Strategy Library:** Implement 5 new strategies (Stochastic, Awesome Oscillator, Parabolic SAR, Keltner Channels, Ichimoku Cloud) to provide a diverse set of "research probes".

### **Phase 2: The "Advanced Arsenal" - AI & News Integration**
- **Goal:** Enhance the bot's intelligence by integrating Machine Learning predictions and real-time news sentiment.
- **Key Features:**
    - **1. Implement ML Backend:**
        - Add `lightgbm` and `scikit-learn` as dependencies.
        - Create a feature engineering module to prepare data.
        - Create a script to train a LightGBM model to predict short-term price movements and save the artifact.
        - Build a prediction service to serve forecasts from the trained model.
    - **2. Implement News Sentiment Analysis:**
        - Add `textblob` as a dependency.
        - Create a module to fetch news via the News API and calculate a sentiment score.
    - **3. Integrate into Master Controller:** Upgrade the Master Controller's logic to use the ML forecast and news sentiment as additional filters for making trading decisions.

---

### **Frontend Vision: The Nebula Command Bridge**

This describes the target user interface. The implementation will be phased.

#### **Core Concept**
The user is the Admiral of a fleet of 10 "research probes" (strategies). The primary interface is a "Simulation Deck" used to backtest these probes against historical data ("spacetime anomalies") to select the best one for a live mission.

#### **Sector 1: Main Bridge (Dashboard)**
- **"Engine Status":** Central animated widget for active bot status.
- **"Asset Galaxy":** Interactive visualization of portfolio distribution.
- **"Profitability Compass":** Circular PnL indicator.
- **"Captain's Log":** Event feed for the active bot.

#### **Sector 2: Navigation (Market Analysis)**
- **"Star Charts":** Holographic-style price charts.
- **"Sector Scanner":** Market-wide heatmap.
- **"Flight Trajectory":** Trade history visualized on the price chart.

#### **Sector 3: Engineering (Live Bot Management)**
- **"Active Strategy Module":** Widget showing the live strategy's name and parameters, with pause/stop controls.
- **"Hyperdrive Settings":** Panel for real-time adjustments to the live bot (e.g., order size, risk).

#### **Sector 4: Simulation Deck (Backtesting & Optimization)**
- **This is the primary goal of the current development phase.**
- **Step 1: "Pre-flight Briefing" (Setup):** A UI with 10 "docking slots" for strategies. Users select a trading pair, a date range (via a "time loop" timeline), and configure the 10 strategies to be tested.
- **Step 2: "'Quantum Leap' Simulation" (Execution):** An animated sequence showing the 10 "probes" traveling along parallel time-trajectories. Real-time metrics (PnL, Drawdown) are displayed for each probe during the simulation.
- **Step 3: "Holographic Debrief" (Results Analysis):**
    - **"Parallel Universes" Chart:** A primary chart showing the price history with 10 semi-transparent, colored equity curves overlaid. Interactive tooltips and the ability to toggle lines.
    - **"Fleet Comparison" Table:** A sortable data grid comparing all 10 strategies across key metrics (Total PnL, Max Drawdown, Profit Factor, Winrate, etc.).
    - **"Efficiency Web" Radar Chart:** A spider chart for visually comparing the strengths and weaknesses of the top strategies across multiple axes (Profitability, Reliability, Frequency).
- **Step 4: "Select Flagship" (Deployment):** A button next to the winning strategy in the results table that sends the user to Sector 3 (Engineering) with the chosen strategy and parameters loaded, ready for live deployment.

#### **Sector 5: Admiral's Hall of Fame (Gamification)**
- A section for achievements related to backtesting and live performance.
- **Examples:** "Theorist-Engineer" (first simulation), "Time Lord" (test >1 year), "Quantum Optimizer" (find a >50% PnL strategy).

#### **Sector 6: Astrophysics Lab (AI Analysis)**
- **"Gravity Anomaly Scanner":** 3D visualization of market volume and volatility for various assets.
- **"'Oracle' Trajectory Prediction":** An ML model that projects a cone of future price probabilities on the main chart.

#### **Sector 7: Shield Room (Risk Management)**
- **"Ship Integrity" Widget:** A 3D model of the "Nebula" ship with shield sectors representing different risk metrics (Concentration, Drawdown, Correlation). Shields weaken and flash warnings as risk increases.
- **"Power Distribution" Module:** An interactive UI for allocating capital between different live strategies, showing the real-time impact on the "Ship Integrity" risk profile.

#### **Sector 8: Communications Hub (Social & News)**
- **"Galaxy Echo" Widget:** A visualization of social media sentiment (e.g., a vortex of green/red particles for positive/negative sentiment).
- **"Outpost Signals":** Integration to display trading ideas from selected TradingView authors directly on the price chart.
