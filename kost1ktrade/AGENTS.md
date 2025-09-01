# Agent Instructions

This document provides instructions for developing the Kost1kTrade project.

## General
- The project follows a monorepo structure with a Python backend and a JavaScript/TypeScript frontend.
- The user prefers detailed explanations of progress and clear commit messages.

## Backend
- The backend is a Python project using `pipenv` for dependency management. The Python version is 3.12.
- All backend source code should be placed within the `kost1ktrade/backend/src` directory, following the existing modular structure (`api`, `database`, `core`, etc.).
- When adding new dependencies, use `pipenv install <package_name>`.
- To run commands within the project's virtual environment, use `pipenv run <command>`.
- The database is PostgreSQL. All database models are defined using SQLAlchemy in `kost1ktrade/backend/src/database/models.py`.
- The API is built with FastAPI. New endpoints should be added logically within the `kost1ktrade/backend/src/api` module.
- Notifications are handled in the `src/notifications` module.
- The project now supports two types of bots, managed by separate controllers:
  - `src/core/bot_controller.py` for signal-based strategies.
  - `src/core/grid_bot_controller.py` for grid trading logic.
- Signal-based strategies should be created in `src/strategies` and inherit from `BaseStrategy`. Grid strategies are self-contained.
- The `src/optimization` module contains the `Optimizer` class and the `walk_forward.py` splitter for advanced backtesting.

## Frontend
- The frontend will be a modern JavaScript application (e.g., React, Vue).
- All frontend code resides in the `kost1ktrade/frontend` directory.
- The visual style should adhere to a "cosmic" or "space" theme.

## Running the Application

- **Backend:** From the `kost1ktrade/backend` directory, run `pipenv run uvicorn src.api.main:app --reload`.
- **Frontend:** From the `kost1ktrade/frontend` directory, run `npm run dev`.
- **Database initialization:** From the `kost1ktrade/backend` directory, run `pipenv run python scripts/create_tables.py`.

## Workflow
1.  Always clarify requirements if they are ambiguous.
2.  Propose a clear, step-by-step plan before starting work.
3.  Verify each step after completion (e.g., by reading files, running commands).
4.  Run tests before submitting work. If no automated tests exist, perform manual verification of the functionality.
5.  Request a code review before submitting the final changes.
6.  Use descriptive branch names (e.g., `feature/add-new-strategy`) and commit messages.

---

## Recent Architectural Updates (August 2025)

This section details significant changes and improvements made to the backend architecture. Future agents should be aware of these systems.

### 1. Strategy Activation Bug Fix
- **Problem:** The `MasterController` was not activating signal-based strategies because of a bug in the `/api/strategies/status` endpoint. The endpoint only updated the status of strategies already present in the `bot_state.active_strategies` dictionary, which was empty on startup.
- **Solution:**
    - The check in the API endpoint was removed.
    - The `BotState` class was refactored to dynamically load all strategies from `strategy_params.json` on startup, making the system more robust.

### 2. Advanced Machine Learning Pipeline (September 2025)
The ML pipeline was fundamentally overhauled to improve prediction quality and robustness, addressing the low performance of the previous model. The new methodology is implemented in `scripts/train_model.py`.

#### **Core Methodological Changes**

-   **Binary Classification with Noise Filtering:** The model no longer predicts "Sideways" movements. It's now a binary classifier (Up/Down). The training data is filtered using a volatility threshold (based on ATR) to remove low-impact, noisy price movements, focusing the model on significant events.
-   **Feature Stationarity:** All price-based technical indicators (e.g., SMAs, Bollinger Bands) are transformed into stationary series (e.g., by normalizing against the current price). This prevents the model from learning spurious, price-dependent correlations. An ADF test is included in the pipeline to verify the stationarity of key features.
-   **Walk-Forward Validation:** The pipeline continues to use `TimeSeriesSplit` for cross-validation, ensuring that the model is always trained on past data and validated on future data to prevent lookahead bias.

#### **Expanded Feature Set**

The model now incorporates a much richer set of features:

-   **Technical Indicators:** A comprehensive set of indicators from `pandas-ta-openbb`, including VWAP.
-   **Market Sentiment:**
    -   **Fear & Greed Index:** Daily historical F&G index values are merged into the dataset.
    -   **News Sentiment:** Daily sentiment is calculated from crypto news headlines via the CryptoPanic API.
-   **On-Chain Metrics (Placeholder):** The pipeline includes a placeholder framework to integrate on-chain data (e.g., Net Exchange Flow, SOPR). A real implementation requires an API key from a provider like Glassnode.

#### **Dependencies**

The new pipeline requires several new libraries. Ensure they are in your `Pipfile` and installed (`pipenv install`):
-   `statsmodels`: For the ADF stationarity test.
-   `cryptopanic-api`: To fetch news data.
-   `fear-greed-index`: To fetch the Fear & Greed index.
-   `optuna`: For Bayesian hyperparameter optimization.
-   `shap`: For model interpretation.

#### **Configuration**

To use the external data features, you must set the following environment variable in your `.env` file:
-   `CRYPTOPANIC_API_KEY`: Your API key from [CryptoPanic](https://cryptopanic.com/developers/api/).
-   `ONCHAIN_API_KEY`: (For future use) Your API key from a provider like Glassnode.

#### **Advanced Usage**

-   **Bayesian Optimization:** The script includes a full implementation of hyperparameter tuning with `Optuna`. By default, it is **commented out** to allow for fast baseline training. To enable it, uncomment the relevant block in the "Model Training & Hyperparameter Tuning" section of `train_model.py`.
-   **Model Interpretation:** The script automatically runs two forms of feature analysis after training:
    1.  **Feature Importance:** Logs the top 15 features based on the LightGBM model's internal importance score.
    2.  **SHAP Analysis:** Calculates and logs the top 15 features based on their mean absolute SHAP value, providing a more robust measure of feature impact. The script also contains a commented-out example for generating local, per-prediction SHAP explanations.
-   **Feature Selection:** The script automatically identifies and removes one feature from any pair with a correlation greater than 0.9, keeping the feature with higher importance.

### 3. Dependency Update
- **Package:** `pandas-ta` was found to be incompatible with the current `numpy` version.
- **Resolution:** It has been replaced with the community-maintained fork `pandas-ta-openbb` in the `Pipfile`. This fork is actively maintained and resolves the dependency issues. The code was updated to use `import pandas_ta as ta` as the fork maintains the original import name.

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
