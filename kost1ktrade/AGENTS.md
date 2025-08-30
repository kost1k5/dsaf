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

## Future Roadmap: The "Nebula Command Bridge" Vision

This document outlines the grand strategic vision for the trading bot. It is divided into phases and sectors. Completed items should be removed or marked as complete.

### **Phase 1: The Simulation Deck Engine (In Progress)**
- **Goal:** Build the backend core required to power the "Simulation Deck" frontend.
- **Key Features:**
    - **1. Implement Backend for Simulation Deck:** Create a new `Backtester` module and a `POST /api/simulation/run` endpoint that can simultaneously backtest up to 10 strategies and return detailed performance reports.
    - **2. Expand Strategy Library:** Implement 5 new strategies (Stochastic, Awesome Oscillator, Parabolic SAR, Keltner Channels, Ichimoku Cloud) to provide a diverse set of "research probes".

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
