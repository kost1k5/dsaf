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
