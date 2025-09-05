# Kost1kTrade: Full Setup and Launch Guide

This document provides a complete, step-by-step guide to set up the project, generate the trading model, and run the application.

---

### **Step 1: One-Time Project Setup**

This step only needs to be performed once.

1.  **Install Dependencies:**
    *   **Backend (Python):** Navigate to the `kost1ktrade/backend` directory and run:
        ```bash
        pipenv install
        ```
        *Note: If you encounter errors, you may need to manually install `matplotlib` by running `pipenv install matplotlib`.*

    *   **Frontend (JavaScript):** Navigate to the `kost1ktrade/frontend` directory and run:
        ```bash
        npm install
        ```

2.  **Configure Environment Variables:**
    *   In the `kost1ktrade/backend` directory, copy the `.env.example` file to a new file named `.env`.
    *   Open the `.env` file and fill in your API keys for the exchange and your Telegram Bot token.

3.  **Initialize the Database:**
    *   From the `kost1ktrade/backend` directory, run the following command to create the database tables:
        ```bash
        pipenv run python scripts/create_tables.py
        ```

---

### **Step 2: Generate the Production Trading Model**

This is the new core workflow. This process runs the entire quantitative analysis pipeline to produce the final, production-ready model that the live bot will use.

**You should run this script whenever you want to generate or update the trading models for all symbols configured in your `.env` file.**

This single command will execute the entire quantitative pipeline (data collection, feature engineering, labeling, training, and saving) for all assets.

From the `kost1ktrade/backend` directory, run:
```bash
pipenv run python scripts/run_full_pipeline.py
```

After the script finishes, new production-ready models will be available in the `kost1ktrade/backend/models/production/` directory. The bot will automatically find and use these new models the next time it makes a prediction.

---

### **Step 3: Run the Application**

Once the setup is done and the model is generated, you can run the application.

1.  **Navigate to the project's root directory** (the one that contains the `ecosystem.config.js` file).

2.  **Start the application using PM2 (Recommended):**
    ```bash
    pm2 start ecosystem.config.js
    ```

The trading bot and the web interface are now running. The bot will use the model you generated in Step 2.

*   **Web Interface:** `http://localhost:5173`
*   **To monitor logs:** `pm2 logs`
*   **To stop the application:** `pm2 stop all`

#### **Alternative: Manual/Development Launch**

You can also run the backend and frontend in separate terminals for development.

*   **Backend:** In `kost1ktrade/backend`, run `pipenv run uvicorn src.api.main:app --reload`
*   **Frontend:** In `kost1ktrade/frontend`, run `npm run dev`
