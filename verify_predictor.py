import os
import sys

# Add the project root to the python path, assuming script is in backend/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.ml.predictor import Predictor

def run_verification():
    print("--- Starting Predictor Verification ---")

    # The symbol the bot would use
    symbol_to_test = "BTC/USDT"

    # 1. Instantiate the predictor
    print("\n1. Instantiating Predictor...")
    try:
        predictor = Predictor()
        print("SUCCESS: Predictor instantiated.")
    except Exception as e:
        print(f"FAILURE: Could not instantiate Predictor. Error: {e}")
        return

    # 2. Check if the model is ready
    print(f"\n2. Checking if model is ready for symbol: {symbol_to_test}")
    is_ready = predictor.is_ready(symbol_to_test)
    if is_ready:
        print(f"SUCCESS: Predictor.is_ready() returned True.")
    else:
        print(f"FAILURE: Predictor.is_ready() returned False.")

    # 3. Attempt to load the model and get a prediction
    print(f"\n3. Calling predict() for symbol: {symbol_to_test}")
    # The predict method calls _load_model internally. We need a dummy dataframe.
    import pandas as pd
    # The dummy features must match what we put in the dummy json file
    dummy_df = pd.DataFrame([[0, 0]], columns=['feature1', 'feature2'])

    # We are checking the log output for "Loading PRODUCTION model"
    prediction = predictor.predict(dummy_df, symbol_to_test)
    print(f"\n4. Predictor returned: {prediction}")

    print("\n--- Verification Script Finished ---")
    print("Please check the logs above for the 'Loading PRODUCTION model' message.")

if __name__ == "__main__":
    run_verification()
