import joblib
import pandas as pd
import os
import json
from typing import Dict, List, Any

MODEL_DIR = "src/ml/models"

def sanitize_symbol(symbol: str) -> str:
    """Converts a symbol like 'BTC/USDT' to 'BTC_USDT' for filenames."""
    return symbol.replace('/', '_')

class Predictor:
    """
    A class to load and manage trained models for multiple symbols
    and make predictions. Models are loaded on-demand and cached in memory.
    """
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.features: Dict[str, List[str]] = {}
        print("Predictor initialized. Models will be loaded on demand.")

    def _load_model(self, symbol: str) -> bool:
        """
        Loads the model and feature list for a specific symbol into the cache.
        Returns True if successful, False otherwise.
        """
        sanitized_symbol = sanitize_symbol(symbol)
        model_path = os.path.join(MODEL_DIR, f"lgbm_classifier_{sanitized_symbol}.joblib")
        features_path = os.path.join(MODEL_DIR, f"features_{sanitized_symbol}.json")

        if os.path.exists(model_path) and os.path.exists(features_path):
            print(f"Loading model for symbol '{symbol}' from disk...")
            try:
                self.models[symbol] = joblib.load(model_path)
                with open(features_path, 'r') as f:
                    self.features[symbol] = json.load(f)
                print(f"Model and features for '{symbol}' loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading model for '{symbol}': {e}")
                return False
        else:
            # This is not an error, just means a model for this symbol hasn't been trained.
            # print(f"Model file for symbol '{symbol}' not found. Cannot make predictions for this symbol.")
            return False

    def is_ready(self, symbol: str) -> bool:
        """
        Checks if a model for the given symbol is either already loaded in cache
        or is available to be loaded from disk.
        """
        if symbol in self.models:
            return True

        # Check if files exist on disk without loading them
        sanitized_symbol = sanitize_symbol(symbol)
        model_path = os.path.join(MODEL_DIR, f"lgbm_classifier_{sanitized_symbol}.joblib")
        return os.path.exists(model_path)


    def predict(self, features_df: pd.DataFrame, symbol: str) -> int:
        """
        Makes a prediction for a given symbol.
        It will load the symbol-specific model if it's not already cached.
        """
        # Load model if not already in cache
        if symbol not in self.models:
            if not self._load_model(symbol):
                return 0 # Return neutral prediction if model can't be loaded

        try:
            model = self.models[symbol]
            model_features = self.features[symbol]

            # Ensure the DataFrame has the correct columns in the correct order
            features_df_ordered = features_df[model_features]

            prediction = model.predict(features_df_ordered)
            return int(prediction[0])

        except KeyError as e:
            print(f"A required feature is missing from the input data for {symbol}: {e}")
            return 0
        except Exception as e:
            print(f"An error occurred during prediction for {symbol}: {e}")
            return 0
