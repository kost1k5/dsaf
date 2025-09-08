import joblib
import pandas as pd
import os
import json
from typing import Dict, List, Any
import numpy as np

# Import global settings to access the timeframe
from src.core.config import settings
from src.core.utils import sanitize_symbol, parse_asset_from_symbol

# --- Constants for model directories ---
PROD_MODEL_DIR = "models/production"
OLD_MODEL_DIR = "src/ml/models"


class Predictor:
    """
    A class to load and manage trained models for multiple symbols
    and make predictions. It prioritizes loading new "production" models
    and falls back to old models if production ones are not found.
    """
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.features: Dict[str, List[str]] = {}
        print("Predictor initialized. Models will be loaded on demand.")

    def _load_model(self, symbol: str) -> bool:
        """
        Loads the model and feature list for a specific symbol into the cache.
        It first tries to load the new production model, then falls back to the old format.
        Returns True if successful, False otherwise.
        """
        # --- Attempt 1: Load New Production Model ---
        asset = parse_asset_from_symbol(symbol)
        timeframe = settings.TIMEFRAME
        prod_model_path = os.path.join(PROD_MODEL_DIR, f"prod_model_{asset}_{timeframe}.lgb")
        prod_features_path = os.path.join(PROD_MODEL_DIR, f"prod_features_{asset}_{timeframe}.json")

        if os.path.exists(prod_model_path) and os.path.exists(prod_features_path):
            print(f"Loading PRODUCTION model for symbol '{symbol}' from disk...")
            try:
                self.models[symbol] = joblib.load(prod_model_path)
                with open(prod_features_path, 'r') as f:
                    self.features[symbol] = json.load(f)
                print(f"Production model and features for '{symbol}' loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading PRODUCTION model for '{symbol}': {e}")
                # Don't return yet, allow fallback to old model

        # --- Attempt 2: Fallback to Old Model ---
        sanitized_symbol = sanitize_symbol(symbol)
        old_model_path = os.path.join(OLD_MODEL_DIR, f"lgbm_classifier_{sanitized_symbol}.joblib")
        old_features_path = os.path.join(OLD_MODEL_DIR, f"features_{sanitized_symbol}.json")

        if os.path.exists(old_model_path) and os.path.exists(old_features_path):
            print(f"Loading FALLBACK (old) model for symbol '{symbol}' from disk...")
            try:
                self.models[symbol] = joblib.load(old_model_path)
                with open(old_features_path, 'r') as f:
                    self.features[symbol] = json.load(f)
                print(f"Fallback model and features for '{symbol}' loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading FALLBACK model for '{symbol}': {e}")

        # If both attempts fail
        return False

    def is_ready(self, symbol: str) -> bool:
        """
        Checks if a model for the given symbol is either already loaded in cache
        or is available to be loaded from disk (checks both production and old paths).
        """
        if symbol in self.models:
            return True

        # Check for production model
        asset = parse_asset_from_symbol(symbol)
        timeframe = settings.TIMEFRAME
        prod_model_path = os.path.join(PROD_MODEL_DIR, f"prod_model_{asset}_{timeframe}.lgb")
        if os.path.exists(prod_model_path):
            return True

        # Check for old model
        sanitized_symbol = sanitize_symbol(symbol)
        old_model_path = os.path.join(OLD_MODEL_DIR, f"lgbm_classifier_{sanitized_symbol}.joblib")
        if os.path.exists(old_model_path):
            return True

        return False

    def predict(self, features_df: pd.DataFrame, symbol: str) -> int:
        """
        Makes a prediction for a given symbol.
        It will load the symbol-specific model if it's not already cached.
        The loaded prediction is for the multiclass model: 0=Sell, 1=Hold, 2=Buy.
        We will return this directly.
        """
        if symbol not in self.models:
            if not self._load_model(symbol):
                return 1 # Return neutral prediction (Hold) if model can't be loaded

        try:
            model = self.models[symbol]
            model_features = self.features[symbol]

            # Ensure the DataFrame has the correct columns in the correct order
            # and handle potential missing columns from the input df
            features_df_ordered = features_df.reindex(columns=model_features, fill_value=0)

            # The multiclass model returns probabilities for [Sell, Hold, Buy]
            prediction_proba = model.predict_proba(features_df_ordered)

            # Get the class with the highest probability
            prediction_class = np.argmax(prediction_proba, axis=1)

            # The prediction is an array, we need the first element.
            return int(prediction_class[0])

        except KeyError as e:
            print(f"A required feature is missing from the input data for {symbol}: {e}")
            return 1 # Hold
        except Exception as e:
            print(f"An error occurred during prediction for {symbol}: {e}")
            return 1 # Hold
