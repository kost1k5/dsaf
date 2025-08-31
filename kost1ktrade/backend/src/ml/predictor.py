import joblib
import pandas as pd
import os
import json

class Predictor:
    """
    A class to load a trained model and make predictions.
    """
    def __init__(self, model_path="src/ml/models/lgbm_classifier.joblib", features_path="src/ml/models/features.json"):
        self.model = None
        self.features = None

        if os.path.exists(model_path) and os.path.exists(features_path):
            print("Loading trained model and feature list...")
            self.model = joblib.load(model_path)
            with open(features_path, 'r') as f:
                self.features = json.load(f)
            print("Model and features loaded successfully.")
        else:
            print("WARNING: Model file or features file not found. Predictor will not be able to make predictions.")

    def is_ready(self) -> bool:
        """
        Checks if the model and features are loaded.
        """
        return self.model is not None and self.features is not None

    def predict(self, features_df: pd.DataFrame) -> int:
        """
        Makes a prediction on a single row of features.

        :param features_df: A DataFrame containing the features for a single time step.
                            Must include all columns that the model was trained on.
        :return: The predicted class (-1 for Down, 0 for Sideways, 1 for Up).
        """
        if not self.is_ready():
            print("ERROR: Predictor is not ready. Cannot make a prediction.")
            return 0 # Default to a neutral prediction

        try:
            # Ensure the DataFrame has the correct columns in the correct order
            features_df = features_df[self.features]

            # Predict
            prediction = self.model.predict(features_df)

            # predict() returns an array, get the first element
            return int(prediction[0])

        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return 0 # Return neutral prediction on error
