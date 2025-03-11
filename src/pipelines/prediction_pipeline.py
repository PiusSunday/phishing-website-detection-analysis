import os
import sys

import pandas as pd

from ..constants import ARTIFACTS_DIR
from ..utils.common import load_object
from ..utils.exception import PhishingDetectionException
from ..utils.logging import logger


class PredictionPipeline:
    def __init__(self):
        self.preprocessor = load_object(
            os.path.join(ARTIFACTS_DIR, "saved_models", "preprocessor.pkl")
        )
        self.model = load_object(
            os.path.join(ARTIFACTS_DIR, "saved_models", "model.pkl")
        )
        self.predictions_dir = os.path.join(ARTIFACTS_DIR, "predictions")
        os.makedirs(self.predictions_dir, exist_ok=True)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Starting prediction process.")

            transformed_data = self.preprocessor.transform(data)
            predictions = self.model.predict(transformed_data)

            data["predicted_column"] = predictions

            logger.info("Prediction process completed successfully.")

            return data
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise PhishingDetectionException(str(e), sys)

    def save_prediction_output(
        self, data: pd.DataFrame, file_name: str = "prediction_output.csv"
    ):
        try:
            output_path = os.path.join(self.predictions_dir, file_name)

            data.to_csv(output_path, index=False)

            logger.info(f"Prediction output saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving prediction output: {e}")
            raise PhishingDetectionException(str(e), sys)
