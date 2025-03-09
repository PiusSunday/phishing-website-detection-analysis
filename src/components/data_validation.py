import os
import sys

import pandas as pd
from scipy.stats import ks_2samp

from ..entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from ..entity.config_entity import DataValidationConfig
from ..utils.common import read_yaml_file, write_yaml_file
from ..utils.exception import PhishingDetectionException
from ..utils.logging import logger


class DataValidation:
    """
    Class for validating data after ingestion.
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        """
        Initializes the DataValidation class.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Artifact from the data ingestion stage.
            data_validation_config (DataValidationConfig): Configuration for data validation.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(
                self.data_validation_config.schema_file_path
            )
        except Exception as e:
            raise PhishingDetectionException(str(e), sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file into a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the data.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise PhishingDetectionException(
                f"Error reading data from {file_path}: {e}", sys
            )

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates that the DataFrame has the correct number of columns as per the schema.

        Args:
            dataframe (pd.DataFrame): DataFrame to validate.

        Returns:
            bool: True if the number of columns matches the schema, False otherwise.
        """
        try:
            number_of_columns = len(self._schema_config["columns"])
            logger.info(f"Required number of columns: {number_of_columns}")
            logger.info(f"DataFrame has columns: {len(dataframe.columns)}")
            return len(dataframe.columns) == number_of_columns
        except Exception as e:
            raise PhishingDetectionException(
                f"Error validating number of columns: {e}", sys
            )

    def detect_dataset_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
    ) -> bool:
        """
        Detects dataset drift using the Kolmogorov-Smirnov test.

        Args:
            base_df (pd.DataFrame): Base DataFrame (e.g., training data).
            current_df (pd.DataFrame): Current DataFrame (e.g., testing data).
            threshold (float): Threshold for p-value to determine drift.

        Returns:
            bool: True if drift is detected, False otherwise.
        """
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update(
                    {
                        column: {
                            "p_value": float(is_same_dist.pvalue),
                            "drift_status": is_found,
                        }
                    }
                )

            # Save drift report
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status
        except Exception as e:
            raise PhishingDetectionException(f"Error detecting dataset drift: {e}", sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates the data validation process.

        Returns:
            DataValidationArtifact: Artifact containing validation results.
        """
        try:
            logger.info("Starting data validation process.")

            # Read train and test data from the data ingestion artifact
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            train_dataframe = self.read_data(train_file_path)
            test_dataframe = self.read_data(test_file_path)

            # Validate number of columns
            train_status = self.validate_number_of_columns(train_dataframe)
            test_status = self.validate_number_of_columns(test_dataframe)

            invalid_train_path = None
            invalid_test_path = None

            if not train_status:
                invalid_train_path = self.data_validation_config.invalid_train_file_path
                train_dataframe.to_csv(invalid_train_path, index=False, header=True)

            if not test_status:
                invalid_test_path = self.data_validation_config.invalid_test_file_path
                test_dataframe.to_csv(invalid_test_path, index=False, header=True)

            if not train_status or not test_status:
                error_message = (
                    "Train or test DataFrame does not contain all required columns."
                )
                logger.error(error_message)

            # Detect dataset drift
            drift_status = self.detect_dataset_drift(train_dataframe, test_dataframe)

            # Save validated data
            os.makedirs(
                os.path.dirname(self.data_validation_config.valid_train_file_path),
                exist_ok=True,
            )
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path,
                index=False,
                header=True,
            )
            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path,
                index=False,
                header=True,
            )

            # Create and return DataValidationArtifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=invalid_train_path,
                invalid_test_file_path=invalid_test_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logger.info("Data validation process completed successfully.")
            return data_validation_artifact

        except Exception as e:
            raise PhishingDetectionException(f"Error during data validation: {e}", sys)
