import sys

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from ..entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from ..entity.config_entity import DataTransformationConfig
from ..utils.common import save_numpy_array_data, save_object
from ..utils.exception import PhishingDetectionException
from ..utils.logging import logger


class DataTransformation:
    """
    Class for transforming data after validation.
    """

    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        """
        Initializes the DataTransformation class.

        Args:
            data_validation_artifact (DataValidationArtifact): Artifact from the data validation stage.
            data_transformation_config (DataTransformationConfig): Configuration for data transformation.
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
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

    def get_data_transformer_object(self) -> Pipeline:
        """
        Initializes a KNNImputer object with the specified parameters and returns a Pipeline object.

        Returns:
            Pipeline: Pipeline object containing the KNNImputer.
        """
        try:
            logger.info("Initializing KNNImputer for data transformation.")
            imputer = KNNImputer(**self.data_transformation_config.imputer_params)
            processor = Pipeline([("imputer", imputer)])
            logger.info("KNNImputer initialized successfully.")
            return processor
        except Exception as e:
            raise PhishingDetectionException(f"Error initializing KNNImputer: {e}", sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation process.

        Returns:
            DataTransformationArtifact:
            Artifact containing paths to the transformed data and transformation object.
        """
        try:
            logger.info("Starting data transformation process.")

            # Read validated train and test data
            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Separate input features and target feature for training data
            input_feature_train_df = train_df.drop(
                columns=[self.data_transformation_config.target_column], axis=1
            )
            target_feature_train_df = train_df[
                self.data_transformation_config.target_column
            ]
            target_feature_train_df = target_feature_train_df.replace(
                -1, 0
            )  # Replace -1 with 0 for binary classification

            # Separate input features and target feature for testing data
            input_feature_test_df = test_df.drop(
                columns=[self.data_transformation_config.target_column], axis=1
            )
            target_feature_test_df = test_df[
                self.data_transformation_config.target_column
            ]
            target_feature_test_df = target_feature_test_df.replace(
                -1, 0
            )  # Replace -1 with 0 for binary classification

            # Get the data transformer object (KNNImputer)
            preprocessor = self.get_data_transformer_object()

            # Fit and transform the training data
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(
                input_feature_train_df
            )

            # Transform the testing data
            transformed_input_test_feature = preprocessor_object.transform(
                input_feature_test_df
            )

            # Combine transformed features with target features
            train_arr = np.c_[
                transformed_input_train_feature, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                transformed_input_test_feature, np.array(target_feature_test_df)
            ]

            # Save transformed data as numpy arrays
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )

            # Save the preprocessor object
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object,
            )

            # Save the preprocessor object to the saved_models directory
            save_object(
                self.data_transformation_config.saved_preprocessor_path,
                preprocessor_object,
            )

            # Prepare and return the data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logger.info("Data transformation process completed successfully.")
            return data_transformation_artifact

        except Exception as e:
            raise PhishingDetectionException(
                f"Error during data transformation: {e}", sys
            )
