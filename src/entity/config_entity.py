import os
from dataclasses import dataclass, field

import numpy as np

from ..constants import (
    ARTIFACTS_DIR,
    COLLECTION_NAME,
    DATABASE_NAME,
    DRIFT_REPORT_FILE_NAME,
    FEATURE_STORE_FILE_NAME,
    INGESTED_DATA_DIR,
    INVALID_TEST_FILE_NAME,
    INVALID_TRAIN_FILE_NAME,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_EXPECTED_SCORE,
    MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_TRAINER_TRAINED_MODEL_NAME,
    RAW_DATA_DIR,
    SAVED_MODELS_DIR,
    SCHEMA_FILE_PATH,
    TARGET_COLUMN,
    TEST_FILE_NAME,
    TRAIN_FILE_NAME,
    TRANSFORMED_DATA_DIR,
    TRANSFORMED_DATA_OBJECTS_DIR,
    TRANSFORMED_OBJECT_FILE_NAME,
    TRANSFORMED_TEST_FILE_NAME,
    TRANSFORMED_TRAIN_FILE_NAME,
    VALIDATED_DATA_DIR,
    VALIDATED_DATA_OBJECTS_DIR,
    VALID_TEST_FILE_NAME,
    VALID_TRAIN_FILE_NAME,
)


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion.
    """

    raw_data_dir: str = RAW_DATA_DIR  # Directory to store raw data
    ingested_data_dir: str = INGESTED_DATA_DIR  # Directory to store ingested data
    database_name: str = DATABASE_NAME  # MongoDB database name
    collection_name: str = COLLECTION_NAME  # MongoDB collection name
    feature_store_file_path: str = os.path.join(
        INGESTED_DATA_DIR, FEATURE_STORE_FILE_NAME
    )  # Path to feature store
    training_file_path: str = os.path.join(
        INGESTED_DATA_DIR, TRAIN_FILE_NAME
    )  # Path to training data
    testing_file_path: str = os.path.join(
        INGESTED_DATA_DIR, TEST_FILE_NAME
    )  # Path to testing data
    train_test_split_ratio: float = 0.2  # Train-test split ratio


@dataclass
class DataValidationConfig:
    """
    Configuration for data validation.
    """

    schema_file_path: str = (
        SCHEMA_FILE_PATH  # Path to the schema file (e.g., JSON or YAML)
    )
    validated_data_dir: str = VALIDATED_DATA_DIR  # Directory to store validated data
    validated_object_dir: str = (
        VALIDATED_DATA_OBJECTS_DIR  # Directory to store validation objects
    )
    valid_train_file_path: str = os.path.join(
        VALIDATED_DATA_DIR, VALID_TRAIN_FILE_NAME
    )  # Path to valid training data
    valid_test_file_path: str = os.path.join(
        VALIDATED_DATA_DIR, VALID_TEST_FILE_NAME
    )  # Path to valid testing data
    invalid_train_file_path: str = os.path.join(
        VALIDATED_DATA_DIR, INVALID_TRAIN_FILE_NAME
    )  # Path to invalid training data
    invalid_test_file_path: str = os.path.join(
        VALIDATED_DATA_DIR, INVALID_TEST_FILE_NAME
    )  # Path to invalid testing data
    drift_report_file_path: str = os.path.join(
        VALIDATED_DATA_OBJECTS_DIR, DRIFT_REPORT_FILE_NAME
    )  # Path to drift report


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    """

    target_column: str = TARGET_COLUMN  # Name of the target column
    imputer_params: dict = field(
        default_factory=lambda: {
            "missing_values": np.nan,  # Replace NaN values
            "n_neighbors": 3,  # Number of neighbors to use
            "weights": "uniform",  # Weight function used in prediction
        }
    )
    transformed_data_dir: str = (
        TRANSFORMED_DATA_DIR  # Directory to store transformed data
    )
    transformed_object_dir: str = (
        TRANSFORMED_DATA_OBJECTS_DIR  # Directory to store transformation objects
    )
    transformed_train_file_path: str = os.path.join(
        TRANSFORMED_DATA_DIR, TRANSFORMED_TRAIN_FILE_NAME
    )  # Path to transformed training data
    transformed_test_file_path: str = os.path.join(
        TRANSFORMED_DATA_DIR, TRANSFORMED_TEST_FILE_NAME
    )  # Path to transformed testing data
    transformed_object_file_path: str = os.path.join(
        TRANSFORMED_DATA_OBJECTS_DIR, TRANSFORMED_OBJECT_FILE_NAME
    )  # Path to the transformation object
    saved_preprocessor_path: str = os.path.join(
        SAVED_MODELS_DIR, TRANSFORMED_OBJECT_FILE_NAME
    )  # Path to save preprocessor in saved_models


@dataclass
class ModelTrainingConfig:
    """
    Configuration for model training.
    """

    model_trainer_dir: str = os.path.join(
        ARTIFACTS_DIR, MODEL_TRAINER_DIR_NAME
    )  # Directory to store model training artifacts
    trained_model_dir: str = os.path.join(
        model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR
    )  # Directory to store trained models
    trained_model_file_path: str = os.path.join(
        trained_model_dir, MODEL_TRAINER_TRAINED_MODEL_NAME
    )  # Path to the trained model file
    saved_model_path: str = os.path.join(
        SAVED_MODELS_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME
    )  # Path to save the trained model in saved_models
    expected_accuracy: float = (
        MODEL_TRAINER_EXPECTED_SCORE  # Minimum-expected accuracy for the model
    )
    overfitting_underfitting_threshold: float = (
        MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD  # Threshold for overfitting/underfitting
    )
