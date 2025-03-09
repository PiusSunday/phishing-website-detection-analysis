import os
from dataclasses import dataclass

from ..constants import (
    ARTIFACTS_DIR,
    BASE_ACCURACY,
    COLLECTION_NAME,
    DATABASE_NAME,
    INGESTED_DATA_DIR,
    MODEL_CONFIG_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SCHEMA_FILE_PATH,
    TRANSFORMED_DATA_DIR,
    VALIDATED_DATA_DIR,
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
        INGESTED_DATA_DIR, "feature_store.csv"
    )  # Path to feature store
    training_file_path: str = os.path.join(
        INGESTED_DATA_DIR, "train.csv"
    )  # Path to training data
    testing_file_path: str = os.path.join(
        INGESTED_DATA_DIR, "test.csv"
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
    valid_train_file_path: str = os.path.join(
        VALIDATED_DATA_DIR, "train.csv"
    )  # Path to valid training data
    valid_test_file_path: str = os.path.join(
        VALIDATED_DATA_DIR, "test.csv"
    )  # Path to valid testing data
    invalid_train_file_path: str = os.path.join(
        VALIDATED_DATA_DIR, "invalid_train.csv"
    )  # Path to invalid training data
    invalid_test_file_path: str = os.path.join(
        VALIDATED_DATA_DIR, "invalid_test.csv"
    )  # Path to invalid testing data
    drift_report_file_path: str = os.path.join(
        VALIDATED_DATA_DIR, "drift_report.yaml"
    )  # Path to drift report


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    """

    transformed_data_dir: str = os.path.join(
        PROCESSED_DATA_DIR, "transformed"
    )  # Directory to store transformed data
    transformed_train_file_path: str = os.path.join(
        TRANSFORMED_DATA_DIR, "train.npy"
    )  # Path to transformed training data
    transformed_test_file_path: str = os.path.join(
        TRANSFORMED_DATA_DIR, "test.npy"
    )  # Path to transformed testing data
    transformed_object_file_path: str = os.path.join(
        TRANSFORMED_DATA_DIR, "preprocessing.pkl"
    )  # Path to the transformation object


@dataclass
class ModelTrainingConfig:
    """
    Configuration for model training.
    """

    trained_model_dir: str = os.path.join(
        ARTIFACTS_DIR, "trained_models"
    )  # Directory to save trained models
    base_accuracy: float = BASE_ACCURACY  # Minimum accuracy threshold
    model_config_file: str = (
        MODEL_CONFIG_FILE  # Path to model hyperparameters (e.g., JSON or YAML)
    )
