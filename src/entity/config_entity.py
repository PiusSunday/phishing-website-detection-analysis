from dataclasses import dataclass
from datetime import datetime


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion.
    """

    raw_data_dir: str  # Directory to store raw data
    ingested_data_dir: str  # Directory to store processed data
    database_name: str  # MongoDB database name
    collection_name: str  # MongoDB collection name


@dataclass
class DataValidationConfig:
    """
    Configuration for data validation.
    """

    schema_file_path: str  # Path to the schema file (e.g., JSON or YAML)
    report_file_path: str  # Path to save validation reports


@dataclass
class ModelTrainingConfig:
    """
    Configuration for model training.
    """

    trained_model_dir: str  # Directory to save trained models
    base_accuracy: float  # Minimum accuracy threshold
    model_config_file: str  # Path to model hyperparameters (e.g., JSON or YAML)


@dataclass
class TrainingPipelineConfig:
    """
    Configuration for the entire training pipelines.
    """

    pipeline_name: str  # Name of the pipelines
    artifact_dir: str  # Directory to store pipelines artifacts
    timestamp: str = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )  # Timestamp for unique runs
