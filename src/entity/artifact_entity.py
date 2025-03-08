from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """
    Artifact for data ingestion stage.
    Contains paths to the train and test files.
    """

    trained_file_path: str  # Path to the training dataset file
    test_file_path: str  # Path to the testing dataset file


@dataclass
class DataValidationArtifact:
    """
    Artifact for data validation stage.
    Contains validation status and file paths for valid/invalid data.
    """

    validation_status: bool  # Whether the data validation was successful
    valid_train_file_path: str  # Path to the valid training dataset file
    valid_test_file_path: str  # Path to the valid testing dataset file
    invalid_train_file_path: str  # Path to the invalid training dataset file
    invalid_test_file_path: str  # Path to the invalid testing dataset file
    drift_report_file_path: str  # Path to the data drift report file


@dataclass
class DataTransformationArtifact:
    """
    Artifact for data transformation stage.
    Contains paths to the transformed data and transformation objects.
    """

    transformed_object_file_path: (
        str  # Path to the transformation object (e.g., preprocessing object)
    )
    transformed_train_file_path: str  # Path to the transformed training dataset file
    transformed_test_file_path: str  # Path to the transformed testing dataset file


@dataclass
class ClassificationMetricArtifact:
    """
    Artifact for classification metrics.
    Contains F1 score, precision, and recall.
    """

    f1_score: float  # F1 score
    precision_score: float  # Precision score
    recall_score: float  # Recall score


@dataclass
class ModelTrainerArtifact:
    """
    Artifact for model training stage.
    Contains paths to the trained model and evaluation metrics.
    """

    trained_model_file_path: str  # Path to the trained model file
    train_metric_artifact: (
        ClassificationMetricArtifact  # Metrics for the training dataset
    )
    test_metric_artifact: (
        ClassificationMetricArtifact  # Metrics for the testing dataset
    )
