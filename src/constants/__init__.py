import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Weights & Biases (wandb) configuration
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Logging configuration
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "phishing_website_detection_analysis.log")

# Data directories
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

# Processed data subdirectories
INGESTED_DATA_DIR = os.path.join(
    PROCESSED_DATA_DIR, "ingested"
)  # Data from ingestion stage
VALIDATED_DATA_DIR = os.path.join(
    PROCESSED_DATA_DIR, "validated"
)  # Data from validation stage
TRANSFORMED_DATA_DIR = os.path.join(
    PROCESSED_DATA_DIR, "transformed"
)  # Data from transformation stage

# Objects directories for validated and transformed data
VALIDATED_DATA_OBJECTS_DIR = os.path.join(VALIDATED_DATA_DIR, "objects")
TRANSFORMED_DATA_OBJECTS_DIR = os.path.join(TRANSFORMED_DATA_DIR, "objects")

# Data validation report file path
DATA_VALIDATION_REPORT_FILE_PATH = os.path.join(
    ROOT_DIR, "reports", "data_validation_report.yaml"
)

# Schema and model configuration paths
SCHEMA_FILE_PATH = os.path.join(ROOT_DIR, "src", "constants", "schema.yaml")
MODEL_CONFIG_FILE = os.path.join(ROOT_DIR, "src", "constants", "model_config.yaml")

# Database configuration
DATABASE_NAME = "phishing_website_detection_analysis_db"
COLLECTION_NAME = "phishing_data"

# Model configuration
BASE_ACCURACY = 0.85  # Minimum accuracy threshold for model acceptance

# Artifacts directory
ARTIFACTS_DIR_NAME = "artifacts"
ARTIFACTS_DIR = os.path.join(ROOT_DIR, ARTIFACTS_DIR_NAME)

# Target column name
TARGET_COLUMN = "Result"  # Name of the target column in the dataset

# File names for data ingestion
FEATURE_STORE_FILE_NAME = "feature_store.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

# File names for data validation
VALID_TRAIN_FILE_NAME = "train.csv"
VALID_TEST_FILE_NAME = "test.csv"
INVALID_TRAIN_FILE_NAME = "invalid_train.csv"
INVALID_TEST_FILE_NAME = "invalid_test.csv"
DRIFT_REPORT_FILE_NAME = "drift_report.yaml"

# File names for data transformation
TRANSFORMED_TRAIN_FILE_NAME = "train.npy"
TRANSFORMED_TEST_FILE_NAME = "test.npy"
TRANSFORMED_OBJECT_FILE_NAME = "preprocessor.pkl"

# Model training configuration
MODEL_TRAINER_DIR_NAME = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD = 0.05

# Saved models directory
SAVED_MODELS_DIR = os.path.join(ARTIFACTS_DIR, "saved_models")

# AWS S3 BUCKET NAME
TRAINING_BUCKET_NAME = "phishing-website-detection-analysis"

# Weights & Biases (WandB) configuration
WANDB_PROJECT_NAME = "Phishing-Website-Detection"  # Name of the wandb project
WANDB_ENTITY = "your-wandb-username"  # Your wandb username or team name
WANDB_API_KEY = (
    WANDB_API_KEY  # Your wandb API key (store securely in environment variables)
)
WANDB_LOG_MODEL = True  # Whether to log the trained model to wandb
WANDB_LOG_ARTIFACTS = (
    True  # Whether to log artifacts (e.g., datasets, reports) to wandb
)
