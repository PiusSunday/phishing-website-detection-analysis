import os
from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

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

# Schema and model configuration paths
SCHEMA_FILE_PATH = os.path.join(ROOT_DIR, "src", "constants", "schema.yaml")
MODEL_CONFIG_FILE = os.path.join(ROOT_DIR, "src", "constants", "model_config.yaml")

# Database configuration
DATABASE_NAME = "phishing_website_detection_analysis_db"
COLLECTION_NAME = "phishing_data"

# Model configuration
BASE_ACCURACY = 0.85  # Minimum accuracy threshold for model acceptance

# Logging configuration
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "phishing_website_detection_analysis.log")

# Artifacts directory
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
