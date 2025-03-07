import os
from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directory
DATA_ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Data paths
RAW_DATA_DIR = os.path.join(DATA_ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_ROOT_DIR, "data", "processed")
SCHEMA_FILE_PATH = os.path.join(DATA_ROOT_DIR, "src", "constants", "schema.yaml")

# Database configuration
DATABASE_NAME = "phishing_website_detection_analysis_db"
COLLECTION_NAME = "phishing_data"

# Model configuration
MODEL_CONFIG_FILE = os.path.join(ROOT_DIR, "src", "constants", "model_config.yaml")
BASE_ACCURACY = 0.85  # Minimum accuracy threshold for model acceptance

# Logging configuration
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "phishing_website_detection_analysis.log")

# Artifacts directory
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
