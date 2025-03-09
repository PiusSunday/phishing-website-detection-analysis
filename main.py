import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.utils.exception import PhishingDetectionException
from src.utils.logging import logger


def main():
    try:

        # DATA INGESTION

        logger.info("Starting the data ingestion process.")

        # Initialize DataIngestionConfig
        data_ingestion_config = DataIngestionConfig()

        # Initialize DataIngestion
        data_ingestion = DataIngestion(data_ingestion_config)

        # Run the data ingestion process
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        logger.info("Data ingestion process completed successfully.")
        logger.info(f"Train file path: {data_ingestion_artifact.trained_file_path}")
        logger.info(f"Test file path: {data_ingestion_artifact.test_file_path}")

        print("Data ingestion process completed successfully!")

        # *****************************************************************************************

        # DATA VALIDATION

        logger.info("Starting the data validation process.")

        # Initialize DataValidationConfig
        data_validation_config = DataValidationConfig()

        # Initialize DataValidation
        data_validation = DataValidation(
            data_ingestion_artifact, data_validation_config
        )

        # Run the data validation process
        data_validation_artifact = data_validation.initiate_data_validation()

        logger.info("Data validation process completed successfully.")
        logger.info(f"Validation status: {data_validation_artifact.validation_status}")
        logger.info(
            f"Valid train file path: {data_validation_artifact.valid_train_file_path}"
        )
        logger.info(
            f"Valid test file path: {data_validation_artifact.valid_test_file_path}"
        )
        logger.info(
            f"Invalid train file path: {data_validation_artifact.invalid_train_file_path}"
        )
        logger.info(
            f"Invalid test file path: {data_validation_artifact.invalid_test_file_path}"
        )
        logger.info(
            f"Drift report file path: {data_validation_artifact.drift_report_file_path}"
        )

        print("Data validation process completed successfully!")

    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise PhishingDetectionException(str(e), sys)


if __name__ == "__main__":
    main()
