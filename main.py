import sys

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.utils.exception import PhishingDetectionException
from src.utils.logging import logger


def main():
    try:
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

    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise PhishingDetectionException(str(e), sys)


if __name__ == "__main__":
    main()
