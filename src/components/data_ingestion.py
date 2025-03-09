import os
import sys

import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from ..entity.artifact_entity import DataIngestionArtifact
from ..entity.config_entity import DataIngestionConfig
from ..utils.exception import PhishingDetectionException
from ..utils.logging import logger

# Load environment variables
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    """
    Class for handling data ingestion from MongoDB, saving to feature store, and splitting into train/test sets.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion class.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration for data ingestion.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            logger.info("DataIngestion configuration loaded successfully.")
        except Exception as e:
            raise PhishingDetectionException(str(e), sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Exports data from MongoDB collection as a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the data from MongoDB.
        """
        try:
            logger.info("Exporting data from MongoDB collection as DataFrame.")
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            # Connect to MongoDB
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            # Fetch data and convert to DataFrame
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(
                    columns=["_id"], axis=1
                )  # Drop MongoDB's default `_id` column

            # Replace "na" with NaN
            df.replace({"na": np.nan}, inplace=True)
            logger.info("Successfully exported data from MongoDB.")
            return df

        except Exception as e:
            raise PhishingDetectionException(
                f"Error exporting data from MongoDB: {e}", sys
            )

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Saves the DataFrame to the feature store directory.

        Args:
            dataframe (pd.DataFrame): DataFrame to be saved.

        Returns:
            pd.DataFrame: The same DataFrame for further processing.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)

            # Save DataFrame to CSV
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logger.info(f"Data saved to feature store at: {feature_store_file_path}")
            return dataframe

        except Exception as e:
            raise PhishingDetectionException(
                f"Error saving data to feature store: {e}", sys
            )

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Splits the DataFrame into train and test sets and saves them to disk.

        Args:
            dataframe (pd.DataFrame): DataFrame to be split.
        """
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logger.info("Performed train-test split on the DataFrame.")

            # Save train and test sets
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logger.info(
                f"Train and test sets saved to: {self.data_ingestion_config.training_file_path} and {self.data_ingestion_config.testing_file_path}"
            )

        except Exception as e:
            raise PhishingDetectionException(
                f"Error splitting data into train/test sets: {e}", sys
            )

    def  initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process.

        Returns:
            DataIngestionArtifact: Artifact containing paths to the train and test files.
        """
        try:
            logger.info("Starting data ingestion process.")
            dataframe = self.export_collection_as_dataframe()
            self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            # Create and return DataIngestionArtifact
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logger.info("Data ingestion process completed successfully.")
            return data_ingestion_artifact

        except Exception as e:
            raise PhishingDetectionException(f"Error during data ingestion: {e}", sys)
