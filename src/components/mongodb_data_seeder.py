import json
import os
import sys

import pandas as pd
import pymongo
from dotenv import load_dotenv

from ..constants import COLLECTION_NAME, DATABASE_NAME, RAW_DATA_DIR
from ..utils.exception import PhishingDetectionException
from ..utils.logging import logger

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

if not MONGO_DB_URL:
    raise PhishingDetectionException(
        "MONGO_DB_URL environment variable is not set.", sys
    )


class MongoDBSeeder:
    """
    Class to load data from a CSV file into MongoDB.
    """

    def __init__(self):
        """
        Initializes the MongoDBSeeder class.
        """
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            logger.info("Connected to MongoDB Atlas successfully.")
        except pymongo.errors.ConnectionFailure as e:
            raise PhishingDetectionException(f"Failed to connect to MongoDB: {e}", sys)
        except Exception as e:
            raise PhishingDetectionException(
                f"An error occurred during MongoDB connection: {e}", sys
            )

    def csv_to_json_converter(self, file_path: str) -> list:
        """
        Converts a CSV file to a list of JSON records.
        """
        try:
            logger.info(f"Reading CSV file from: {file_path}")
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            logger.info(f"Successfully converted CSV to JSON. Records: {len(records)}")
            return records
        except FileNotFoundError as e:
            raise PhishingDetectionException(f"CSV file not found: {e}", sys)
        except Exception as e:
            raise PhishingDetectionException(f"Error converting CSV to JSON: {e}", sys)

    def insert_data_into_mongodb(
        self, records: list, database: str, collection: str
    ) -> int:
        """
        Inserts a list of JSON records into a MongoDB collection.
        """
        try:
            logger.info(
                f"Inserting data into MongoDB database: {database}, collection: {collection}"
            )
            db = self.mongo_client[database]
            col = db[collection]
            result = col.insert_many(records)
            logger.info(
                f"Successfully inserted {len(result.inserted_ids)} records into MongoDB."
            )
            return len(result.inserted_ids)
        except pymongo.errors.PyMongoError as e:
            raise PhishingDetectionException(f"MongoDB insertion error: {e}", sys)
        except Exception as e:
            raise PhishingDetectionException(
                f"Error inserting data into MongoDB: {e}", sys
            )


if __name__ == "__main__":
    try:
        FILE_PATH = os.path.join(RAW_DATA_DIR, "phishing-website-data.csv")
        DATABASE = DATABASE_NAME
        COLLECTION = COLLECTION_NAME

        data_seeder = MongoDBSeeder()
        records = data_seeder.csv_to_json_converter(file_path=FILE_PATH)
        no_of_records = data_seeder.insert_data_into_mongodb(
            records, DATABASE, COLLECTION
        )
        logger.info(f"Total records inserted: {no_of_records}")

    except PhishingDetectionException as e:
        logger.error(f"PhishingDetectionException: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


# **********************************************************************************************************************


# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi
#
# uri = "mongodb+srv://Sunnythesage:MongoAdmin11560@cluster-0.kqum1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-0"
#
# # Create a new client and connect to the server
# client = MongoClient(uri, server_api=ServerApi("1"))
#
# # Send a ping to confirm a successful connection
# try:
#     client.admin.command("ping")
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)
