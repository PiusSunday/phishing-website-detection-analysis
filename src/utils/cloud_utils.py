import subprocess
import sys

from .exception import PhishingDetectionException
from .logging import logger


class S3Sync:
    @staticmethod
    def sync_folder_to_s3(folder: str, aws_bucket_url: str):
        """Sync a local folder to an S3 bucket."""
        try:
            logger.info(f"Syncing folder {folder} to S3 at {aws_bucket_url}")

            # Use subprocess.run with shell=True and proper quoting
            command = f'aws s3 sync "{folder}" "{aws_bucket_url}"'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"S3 sync failed: {result.stderr}")
                raise PhishingDetectionException(
                    f"S3 sync failed: {result.stderr}", sys
                )

            logger.info(f"S3 sync completed: {result.stdout}")

        except Exception as e:
            raise PhishingDetectionException(str(e), sys)

    @staticmethod
    def sync_folder_from_s3(folder: str, aws_bucket_url: str):
        """Sync a folder from an S3 bucket to a local directory."""
        try:
            logger.info(f"Syncing folder from S3 at {aws_bucket_url} to {folder}")

            # Use subprocess.run with shell=True and proper quoting
            command = f'aws s3 sync "{aws_bucket_url}" "{folder}"'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"S3 sync failed: {result.stderr}")
                raise PhishingDetectionException(
                    f"S3 sync failed: {result.stderr}", sys
                )

            logger.info(f"S3 sync completed: {result.stdout}")

        except Exception as e:
            raise PhishingDetectionException(str(e), sys)
