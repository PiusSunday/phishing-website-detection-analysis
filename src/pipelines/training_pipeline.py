import sys
from datetime import datetime

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from ..components.data_ingestion import DataIngestion
from ..components.data_transformation import DataTransformation
from ..components.data_validation import DataValidation
from ..components.model_training import ModelTraining
from ..constants import PROCESSED_DATA_DIR, SAVED_MODELS_DIR, TRAINING_BUCKET_NAME
from ..entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
)
from ..entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainingConfig,
)
from ..utils.cloud_utils import S3Sync
from ..utils.exception import PhishingDetectionException
from ..utils.logging import logger


class TrainingPipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_training_config = ModelTrainingConfig()
        self.s3_sync = S3Sync()
        self.timestamp = datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S"
        )  # Unique timestamp for each run

        # Initialize WandB if enabled
        if self.cfg.wandb.enabled:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                tags=self.cfg.wandb.tags,
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion process.")
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info("Data ingestion completed successfully.")
            return data_ingestion_artifact
        except Exception as e:
            raise PhishingDetectionException(str(e), sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation process.")
            data_validation = DataValidation(
                data_ingestion_artifact, self.data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info("Data validation completed successfully.")
            return data_validation_artifact
        except Exception as e:
            raise PhishingDetectionException(str(e), sys)

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation process.")
            data_transformation = DataTransformation(
                data_validation_artifact, self.data_transformation_config
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            logger.info("Data transformation completed successfully.")
            return data_transformation_artifact
        except Exception as e:
            raise PhishingDetectionException(str(e), sys)

    def start_model_training(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model training process.")
            model_trainer = ModelTraining(
                self.cfg, self.model_training_config, data_transformation_artifact
            )
            model_training_artifact = model_trainer.initiate_model_training()
            logger.info("Model training completed successfully.")
            return model_training_artifact
        except Exception as e:
            raise PhishingDetectionException(str(e), sys)

    def sync_artifacts_to_s3(self):
        """Sync all artifacts (processed data and saved models) to S3."""
        try:
            logger.info("Syncing artifacts to S3.")

            # Sync processed data (from data/processed)
            processed_data_s3_path = (
                f"s3://{TRAINING_BUCKET_NAME}/processed/{self.timestamp}"
            )
            self.s3_sync.sync_folder_to_s3(
                folder=str(PROCESSED_DATA_DIR),  # Convert the path object to string
                aws_bucket_url=processed_data_s3_path,
            )
            logger.info(f"Processed data synced to S3 at {processed_data_s3_path}.")

            # Sync saved models (from artifacts/saved_models)
            saved_models_s3_path = (
                f"s3://{TRAINING_BUCKET_NAME}/saved_models/{self.timestamp}"
            )
            self.s3_sync.sync_folder_to_s3(
                folder=str(SAVED_MODELS_DIR),  # Convert the path object to string
                aws_bucket_url=saved_models_s3_path,
            )
            logger.info(f"Saved models synced to S3 at {saved_models_s3_path}.")
        except Exception as e:
            raise PhishingDetectionException(str(e), sys)

    def run_pipeline(self):
        try:
            logger.info("Starting training pipeline.")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact
            )
            model_training_artifact = self.start_model_training(
                data_transformation_artifact
            )

            # Sync artifacts to S3
            self.sync_artifacts_to_s3()

            logger.info("Training pipeline completed successfully.")
            return model_training_artifact
        except Exception as e:
            raise PhishingDetectionException(str(e), sys)
        finally:
            if self.cfg.wandb.enabled:
                wandb.finish()
                logger.info("wandb run completed.")


@hydra.main(config_path="../../conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    try:
        training_pipeline = TrainingPipeline(cfg)
        training_pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise PhishingDetectionException(str(e), sys)


if __name__ == "__main__":
    main()
