import sys

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_training import ModelTraining
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainingConfig,
)
from src.utils.exception import PhishingDetectionException
from src.utils.logging import logger


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    # Initialize WandB if enabled
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
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

        # *****************************************************************************************

        # DATA TRANSFORMATION (PREPROCESSING OR FEATURE ENGINEERING)

        logger.info("Starting the data transformation process.")

        # Initialize DataTransformationConfig
        data_transformation_config = DataTransformationConfig()

        # Initialize DataTransformation
        data_transformation = DataTransformation(
            data_validation_artifact, data_transformation_config
        )

        # Run the data transformation process
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )

        logger.info("Data transformation process completed successfully.")
        logger.info(
            f"Transformed train file path: {data_transformation_artifact.transformed_train_file_path}"
        )
        logger.info(
            f"Transformed test file path: {data_transformation_artifact.transformed_test_file_path}"
        )
        logger.info(
            f"Transformed object file path: {data_transformation_artifact.transformed_object_file_path}"
        )

        print("Data transformation process completed successfully!")

        # *****************************************************************************************

        # MODEL TRAINING

        logger.info("Starting the model training process.")

        # Initialize ModelTrainingConfig
        model_training_config = ModelTrainingConfig()

        # Initialize ModelTrainer
        model_trainer = ModelTraining(
            cfg, model_training_config, data_transformation_artifact
        )

        # Run the model training process
        model_training_artifact = model_trainer.initiate_model_training()

        logger.info("Model training process completed successfully.")
        logger.info(
            f"Trained model file path: {model_training_artifact.trained_model_file_path}"
        )
        logger.info(
            f"Train metric artifact: {model_training_artifact.train_metric_artifact}"
        )
        logger.info(
            f"Test metric artifact: {model_training_artifact.test_metric_artifact}"
        )

        print("Model training process completed successfully!")

    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise PhishingDetectionException(str(e), sys)
    finally:
        # Finish wandb run
        if cfg.wandb.enabled:
            wandb.finish()
        logger.info("wandb run completed.")


if __name__ == "__main__":
    main()
