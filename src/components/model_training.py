import os
import re
import sys
import warnings

import dagshub
import mlflow
import wandb
from mlflow.models import infer_signature
from omegaconf import DictConfig
from scipy.optimize import OptimizeWarning

from ..entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from ..entity.config_entity import ModelTrainingConfig
from ..utils.common import load_numpy_array_data, save_object
from ..utils.exception import PhishingDetectionException
from ..utils.logging import logger
from ..utils.model_utils import (
    evaluate_models,
    get_base_models,
    get_classification_score,
    get_hyperparameter_grids,
)

dagshub.init(
    repo_owner="PiusSunday",
    repo_name="phishing-website-detection-analysis",
    mlflow=True,
)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarning
warnings.filterwarnings(
    "ignore", category=OptimizeWarning
)  # Suppress optimization warnings


class ModelTraining:
    """
    Class for training machine learning models.
    """

    def __init__(
        self,
        config: DictConfig,
        model_training_config: ModelTrainingConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        """
        Initializes the ModelTrainer class.

        Args:
            model_training_config (ModelTrainingConfig): Configuration for model training.
            data_transformation_artifact (DataTransformationArtifact): Artifact from the data transformation stage.
        """
        try:
            self.config = config
            self.model_training_config = model_training_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise PhishingDetectionException(str(e), sys)

    def train_model(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        """
        Trains the best model and returns the training artifact.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
            X_test (np.array): Testing features.
            y_test (np.array): Testing labels.

        Returns:
            ModelTrainerArtifact: Artifact containing the trained model and evaluation metrics.
        """
        try:
            models = get_base_models()
            params = get_hyperparameter_grids()

            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            y_train_pred = best_model.predict(X_train)
            train_metric_artifact = get_classification_score(y_train, y_train_pred)

            y_test_pred = best_model.predict(X_test)
            test_metric_artifact = get_classification_score(y_test, y_test_pred)

            os.makedirs(
                os.path.dirname(self.model_training_config.trained_model_file_path),
                exist_ok=True,
            )
            save_object(self.model_training_config.trained_model_file_path, best_model)

            # Pass parameters to log_mlflow
            self.log_mlflow(
                best_model,
                train_metric_artifact,
                test_metric_artifact,
                best_model_name,
                X_train,
            )

            # Log to wandb
            self.log_wandb(
                best_model,
                train_metric_artifact,
                test_metric_artifact,
                best_model_name,
                best_model_score,
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_training_config.trained_model_file_path,
                train_metric_artifact=train_metric_artifact,
                test_metric_artifact=test_metric_artifact,
            )
        except Exception as e:
            raise PhishingDetectionException(f"Error training model: {e}", sys)

    @staticmethod
    def log_mlflow(
        best_model,
        train_metric_artifact,
        test_metric_artifact,
        best_model_name,
        X_train,
    ):
        """Logs metrics, parameters, and model to MLFlow."""
        experiment_name = "Phishing_Detection_Model_Training"
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(best_model.get_params())

            # Log metrics
            mlflow.log_metric("train_f1", train_metric_artifact.f1_score)
            mlflow.log_metric("train_precision", train_metric_artifact.precision_score)
            mlflow.log_metric("train_recall", train_metric_artifact.recall_score)
            mlflow.log_metric("test_f1", test_metric_artifact.f1_score)
            mlflow.log_metric("test_precision", test_metric_artifact.precision_score)
            mlflow.log_metric("test_recall", test_metric_artifact.recall_score)

            # Log the best model name as a tag
            mlflow.set_tag("best_model", best_model_name)

            # Infer model signature and log input example
            signature = infer_signature(X_train, best_model.predict(X_train))
            input_example = X_train[:5]  # Log the first 5 rows as an input example

            # Log the model with signature and input example
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model",
                signature=signature,
                input_example=input_example,
            )

            mlflow.end_run()

    def log_wandb(
        self,
        best_model,
        train_metric_artifact,
        test_metric_artifact,
        best_model_name,
        best_model_score,
    ):
        """Logs metrics, parameters, and model to Weights & Biases (wandb)."""
        try:
            if self.config.wandb.enabled:
                # Log metrics
                wandb.log(
                    {
                        "train_f1": train_metric_artifact.f1_score,
                        "train_precision": train_metric_artifact.precision_score,
                        "train_recall": train_metric_artifact.recall_score,
                        "test_f1": test_metric_artifact.f1_score,
                        "test_precision": test_metric_artifact.precision_score,
                        "test_recall": test_metric_artifact.recall_score,
                        "best_model_name": best_model_name,
                        "best_model_score": best_model_score,
                    }
                )

                # Log model hyperparameters
                wandb.log({"best_model_params": best_model.get_params()})

                # Sanitize the model name for artifact naming
                sanitized_model_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", best_model_name)

                # Save the model as a wandb artifact
                model_artifact = wandb.Artifact(
                    name=f"{sanitized_model_name}_model",  # Use sanitized name
                    type="model",
                    description=f"Trained {best_model_name} model with score {best_model_score}",
                )
                model_artifact.add_file(
                    self.model_training_config.trained_model_file_path
                )
                wandb.log_artifact(model_artifact)

                logger.info("wandb logging completed successfully.")
        except Exception as e:
            logger.error(f"Error logging to wandb: {e}")
            raise PhishingDetectionException(f"Error logging to wandb: {e}", sys)

    def initiate_model_training(self) -> ModelTrainerArtifact:
        """
        Initiates the model training process.

        Returns:
            ModelTrainerArtifact: Artifact containing the trained model and evaluation metrics.
        """
        try:
            logger.info("Starting model training process.")
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            logger.info("Model training process completed successfully.")
            return model_trainer_artifact

        except Exception as e:
            raise PhishingDetectionException(f"Error during model training: {e}", sys)
