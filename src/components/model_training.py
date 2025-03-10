import os
import sys

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from ..entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from ..entity.config_entity import ModelTrainingConfig
from ..utils.common import load_numpy_array_data, save_object
from ..utils.exception import PhishingDetectionException
from ..utils.logging import logger


class ModelTraining:
    """
    Class for training machine learning models.
    """

    def __init__(
        self,
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
            self.model_training_config = model_training_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise PhishingDetectionException(str(e), sys)

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params) -> dict:
        """
        Evaluates multiple models using GridSearchCV and returns the best model.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
            X_test (np.array): Testing features.
            y_test (np.array): Testing labels.
            models (dict): Dictionary of models to evaluate.
            params (dict): Dictionary of hyperparameters for each model.

        Returns:
            dict: Dictionary containing the best model and its score.
        """
        try:
            report = {}

            for model_name, model in models.items():
                logger.info(f"Training {model_name}...")
                gs = GridSearchCV(model, params[model_name], cv=3)
                gs.fit(X_train, y_train)

                # Set the best parameters and fit the model
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                # Evaluate the model
                y_test_pred = model.predict(X_test)
                test_model_score = f1_score(y_test, y_test_pred)

                report[model_name] = test_model_score

            return report
        except Exception as e:
            raise PhishingDetectionException(f"Error evaluating models: {e}", sys)

    def get_classification_score(self, y_true, y_pred) -> ClassificationMetricArtifact:
        """
        Calculates classification metrics (F1 score, precision, recall).

        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.

        Returns:
            ClassificationMetricArtifact: Artifact containing classification metrics.
        """
        try:
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            return ClassificationMetricArtifact(
                f1_score=f1, precision_score=precision, recall_score=recall
            )
        except Exception as e:
            raise PhishingDetectionException(
                f"Error calculating classification metrics: {e}", sys
            )

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
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            # Evaluate models and select the best one
            model_report = self.evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Calculate classification metrics
            y_train_pred = best_model.predict(X_train)
            train_metric_artifact = self.get_classification_score(y_train, y_train_pred)

            y_test_pred = best_model.predict(X_test)
            test_metric_artifact = self.get_classification_score(y_test, y_test_pred)

            # Save the trained model
            os.makedirs(
                os.path.dirname(self.model_training_config.trained_model_file_path),
                exist_ok=True,
            )
            save_object(self.model_training_config.trained_model_file_path, best_model)

            # Return the training artifact
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_training_config.trained_model_file_path,
                train_metric_artifact=train_metric_artifact,
                test_metric_artifact=test_metric_artifact,
            )
        except Exception as e:
            raise PhishingDetectionException(f"Error training model: {e}", sys)

    def initiate_model_training(self) -> ModelTrainerArtifact:
        """
        Initiates the model training process.

        Returns:
            ModelTrainerArtifact: Artifact containing the trained model and evaluation metrics.
        """
        try:
            logger.info("Starting model training process.")

            # Load transformed data
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # Split features and labels
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Train the model
            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            logger.info("Model training process completed successfully.")
            return model_trainer_artifact

        except Exception as e:
            raise PhishingDetectionException(f"Error during model training: {e}", sys)
