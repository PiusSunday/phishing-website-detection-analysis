import sys
import warnings
from scipy.optimize import OptimizeWarning

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from ..entity.artifact_entity import ClassificationMetricArtifact
from ..utils.exception import PhishingDetectionException
from ..utils.logging import logger

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarning
warnings.filterwarnings("ignore", category=OptimizeWarning)  # Suppress optimization warnings


def evaluate_models(X_train, y_train, X_test, y_test, models, params) -> dict:
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
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = f1_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise PhishingDetectionException(f"Error evaluating models: {e}", sys)


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
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


def get_base_models():
    """Returns a dictionary of base models."""
    return {
        "Random Forest": RandomForestClassifier(verbose=1),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(verbose=1),
        "Logistic Regression": LogisticRegression(verbose=1),
        "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),  # Use SAMME instead of SAMME.R
    }


def get_hyperparameter_grids():
    """Returns a dictionary of hyperparameter grids for each model."""
    return {
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
            "algorithm": ["SAMME"],  # Explicitly set algorithm to SAMME
        },
    }