import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelConfig:
    """
    Dataclass to hold model configurations.
    """
    models: dict = field(default_factory=lambda: {
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier()
    })
    subset_size: int = 10000
    scoring: str = "accuracy"
    cv_folds: int = 3
    n_jobs: int = -1

class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        logging.info("ModelTrainer initialized with configuration: %s", self.config)

    def evaluate_models(self, X_train, y_train):
        """
        Evaluate models using cross-validation.

        Parameters:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.

        Returns:
        None: Prints the cross-validation scores for each model.
        """
        # Select subset of training data
        X_subset = X_train[:self.config.subset_size]
        y_subset = y_train[:self.config.subset_size]

        logging.info("Evaluating models on a subset of size: %d", self.config.subset_size)

        # Evaluate models
        for name, model in self.config.models.items():
            logging.info(f"Evaluating model: {name}")
            try:
                scores = cross_val_score(
                    model,
                    X_subset,
                    y_subset,
                    cv=self.config.cv_folds,
                    scoring=self.config.scoring,
                    n_jobs=self.config.n_jobs
                )
                logging.info(f"{name} Mean CV Accuracy: {scores.mean():.4f}")
            except Exception as e:
                logging.error(f"Error occurred while evaluating {name}: {e}")
                continue

if __name__ == "__main__":
    # Example usage
    try:
        # Load saved training data
        X_train = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_credit_card_fraud_detection_pipeline/data/data_splits/X_train.csv")
        y_train = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_credit_card_fraud_detection_pipeline/data/data_splits/y_train.csv")

        # Initialize ModelConfig and ModelTrainer
        config = ModelConfig(subset_size=10000, scoring="accuracy", cv_folds=3)
        trainer = ModelTrainer(config)

        # Evaluate models
        trainer.evaluate_models(X_train, y_train)

    except Exception as e:
        print(f"An error occurred: {e}")
