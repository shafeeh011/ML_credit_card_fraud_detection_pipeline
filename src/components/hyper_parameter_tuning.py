from src.logger import logging
from src.exception import CustomException
from abc import ABC, abstractmethod

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import pickle

# Abstract Base Class for Hyperparameter Tuning Strategy
class HyperparameterTuningStrategy(ABC):
    @abstractmethod
    def tune(self, model: BaseEstimator, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Abstract method to perform hyperparameter tuning.

        Parameters:
        model (BaseEstimator): The machine learning model to be tuned.
        param_grid (dict): Dictionary of hyperparameters to search.
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target variable.

        Returns:
        BaseEstimator: The model with the best parameters found.
        """
        pass

# Concrete Strategy for Grid Search
class GridSearchStrategy(HyperparameterTuningStrategy):
    def tune(self, model: BaseEstimator, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series):
        logging.info("Performing hyperparameter tuning using Grid Search.")
        try:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            logging.info(f"Best parameters found: {grid_search.best_params_}")
            return grid_search.best_estimator_
        except Exception as e:
            logging.error("Error occurred during Grid Search.")
            raise CustomException(e)

# Concrete Strategy for Randomized Search
class RandomSearchStrategy(HyperparameterTuningStrategy):
    def tune(self, model: BaseEstimator, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series):
        logging.info("Performing hyperparameter tuning using Randomized Search.")
        try:
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5, scoring='r2', n_jobs=-1, n_iter=50, random_state=42)
            random_search.fit(X_train, y_train)
            logging.info(f"Best parameters found: {random_search.best_params_}")
            return random_search.best_estimator_
        except Exception as e:
            logging.error("Error occurred during Randomized Search.")
            raise CustomException(e)

# Context Class for Hyperparameter Tuning
class HyperparameterTuner:
    def __init__(self, strategy: HyperparameterTuningStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: HyperparameterTuningStrategy):
        logging.info("Switching hyperparameter tuning strategy.")
        self._strategy = strategy

    def tune(self, model: BaseEstimator, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series):
        logging.info("Executing hyperparameter tuning strategy.")
        return self._strategy.tune(model, param_grid, X_train, y_train)

# Model Trainer Class
class ModelTrainer:
    def __init__(self, models: dict, params: dict):
        self.models = models
        self.params = params
        self.trained_model_path = "artifacts/model.pkl"

    def evaluate_models(self, X_train, y_train, X_test, y_test):
        logging.info("Evaluating models with hyperparameter tuning.")
        results = {}
        for name, model in self.models.items():
            try:
                logging.info(f"Tuning model: {name}")
                tuner = HyperparameterTuner(GridSearchStrategy())  # Default to Grid Search
                best_model = tuner.tune(model, self.params.get(name, {}), X_train, y_train)
                predictions = best_model.predict(X_test)
                r2 = r2_score(y_test, predictions)
                results[name] = (best_model, r2)
                logging.info(f"Model: {name}, R2 Score: {r2}")
            except Exception as e:
                logging.warning(f"Model {name} failed: {e}")
                results[name] = (None, -np.inf)
        return results

    def train_and_select_best_model(self, X_train, y_train, X_test, y_test):
        try:
            model_results = self.evaluate_models(X_train, y_train, X_test, y_test)
            best_model_name, (best_model, best_score) = max(model_results.items(), key=lambda item: item[1][1])

            if best_score < 0.6:
                raise CustomException("No suitable model found with R2 score above threshold.")

            logging.info(f"Best model: {best_model_name} with R2 score: {best_score}")

            # Save the best model
            with open(self.trained_model_path, 'wb') as f:
                pickle.dump(best_model, f)

            return best_model_name, best_score

        except Exception as e:
            logging.error("Error during model training and selection.")
            raise CustomException(e)

# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes

    # Load example dataset
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models and parameters
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }

    params = {
        "Decision Tree": {
            'criterion': ['squared_error', 'friedman_mse'],
            'max_depth': [None, 10, 20]
        },
        "Random Forest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        },
        "Linear Regression": {}
    }

    trainer = ModelTrainer(models, params)
    best_model_name, best_score = trainer.train_and_select_best_model(X_train, y_train, X_test, y_test)

    print(f"Best Model: {best_model_name} with R2 Score: {best_score}")
