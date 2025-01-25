from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, load_object
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class LogisticRegressionModel:
    """
    Class to handle training, predicting, evaluating, and saving a Logistic Regression model.
    """

    def __init__(self, C=1.0, max_iter=100, penalty='l2', solver='lbfgs', random_state=None):
        """
        Initialize the Logistic Regression model with specified hyperparameters.
        """
        logging.info("Initializing Logistic Regression model...")
        self.model = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver, random_state=random_state)

    def train(self, X_train, Y_train):
        """
        Train the Logistic Regression model on the provided training data.

        Parameters:
        X_train (array-like): Feature matrix for training.
        Y_train (array-like): Target labels for training.

        Returns:
        None
        """
        logging.info("Training the Logistic Regression model...")
        self.model.fit(X_train, Y_train)
        logging.info("Model training complete.")

    def predict(self, X_test):
        """
        Make predictions using the trained model.

        Parameters:
        X_test (array-like): Feature matrix for testing.

        Returns:
        array-like: Predicted labels for the test data.
        """
        logging.info("Making predictions...")
        predictions = self.model.predict(X_test)
        logging.info("Predictions complete.")
        return predictions

    def evaluate(self, X_test, Y_test):
        """
        Evaluate the model's performance on the test set.

        Parameters:
        X_test (array-like): Feature matrix for testing.
        Y_test (array-like): True labels for testing.

        Returns:
        dict: Evaluation metrics, including accuracy and classification report.
        """
        logging.info("Evaluating the model...")
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(Y_test, y_pred)
        logging.info(f"Model Accuracy: {accuracy:.4f}")
        report = classification_report(Y_test, y_pred)
        logging.info("Evaluation complete.")
        return {
            'accuracy': accuracy,
            'classification_report': report
        }

    def save_model(self, file_path):
        """
        Save the trained model using the save_object utility.

        Parameters:
        file_path (str): Path to save the model file.

        Returns:
        None
        """
        logging.info(f"Saving the model to {file_path}...")
        save_object(file_path, self.model)
        logging.info("Model saved successfully.")

    @staticmethod
    def load_model(file_path):
        """
        Load a saved model using the load_object utility.

        Parameters:
        file_path (str): Path to the model file.

        Returns:
        LogisticRegression: Loaded Logistic Regression model.
        """
        logging.info(f"Loading the model from {file_path}...")
        model = load_object(file_path)
        logging.info("Model loaded successfully.")
        return model


# Example usage
if __name__ == "__main__":
    try:
        # Load the dataset
        X_train = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_credit_card_fraud_detection_pipeline/data/data_splits/X_train.csv")
        y_train = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_credit_card_fraud_detection_pipeline/data/data_splits/y_train.csv")
        X_test = pd.read_csv("/home/muhammed-shafeeh/AI_ML/ML_credit_card_fraud_detection_pipeline/data/data_splits/X_test.csv")
        y_test = pd.read_csv('/home/muhammed-shafeeh/AI_ML/ML_credit_card_fraud_detection_pipeline/data/data_splits/y_test.csv')

        # Initialize and train the model
        model = LogisticRegressionModel(C=10, max_iter=10000, penalty='l1', solver='liblinear')
        model.train(X_train, y_train)

        # Save the trained model
        model_file_path = "/home/muhammed-shafeeh/AI_ML/ML_credit_card_fraud_detection_pipeline/models/logistic_regression_model.pkl"
        model.save_model(model_file_path)

        # Evaluate the model
        results = model.evaluate(X_test, y_test)
        print("Accuracy:", results['accuracy'])
        print("Classification Report:\n", results['classification_report'])

        # Load the model for prediction
        loaded_model = LogisticRegressionModel.load_model(model_file_path)
        predictions = loaded_model.predict(X_test)
        print("Predictions on test data:\n", predictions)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
