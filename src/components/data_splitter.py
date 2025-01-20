import os
import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Data Splitting Strategy
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Abstract method to split the data into training and testing sets.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        pass


# Concrete Strategy for Simple Train-Test Split
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initializes the SimpleTrainTestSplitStrategy with specific parameters.

        Parameters:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Splits the data into training and testing sets using a simple train-test split.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Performing simple train-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logging.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test


# Context Class for Data Splitting
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter with a specific data splitting strategy.

        Parameters:
        strategy (DataSplittingStrategy): The strategy to be used for data splitting.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Sets a new strategy for the DataSplitter.

        Parameters:
        strategy (DataSplittingStrategy): The new strategy to be used for data splitting.
        """
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str, save_dir="artifacts"):
        """
        Executes the data splitting using the current strategy and saves the data.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.
        save_dir (str): Directory to save the split data.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Splitting data using the selected strategy.")
        X_train, X_test, y_train, y_test = self._strategy.split_data(df, target_column)

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save splits as CSV files
        X_train.to_csv(os.path.join(save_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(save_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(save_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(save_dir, "y_test.csv"), index=False)

        logging.info(f"Data splits saved in directory: {save_dir}")

        return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    # Example dataframe (replace with actual data loading)
    df = pd.DataFrame({
        "Feature1": [1, 2, 3, 4, 5],
        "Feature2": [5, 4, 3, 2, 1],
        "Target": [0, 1, 0, 1, 0]
    })

    # Initialize data splitter with a specific strategy
    data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    X_train, X_test, y_train, y_test = data_splitter.split(df, target_column="Target")
