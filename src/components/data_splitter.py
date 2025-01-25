import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging  # Assuming you want to use this custom logger

@dataclass
class DataSplitter:
    df: pd.DataFrame
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    save_directory: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    save_dir: str = os.path.join(save_directory, "data/data_splits")

    def split_and_save(self):
        """
        Splits the dataset into training and testing sets and saves the splits as CSV files.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing sets.
        """
        logging.info("Starting data splitting process.")
        
        try:
            # Split data into features (X) and target (y)
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
            
            # Perform train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
            )
            logging.info("Data split into training and testing sets.")
            
            # Ensure save directory exists
            os.makedirs(self.save_dir, exist_ok=True)
            
            # Save splits as CSV files
            X_train.to_csv(os.path.join(self.save_dir, "X_train.csv"), index=False)
            X_test.to_csv(os.path.join(self.save_dir, "X_test.csv"), index=False)
            y_train.to_csv(os.path.join(self.save_dir, "y_train.csv"), index=False)
            y_test.to_csv(os.path.join(self.save_dir, "y_test.csv"), index=False)
            logging.info(f"Data splits saved in directory: {self.save_dir}")
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(f"An error occurred during data splitting: {e}")

# Example Usage
if __name__ == "__main__":
    # Load the example DataFrame
    df = pd.read_csv(
        "/home/muhammed-shafeeh/AI_ML/ML_credit_card_fraud_detection_pipeline/data/balanced_data/balanced_data.csv"
    )

    # Create an instance of DataSplitter
    splitter = DataSplitter(df=df, target_column="Class")
    
    # Perform data splitting and save the splits
    splitter.split_and_save()
