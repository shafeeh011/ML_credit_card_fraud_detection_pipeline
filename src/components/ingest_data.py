import os
import sys
import pandas as pd
import kaggle  # Import the Kaggle API
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException


# Define a DataIngestor dataclass
@dataclass
class KaggleDataIngestor:
    dataset_identifier: str

    def ingest(self) -> pd.DataFrame:
        """
        Downloads a Kaggle dataset and returns its content as a pandas DataFrame.
        :param dataset_identifier: Kaggle dataset identifier (e.g., 'username/dataset-name')
        :return: DataFrame containing the dataset.
        """
        try:
            # Initialize the Kaggle API
            kaggle.api.authenticate()

            # Download the dataset
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
            dataset_dir = os.path.join(project_root, "data/kaggle_data")
            kaggle.api.dataset_download_files(self.dataset_identifier, path=dataset_dir, unzip=True)

            # Find the CSV files
            extracted_files = os.listdir(dataset_dir)
            csv_files = [f for f in extracted_files if f.endswith(".csv")]

            if len(csv_files) == 0:
                raise FileNotFoundError("No CSV file found in the Kaggle dataset.")
            if len(csv_files) > 1:
                raise ValueError("Multiple CSV files found. Please specify which one to use.")

            csv_file_path = os.path.join(dataset_dir, csv_files[0])
            df = pd.read_csv(csv_file_path)

            logging.info("Kaggle dataset ingested successfully.")
            return df

        except Exception as e:
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    # Example for a Kaggle dataset
    dataset_identifier = "mlg-ulb/creditcardfraud"  # Replace with actual Kaggle dataset identifier
    data_ingestor = KaggleDataIngestor(dataset_identifier)
    df = data_ingestor.ingest()
    print(df.head())
