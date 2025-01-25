import sys
import os
import numpy as np
import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    """
    Configuration for paths required in data transformation.
    """
    save_directory: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_folder: str = os.path.join(save_directory, "data/balanced_data")
    save_path: str = os.path.join(data_folder, 'balanced_data.csv')

    def __post_init__(self):
        # Ensure the "data/balanced_data" folder exists
        os.makedirs(self.data_folder, exist_ok=True)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def handle_imbalance_and_save(self, df: pd.DataFrame, target_column: str, sampling_strategy: float, save_path: str) -> pd.DataFrame:
        try:
            # Identify majority and minority classes
            class_counts = df[target_column].value_counts()
            max_class = class_counts.idxmax()
            min_class = class_counts.idxmin()

            logging.info(f"Majority class: {max_class} ({class_counts[max_class]} samples), "
                         f"Minority class: {min_class} ({class_counts[min_class]} samples).")

            # Split features and target variable
            df_features = df.drop(target_column, axis=1)
            df_target = df[target_column]

            # Apply SMOTE to oversample the minority class
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            synthetic_features, synthetic_target = smote.fit_resample(df_features, df_target)

            # Combine original and synthetic minority samples
            synthetic_df = pd.DataFrame(synthetic_features, columns=df_features.columns)
            synthetic_df[target_column] = synthetic_target

            # Separate the majority and minority data
            minority_df = synthetic_df[synthetic_df[target_column] == min_class]
            majority_df = df[df[target_column] == max_class]

            logging.info(f"Minority class size after SMOTE: {len(minority_df)} samples.")

            # Downsample the majority class to match the size of the minority class
            majority_downsampled = resample(
                majority_df, 
                replace=False, 
                n_samples=len(minority_df), 
                random_state=42
            )

            logging.info(f"Downsampling majority class to {len(majority_downsampled)} samples.")

            # Combine the downsampled majority class with the oversampled minority class
            balanced_df = pd.concat([majority_downsampled, minority_df]).sample(frac=1, random_state=42).reset_index(drop=True)

            # Save the balanced data
            balanced_df.to_csv(save_path, index=False)
            logging.info(f"Balanced data saved to {save_path}.")

            return balanced_df

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise CustomException(e)

# Example Usage
if __name__ == "__main__":
    try:
        # Initialize DataTransformation class
        data_transformation = DataTransformation()

        # Example DataFrame (replace with actual data)
        df = pd.read_csv('/home/muhammed-shafeeh/AI_ML/ML_credit_card_fraud_detection_pipeline/data/kaggle_data/creditcard.csv')

        # Handle imbalance and save the balanced data
        balanced_df = data_transformation.handle_imbalance_and_save(
            df=df, 
            target_column='Class', 
            sampling_strategy=0.1,  # Adjust as needed
            save_path=data_transformation.data_transformation_config.save_path
        )
    except Exception as e:
        print(f"Error: {e}")

