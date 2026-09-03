# src/data/data_preprocessing.py

import logging
import os

import pandas as pd

from src.preprocessing import preprocess_comment

# logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df["clean_comment"] = df["text"].apply(preprocess_comment)
        logger.debug("Text normalization completed")
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, "interim")
        logger.debug(f"Creating directory {interim_data_path}")

        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)

        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise


def main() -> None:
    try:
        logger.debug("Starting data preprocessing...")

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

        # Fetch the labelled data from data/labelled
        train_data = pd.read_csv(os.path.join(root_dir, "data/labelled/train_labelled.csv"))
        test_data = pd.read_csv(os.path.join(root_dir, "data/labelled/test_labelled.csv"))
        logger.debug("Data loaded successfully")

        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save the processed data
        save_data(
            train_processed_data, test_processed_data, data_path=os.path.join(root_dir, "data")
        )
    except Exception as e:
        logger.error("Failed to complete the data preprocessing process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
