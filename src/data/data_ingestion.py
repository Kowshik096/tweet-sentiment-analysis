# src/data/data_ingestion.py

import logging
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


def load_data(data_path: str) -> pd.DataFrame:
    """Load the raw tweets dataset from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        logger.debug("Data loaded from %s", data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the tweets dataset: keep text/date, drop retweets, nulls,
    duplicates and empty strings."""
    try:
        # Keep only the columns needed for sentiment analysis
        df = df[["text", "date"]].copy()

        # Removing retweets (duplicated amplified content)
        if "is_retweet" in df.columns:
            df = df[~df["is_retweet"].astype(bool)]

        # Removing missing values
        df.dropna(subset=["text"], inplace=True)

        # Removing duplicates
        df.drop_duplicates(subset=["text"], inplace=True)

        # Removing rows with empty strings
        df = df[df["text"].str.strip() != ""]

        logger.debug(
            "Data preprocessing completed: retweets, missing values, "
            "duplicates, and empty strings removed."
        )
        return df
    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, "raw")

        # Create the data/raw directory if it does not exist
        os.makedirs(raw_data_path, exist_ok=True)

        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise


def main() -> None:
    try:
        # Load parameters from the params.yaml in the root directory
        params = load_params(
            params_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../../params.yaml"
            )
        )
        source_path = params["data_ingestion"]["source_path"]
        test_size = params["data_ingestion"]["test_size"]

        root_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
        )

        # Load data from the configured source path
        df = load_data(data_path=os.path.join(root_dir, source_path))

        # Clean the data
        final_df = preprocess_data(df)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Save the split datasets and create the raw folder if it doesn't exist
        save_data(train_data, test_data, data_path=os.path.join(root_dir, "data"))

    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
