# src/data/data_labeling.py
"""Pseudo-label tweets with sentiment using NLTK VADER.

The raw Squid Game tweets dataset has no ground-truth sentiment labels.
This stage assigns weak-supervision labels: compound score >= positive_threshold
maps to category 1 (positive), <= negative_threshold maps to -1 (negative),
otherwise 0 (neutral).
"""

import os
import pandas as pd
import yaml
import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# logging configuration
logger = logging.getLogger('data_labeling')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('data_labeling_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download the VADER lexicon
nltk.download('vader_lexicon', quiet=True)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def label_data(df: pd.DataFrame, positive_threshold: float, negative_threshold: float) -> pd.DataFrame:
    """Assign sentiment categories from VADER compound scores."""
    try:
        sia = SentimentIntensityAnalyzer()
        df['sentiment_polarity'] = df['text'].apply(lambda text: sia.polarity_scores(str(text))['compound'])
        df['category'] = df['sentiment_polarity'].apply(
            lambda score: 1 if score >= positive_threshold else (-1 if score <= negative_threshold else 0)
        )
        logger.debug('VADER labeling completed. Distribution: %s',
                     df['category'].value_counts().to_dict())
        return df
    except Exception as e:
        logger.error('Error during VADER labeling: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the labelled train and test datasets."""
    try:
        labelled_data_path = os.path.join(data_path, 'labelled')
        os.makedirs(labelled_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(labelled_data_path, "train_labelled.csv"), index=False)
        test_data.to_csv(os.path.join(labelled_data_path, "test_labelled.csv"), index=False)

        logger.debug('Labelled data saved to %s', labelled_data_path)
    except Exception as e:
        logger.error('Error occurred while saving the labelled data: %s', e)
        raise


def main():
    try:
        logger.debug("Starting data labeling...")

        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        positive_threshold = params['data_labeling']['positive_threshold']
        negative_threshold = params['data_labeling']['negative_threshold']

        # Load the raw split data
        train_data = pd.read_csv(os.path.join(root_dir, 'data/raw/train.csv'))
        test_data = pd.read_csv(os.path.join(root_dir, 'data/raw/test.csv'))
        logger.debug('Data loaded successfully')

        # Label the data with VADER
        train_labelled = label_data(train_data, positive_threshold, negative_threshold)
        test_labelled = label_data(test_data, positive_threshold, negative_threshold)

        # Save the labelled data
        save_data(train_labelled, test_labelled, data_path=os.path.join(root_dir, 'data'))
    except Exception as e:
        logger.error('Failed to complete the data labeling process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
