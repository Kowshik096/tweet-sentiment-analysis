import pytest
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import os

# Set your tracking URI (env var override, local sqlite default)
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))

@pytest.mark.parametrize("model_name, stage, holdout_data_path, vectorizer_path", [
    ("tweet_sentiment_model", "staging", "data/interim/test_processed.csv", "tfidf_vectorizer.pkl"),
])
def test_model_performance(model_name, stage, holdout_data_path, vectorizer_path):
    try:
        # Load the model from MLflow
        client = mlflow.tracking.MlflowClient()
        latest_version_info = client.get_latest_versions(model_name, stages=[stage])
        latest_version = latest_version_info[0].version if latest_version_info else None

        assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"

        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load the vectorizer
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        # Load the holdout test data
        holdout_data = pd.read_csv(holdout_data_path)
        X_holdout_raw = holdout_data['clean_comment']  # Raw text features
        y_holdout = holdout_data['category']  # Labels

        # Handle NaN values in the text data
        X_holdout_raw = X_holdout_raw.fillna("")

        # Apply TF-IDF transformation
        X_holdout_tfidf = vectorizer.transform(X_holdout_raw)
        X_holdout_tfidf_df = pd.DataFrame(X_holdout_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

        # Make predictions
        y_pred = model.predict(X_holdout_tfidf_df)

        # Ensure predictions and true labels are aligned
        assert len(y_pred) == len(y_holdout), "Mismatch in prediction and label lengths"

        # Calculate performance metrics
        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred, average='weighted')
        recall = recall_score(y_holdout, y_pred, average='weighted')
        f1 = f1_score(y_holdout, y_pred, average='weighted')

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Assert minimum performance thresholds (VADER pseudo-labels are learnable, expect high agreement)
        assert accuracy >= 0.80, f"Accuracy below threshold: {accuracy}"
        assert precision >= 0.80, f"Precision below threshold: {precision}"
        assert recall >= 0.80, f"Recall below threshold: {recall}"
        assert f1 >= 0.80, f"F1 Score below threshold: {f1}"

    except Exception as e:
        pytest.fail(f"Model performance test failed with error: {e}")
