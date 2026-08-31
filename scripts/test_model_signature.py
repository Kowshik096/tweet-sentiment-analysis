import mlflow.pyfunc
import pytest
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import os

# Set your tracking URI (env var override, local sqlite default)
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))


def _signature_to_dict(sig):
    """Convert MLflow ModelSignature to a comparable dict."""
    if sig is None:
        return None
    return {
        "inputs": sig.inputs.to_dict() if sig.inputs else None,
        "outputs": sig.outputs.to_dict() if sig.outputs else None,
    }


@pytest.mark.parametrize("model_name, stage, test_data_path, vectorizer_path", [
    ("tweet_sentiment_model", "staging", "data/interim/test_processed.csv", "tfidf_vectorizer.pkl"),
])
def test_model_signature(model_name, stage, test_data_path, vectorizer_path):
    client = MlflowClient()

    # Get the latest version in the specified stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"

    # Load the latest version of the model
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load the test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data['clean_comment'].fillna('')

    # Load the vectorizer and transform the data
    import pickle
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    X_test_tfidf = vectorizer.transform(X_test)

    # Create a DataFrame for the signature (dense representation)
    input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())

    # Infer the signature (pyfunc models expect the DataFrame input format)
    predictions = model.predict(input_example)
    signature = infer_signature(input_example, predictions)

    # Compare the inferred signature with the model's expected signature
    expected_signature = mlflow.models.get_model_info(model_uri).signature

    assert _signature_to_dict(signature) == _signature_to_dict(expected_signature), \
        f"Model signature mismatch:\n  got:      {_signature_to_dict(signature)}\n  expected: {_signature_to_dict(expected_signature)}"
    print("Model signature test passed.")
