# app.py

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot

import io
import os
import sys

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from nltk.corpus import stopwords
from wordcloud import WordCloud

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import preprocess_comment

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load the model and vectorizer from the model registry and local storage
def get_tracking_uri():
    """MLflow tracking URI (env var override; local default is the sqlite DB at the repo root)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI") or None
    if uri:
        return uri
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return "sqlite:///" + os.path.join(root, "mlflow.db")


_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_APP_DIR, ".."))


def _local_artifact_path(filename: str, env_name: str) -> str:
    """Resolve a local artifact path.

    Preference order:
      1. an explicit env override (e.g. LOCAL_MODEL_PATH);
      2. the repo root -- the local development layout where the DVC outputs
         live next to ``mlflow.db`` (``<repo>/lgbm_model.pkl``);
      3. the app directory -- the Docker image layout where the artifacts are
         COPY'd next to ``app.py`` (``/app/lgbm_model.pkl``).

    If nothing exists, the repo-root path is returned so that ``joblib.load``
    raises a clear ``FileNotFoundError`` rather than a silent wrong-file load.
    """
    override = os.environ.get(env_name)
    if override:
        return override
    for candidate in (os.path.join(_REPO_ROOT, filename), os.path.join(_APP_DIR, filename)):
        if os.path.exists(candidate):
            return candidate
    return os.path.join(_REPO_ROOT, filename)


# Local fallback artifacts used when the MLflow registry is unavailable
# (e.g. inside the Docker image, where mlflow.db is not shipped).
LOCAL_MODEL_PATH = _local_artifact_path("lgbm_model.pkl", "LOCAL_MODEL_PATH")
LOCAL_VECTORIZER_PATH = _local_artifact_path("tfidf_vectorizer.pkl", "LOCAL_VECTORIZER_PATH")


def load_model_and_vectorizer(model_name, model_stage_or_version, vectorizer_path):
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(get_tracking_uri())
    model_uri = f"models:/{model_name}/{model_stage_or_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
    return model, vectorizer


def load_model_and_vectorizer_local():
    """Fallback: load the model and vectorizer directly from local pickle files.

    Used when the MLflow Model Registry is not reachable (e.g. the Docker image
    does not ship mlflow.db). The local artifacts must match the registered
    model's preprocessing (same TF-IDF vocabulary) for predictions to be valid.
    """
    model = joblib.load(LOCAL_MODEL_PATH)
    vectorizer = joblib.load(LOCAL_VECTORIZER_PATH)
    return model, vectorizer


# Lazy-load model and vectorizer on first request
_model = None
_vectorizer = None


def get_model_and_vectorizer():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        try:
            _model, _vectorizer = load_model_and_vectorizer(
                "tweet_sentiment_model",
                os.environ.get("MODEL_STAGE", "Production"),
                "./tfidf_vectorizer.pkl",
            )
        except Exception as e:
            app.logger.warning(
                "MLflow registry unavailable (%s); falling back to local artifacts.", e
            )
            _model, _vectorizer = load_model_and_vectorizer_local()
    return _model, _vectorizer


@app.route("/")
def home():
    return "Welcome to our flask api"


@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get("comments")

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    if not isinstance(comments_data, list):
        return jsonify({"error": "'comments' must be a list of objects"}), 400

    try:
        model, vectorizer = get_model_and_vectorizer()

        comments = []
        timestamps = []
        for item in comments_data:
            if not isinstance(item, dict) or "text" not in item:
                return (
                    jsonify({"error": "Each comment must be an object with a 'text' field"}),
                    400,
                )
            comments.append(item["text"])
            timestamps.append(item.get("timestamp"))

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # The registered model expects a dense DataFrame input (see its signature)
        transformed_df = pd.DataFrame(
            transformed_comments.toarray(), columns=vectorizer.get_feature_names_out()
        )

        # Make predictions
        predictions = model.predict(transformed_df).tolist()  # Convert to list

        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Return the response with original comments, predicted sentiments, and timestamps
    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    comments = data.get("comments")

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    if not isinstance(comments, list):
        return jsonify({"error": "'comments' must be a list of strings"}), 400

    try:
        model, vectorizer = get_model_and_vectorizer()

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # The registered model expects a dense DataFrame input (see its signature)
        transformed_df = pd.DataFrame(
            transformed_comments.toarray(), columns=vectorizer.get_feature_names_out()
        )

        # Make predictions
        predictions = model.predict(transformed_df).tolist()  # Convert to list

        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Return the response with original comments and predicted sentiments
    response = [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(comments, predictions)
    ]
    return jsonify(response)


@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get("sentiment_counts")

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "w"},
        )
        plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get("comments")

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = " ".join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")),
            collocations=False,
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get("sentiment_data")

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Set the timestamp as the index
        df.set_index("timestamp", inplace=True)

        # Ensure the 'sentiment' column is numeric
        df["sentiment"] = df["sentiment"].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample("ME")["sentiment"].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: "red",  # Negative sentiment
            0: "gray",  # Neutral sentiment
            1: "green",  # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker="o",
                linestyle="-",
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
