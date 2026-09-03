FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY flask_app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl
COPY lgbm_model.pkl /app/lgbm_model.pkl
COPY src/ /app/src/

RUN python -m nltk.downloader stopwords wordnet vader_lexicon

EXPOSE 5000

CMD ["python", "app.py"]
