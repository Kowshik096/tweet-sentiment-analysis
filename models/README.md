# Model Card: tweet_sentiment_model

## Model Details
- **Name**: tweet_sentiment_model
- **Version**: Registered in MLflow Model Registry (Staging → Production)
- **Type**: LightGBM multiclass classifier (3 classes: negative, neutral, positive)
- **Features**: TF-IDF (1–3 grams, max 10,000 features)
- **Training Data**: 75,181 VADER-labelled Squid Game tweets (60,144 train / 15,037 test)
- **Framework**: scikit-learn + LightGBM 4.5.0
- **License**: MIT

## Intended Use
- **Primary**: Real-time sentiment classification of Squid Game-related tweets
- **Secondary**: General short-text sentiment analysis (Twitter-like content)
- **Out of scope**: Long-form text, non-English, non-social-media domains

## Performance Metrics (Test Set)

| Metric | Score |
|--------|-------|
| **Accuracy** | **0.872** |
| **Weighted F1** | **0.871** |
| **Macro F1** | **0.858** |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| -1 (negative) | 0.803 | 0.783 | 0.793 | 3,023 |
| 0 (neutral) | 0.868 | 0.932 | 0.899 | 5,849 |
| 1 (positive) | 0.911 | 0.858 | 0.884 | 6,165 |

## Training Configuration

```yaml
model_building:
  ngram_range: [1, 3]
  max_features: 10000
  learning_rate: 0.09
  max_depth: 20
  n_estimators: 367
  reg_alpha: 0.1
  reg_lambda: 0.1
  is_unbalance: true
  class_weight: "balanced"
```

- **Imbalance handling**: `class_weight="balanced"` + `is_unbalance=True` (no resampling)
- **Regularization**: L1 (reg_alpha=0.1) + L2 (reg_lambda=0.1)
- **Objective**: multiclass, metric=multi_logloss

## Data & Labeling
- **Source**: `tweets_v8.csv` (80,019 raw tweets about Netflix Squid Game)
- **Labeling**: Weak supervision via VADER compound score
  - ≥ 0.05 → positive (1)
  - ≤ -0.05 → negative (-1)
  - else → neutral (0)
- **Note**: Labels are pseudo-labels (no human annotation)

## Limitations
1. **Weak labels**: VADER lexicon may misclassify sarcasm, slang, context-dependent sentiment
2. **Domain-specific**: Trained on Squid Game tweets; performance may degrade on other topics
3. **Class imbalance**: Negative class underrepresented (4% of data) → lower recall
4. **No temporal awareness**: Model doesn't capture sentiment evolution over time

## Ethical Considerations
- **Bias risk**: VADER lexicon has known demographic biases
- **Privacy**: Tweets are public but model could be used for surveillance
- **Misuse**: Should not be used for automated content moderation without human review

## Artifacts
- `lgbm_model.pkl` — trained LightGBM classifier
- `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer
- MLflow Model Registry: `models:/tweet_sentiment_model/Production`

## Reproduction
```bash
# Full pipeline (requires DVC + data/external/tweets_v8.csv)
dvc repro

# Or manual training
cd src/model
python model_building.py
python model_evaluation.py
python register_model.py
```

## API Usage
```bash
# Start server
cd flask_app && python app.py

# Predict
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"comments": ["Squid Game is amazing!", "Waste of time."]}'
```

## Files in This Directory
- `best_model_*.h5` — Deep learning checkpoints (notebooks 12-13, gitignored)
- `README.md` — This file

> **Note**: Production artifacts (`lgbm_model.pkl`, `tfidf_vectorizer.pkl`) are DVC-tracked at repo root, not stored in this folder.