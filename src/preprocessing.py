import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Module-level lemmatizer: WordNetLemmatizer is stateless, so a single
# instance is reused across calls instead of being rebuilt per comment.
_LEMMATIZER = WordNetLemmatizer()

# Stopwords retained because they carry sentiment signal for this task.
_STOP_WORDS = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}


def preprocess_comment(comment: str) -> str:
    """Apply preprocessing transformations to a tweet."""
    if not isinstance(comment, str):
        raise TypeError(f"preprocess_comment expects a string, got {type(comment).__name__}")

    # Convert to lowercase
    comment = comment.lower()

    # Remove URLs
    comment = re.sub(r"http\S+|www\S+", "", comment)

    # Remove @mentions
    comment = re.sub(r"@\w+", "", comment)

    # Keep hashtag words but drop the '#' symbol
    comment = re.sub(r"#", "", comment)

    # Remove trailing and leading whitespaces
    comment = comment.strip()

    # Remove newline characters
    comment = re.sub(r"\n", " ", comment)

    # Remove non-alphanumeric characters, except punctuation
    comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

    # Remove stopwords but retain important ones for sentiment analysis
    comment = " ".join([word for word in comment.split() if word not in _STOP_WORDS])

    # Lemmatize the words
    comment = " ".join([_LEMMATIZER.lemmatize(word) for word in comment.split()])

    return comment
