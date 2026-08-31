import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_comment(comment):
    """Apply preprocessing transformations to a tweet."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove URLs
        comment = re.sub(r'http\S+|www\S+', '', comment)

        # Remove @mentions
        comment = re.sub(r'@\w+', '', comment)

        # Keep hashtag words but drop the '#' symbol
        comment = re.sub(r'#', '', comment)

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment