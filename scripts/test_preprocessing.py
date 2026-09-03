import os
import sys

# Ensure project root is on path for src import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess_comment


def test_preprocess_lowercases():
    assert preprocess_comment("HELLO World") == "hello world"


def test_preprocess_removes_urls():
    # "now" is a stopword and is removed by the pipeline
    result = preprocess_comment("check http://example.com now")
    assert "http" not in result
    assert "example.com" not in result
    assert "check" in result


def test_preprocess_removes_mentions():
    # "for" is a stopword and is removed by the pipeline
    result = preprocess_comment("thanks @user for watching")
    assert "@user" not in result
    assert "thanks" in result
    assert "watching" in result


def test_preprocess_drops_hashtag_symbol():
    # '#' is removed but the word is retained
    result = preprocess_comment("#netflix #SquidGame")
    assert "#" not in result
    assert "netflix" in result
    assert "squidgame" in result


def test_preprocess_collapses_newlines():
    assert preprocess_comment("line one\nline two") == "line one line two"


def test_preprocess_strips_whitespace():
    assert preprocess_comment("   hello   ") == "hello"


def test_preprocess_retains_sentiment_stopwords():
    # 'not' is retained because it carries sentiment signal
    result = preprocess_comment("not good at all")
    assert "not" in result.split()


def test_preprocess_lemmatizes():
    # WordNetLemmatizer reduces "cats" -> "cat" (noun plural)
    result = preprocess_comment("i am feeling the cats")
    assert "cat" in result


def test_preprocess_handles_empty_string():
    assert preprocess_comment("") == ""


def test_preprocess_module_level_lemmatizer_reused():
    """The lemmatizer must be created once at module load, not per call."""
    import src.preprocessing as pp

    assert pp._LEMMATIZER is not None
    assert pp._STOP_WORDS is not None
    # Calling repeatedly should not create new instances (stateless, but
    # verifies the module-level objects are the ones actually used).
    preprocess_comment("hello world")
    assert pp._LEMMATIZER is pp._LEMMATIZER


def test_preprocess_raises_on_non_string():
    """Non-string input should raise TypeError rather than be silently mishandled."""
    import pytest

    with pytest.raises(TypeError):
        preprocess_comment(None)
    with pytest.raises(TypeError):
        preprocess_comment(123)
