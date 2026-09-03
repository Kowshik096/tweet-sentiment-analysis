import os
import sys

# Ensure project root is on path for src import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.model_evaluation import _flatten_params


def test_flatten_simple_params():
    params = {"a": 1, "b": "hello", "c": True}
    assert _flatten_params(params) == {"a": 1, "b": "hello", "c": True}


def test_flatten_nested_dict():
    params = {"data_ingestion": {"source_path": "data.csv", "test_size": 0.2}}
    flat = _flatten_params(params)
    assert flat == {"data_ingestion.source_path": "data.csv", "data_ingestion.test_size": 0.2}


def test_flatten_list_values():
    params = {"model_building": {"ngram_range": [1, 3]}}
    flat = _flatten_params(params)
    assert flat == {"model_building.ngram_range": "[1, 3]"}


def test_flatten_deeply_nested():
    params = {"a": {"b": {"c": {"d": 1}}}}
    flat = _flatten_params(params)
    assert flat == {"a.b.c.d": 1}


def test_flatten_all_scalars():
    """Every flattened value must be a scalar (str/int/float/bool), not list/dict."""
    params = {
        "data_ingestion": {"source_path": "x.csv", "test_size": 0.2},
        "model_building": {"ngram_range": [1, 3], "max_features": 10000},
    }
    flat = _flatten_params(params)
    for key, value in flat.items():
        assert not isinstance(value, (dict, list)), f"{key}={value!r} is not a scalar"
