# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = "tweet-sentiment-analysis"
copyright = "2026, tweet-sentiment-analysis contributors"
author = "CampusX Replica"

# The short X.Y version
version = "0.0.1"
# The full version, including alpha/beta/rc tags
release = "0.0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"
