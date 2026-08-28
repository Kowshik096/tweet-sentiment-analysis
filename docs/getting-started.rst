Getting Started
===============

Install requirements and run the DVC pipeline:

.. code-block:: bash

   make requirements
   dvc repro

Serve the API locally:

.. code-block:: bash

   cd flask_app
   python app.py

Configuration
-------------

Set ``MLFLOW_TRACKING_URI`` to point at your MLflow server. The default
``sqlite:///mlflow.db`` creates a local MLflow store that supports the model
registry.
