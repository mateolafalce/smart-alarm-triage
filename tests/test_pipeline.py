import numpy as np
import pandas as pd
import pytest
from src.models.pipelines import build_random_forest_pipeline


def _make_data(n=200, n_features=20):
    X = pd.DataFrame(
        np.random.rand(n, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = np.random.randint(0, 5, n)
    return X, y


def test_rf_pipeline_fit_predict():
    X, y = _make_data()
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
    pipeline = build_random_forest_pipeline(params)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(range(5)))


def test_rf_pipeline_predict_proba():
    X, y = _make_data()
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
    pipeline = build_random_forest_pipeline(params)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 5)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
