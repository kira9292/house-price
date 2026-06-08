"""Unit tests for src/trainer.py"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.trainer import Trainer


def _make_sample_data():
    """Create a small synthetic dataset for testing."""
    np.random.seed(43)
    n = 100
    return pd.DataFrame({
        "area": np.random.uniform(50, 200, n),
        "rooms": np.random.randint(1, 6, n).astype(float),
        "neighborhood": np.random.choice(["A", "B", "C"], n),
        "price": np.random.uniform(100_000, 500_000, n),
    })


def _make_trainer(estimator=None):
    if estimator is None:
        estimator = DummyRegressor(strategy="mean")
    data = _make_sample_data()
    return Trainer(
        data=data,
        target="price",
        estimator=estimator,
        numerical_transformer=[SimpleImputer(strategy="median"), StandardScaler()],
        categorical_transformer=[
            SimpleImputer(strategy="constant", fill_value="undefined"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ],
        features=["area", "rooms", "neighborhood"],
        test_size=0.25,
        random_state=43,
    )


def test_define_pipeline_returns_pipeline():
    """define_pipeline() should build a non-null sklearn Pipeline."""
    from sklearn.pipeline import Pipeline

    trainer = _make_trainer()
    pipeline = trainer.define_pipeline()
    assert pipeline is not None
    assert isinstance(pipeline, Pipeline)


def test_fit_splits_data():
    """fit() should populate X_train, X_test, y_train, y_test."""
    trainer = _make_trainer()
    trainer.fit()
    assert trainer.X_train is not None
    assert trainer.X_test is not None
    assert trainer.y_train is not None
    assert trainer.y_test is not None
    assert len(trainer.X_train) + len(trainer.X_test) == 100


def test_train_returns_metrics():
    """train() should return (pipeline, train_metrics, test_metrics) with rmse/mae/r2."""
    trainer = _make_trainer()
    pipeline, train_metrics, test_metrics = trainer.train()

    for metrics in (train_metrics, test_metrics):
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert isinstance(metrics["rmse"], float)
        assert isinstance(metrics["mae"], float)
        assert isinstance(metrics["r2"], float)


def test_predict_returns_correct_shape():
    """predict() should return an array with the same length as the test set."""
    trainer = _make_trainer()
    trainer.fit()
    preds = trainer.predict()
    assert len(preds) == len(trainer.X_test)


def test_train_test_size_respected():
    """The test set should represent approximately 25% of the data."""
    trainer = _make_trainer()
    trainer.fit()
    expected_test = int(100 * 0.25)
    assert abs(len(trainer.X_test) - expected_test) <= 1


def test_score_returns_float():
    """score() should return a float R² value."""
    trainer = _make_trainer()
    trainer.fit()
    score = trainer.score()
    assert isinstance(score, float)
