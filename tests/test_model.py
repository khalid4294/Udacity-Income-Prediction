"""
test model.py
"""
import pytest
from sklearn.linear_model import LogisticRegression
from starter.ml.model import (
    train_model,
    compute_model_metrics,
    inference,
)


@pytest.fixture
def test_model():
    return LogisticRegression()


@pytest.fixture
def test_trained_model():
    import joblib

    return joblib.load("./model/linear_regression_model.pkl")


@pytest.fixture
def X_train():
    return [[1, 2], [3, 4]]


@pytest.fixture
def y_train():
    return [0, 1]


@pytest.fixture
def X_test():
    # 109 features as input.
    import numpy as np

    return np.random.rand(1, 108)


@pytest.fixture
def y_test():
    return [0, 1]


@pytest.fixture
def preds():
    return [0, 1]


def test_train_model(X_train, y_train, test_model):
    main_model = train_model(X_train, y_train)
    assert main_model.__class__ == test_model.__class__


def test_compute_model_metrics(y_test, preds):
    precision, recall, fbeta, accuracy = compute_model_metrics(y_test, preds)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
    assert accuracy == 1.0


def test_inference(test_trained_model, X_test):
    preds = inference(test_trained_model, X_test)
    print(preds)
    assert preds in [0, 1]
