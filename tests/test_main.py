from fastapi.testclient import TestClient
from main import app
import numpy as np
import requests
import pytest

client = TestClient(app)


@pytest.fixture
def test_data_negative():
    data = {
        "dict_features": [
            {
                "age": "32",
                "workclass": "Private",
                "fnlgt": "205019",
                "education": "Assoc-acdm",
                "education-num": "12",
                "marital-status": "Never-married",
                "occupation": "Sales",
                "relationship": "Not-in-family",
                "race": "Black",
                "sex": "Male",
                "capital-gain": "0",
                "capital-loss": "0",
                "hours-per-week": "50",
                "native-country": "United-States",
                "labels": "0",
            }
        ]
    }
    return data


@pytest.fixture
def test_data_positive():
    data = {
        "dict_features": [
            {
                "age": "46",
                "workclass": "Self-emp-not-inc",
                "fnlgt": "198759",
                "education": "Prof-school",
                "education-num": "15",
                "marital-status": "Married-civ-spouse",
                "occupation": "Prof-specialty",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": "0",
                "capital-loss": "2415",
                "hours-per-week": "80",
                "native-country": "United-States",
                "labels": "1",
            }
        ]
    }
    return data


def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200


def test_api_predict_negative(test_data_negative):
    r = client.post("/predict", json=test_data_negative)
    assert r.status_code == 200
    assert r.json() == "Predicted salary is <= 50K"


def test_api_predict_positive(test_data_positive):
    r = client.post("/predict", json=test_data_positive)
    assert r.status_code == 200
    assert r.json() == "Predicted salary is > 50K"


def test_cloud_api_get():
    end_point = "https://udacity-income-prediction-6a6fa37bfd96.herokuapp.com/"
    r = requests.get(end_point)
    assert r.status_code == 200


def test_cloud_api_post():
    end_point = "https://udacity-income-prediction-6a6fa37bfd96.herokuapp.com/predict"
    data = {
        "dict_features": [
            {
                "age": "46",
                "workclass": "Self-emp-not-inc",
                "fnlgt": "198759",
                "education": "Prof-school",
                "education-num": "15",
                "marital-status": "Married-civ-spouse",
                "occupation": "Prof-specialty",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": "0",
                "capital-loss": "2415",
                "hours-per-week": "80",
                "native-country": "United-States",
                "labels": "1",
            }
        ]
    }
    r = requests.post(end_point, json=data)
    assert r.status_code == 200
    assert r.json() == "Predicted salary is > 50K"
