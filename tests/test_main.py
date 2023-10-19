from fastapi.testclient import TestClient
from main import app
import numpy as np

client = TestClient(app)


def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200


def test_api_predict_negative():
    # create dummy data with 15 features - negative case

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

    r = client.post("/predict", json=data)
    assert r.status_code == 200
    assert r.json() == "Predicted salary is <= 50K"


def test_api_predict_positive():
    # create dummy data with 15 features - positive case
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

    r = client.post("/predict", json=data)
    assert r.status_code == 200
    assert r.json() == "Predicted salary is > 50K"
