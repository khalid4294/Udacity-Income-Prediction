import requests

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

end_point = "https://udacity-income-prediction-6a6fa37bfd96.herokuapp.com/predict"
r = requests.post(end_point, json=data)

print(r.text)
print(r.status_code)
