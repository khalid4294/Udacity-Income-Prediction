import requests

data = {
    "age": 32,
    "workclass": "Private",
    "fnlgt": 205019,
    "education": "Assoc-acdm",
    "education_num": 12,
    "marital_status": "Never-married",
    "occupation": "Sales",
    "relationship": "Not-in-family",
    "race": "Black",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States",
    "labels": 0,
}

end_point = "https://udacity-income-prediction-6a6fa37bfd96.herokuapp.com/predict"
r = requests.post(end_point, json=data)

print(r.text)
print(r.status_code)
