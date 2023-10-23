import os
import pandas as pd
import joblib
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from starter.ml.data import process_data


# setting up dvc
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def to_snake_case(string: str) -> str:
    return string.replace("-", "_")


class Features(BaseModel):
    age: int = Field(example=32)
    workclass: str = Field(example="Private")
    fnlgt: int = Field(example=205019)
    education: str = Field(example="Assoc-acdm")
    education_num: int = Field(example=12)
    marital_status: str = Field(example="Never-married")
    occupation: str = Field(example="Sales")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="Black")
    sex: str = Field(example="Male")
    capital_gain: int = Field(example=0)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=50)
    native_country: str = Field(example="United-States")
    labels: int = Field(example=0)

    class Config:
        alias_generator = to_snake_case
        allow_population_by_field_name = True


app = FastAPI(
    title="Salary Prediction API",
    description="An API runs inference on 108 features and returns a prediction of whether a person makes over 50K a year.",
    version="1.0.0",
)


async def load_model():
    model = joblib.load("./model/linear_regression_model.pkl")
    lb = joblib.load("./model/lb.pkl")
    encoder = joblib.load("./model/encoder.pkl")
    return model, lb, encoder


@app.get("/")
def home():
    return "Welcome to the Salary Prediction API!"


@app.post("/predict")
async def predict(features: Features):
    # adding HTTPException for invalid input

    dict_features = features.dict()
    if len(dict_features) != 15:
        raise HTTPException(
            status_code=400,
            detail="Invalid input. Please provide 15 features for each user.",
        )

    model, lb, encoder = await load_model()
    df = pd.DataFrame(dict_features, index=[0])
    X, Y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="labels",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = model.predict(X)
    if preds[0] == 0:
        return "Predicted salary is <= 50K"
    else:
        return "Predicted salary is > 50K"
