import os
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from starter.ml.data import process_data


# setting up dvc
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class Features(BaseModel):
    # df must contain 15 features.
    dict_features: List[Dict[str, Any]]
    cat_features: Optional[List[str]] = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


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

    if len(features.dict_features) != 1:
        raise HTTPException(
            status_code=400,
            detail="Invalid input. Please provide only 1 user.",
        )

    elif len(features.dict_features[0]) != 15:
        raise HTTPException(
            status_code=400,
            detail="Invalid input. Please provide 15 features for each user.",
        )

    model, lb, encoder = await load_model()
    df = pd.DataFrame(features.dict_features)
    X, Y, encoder, lb = process_data(
        df,
        categorical_features=features.cat_features,
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
