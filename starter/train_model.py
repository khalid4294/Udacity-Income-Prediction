# Script to train machine learning model.
# Add the necessary imports for the starter code.
import sys

sys.path.append("/Users/khalid/Desktop/Udacity-Income-Prediction")

import joblib
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import (
    train_model,
    compute_model_metrics,
    compute_model_slice_metrics,
    inference,
)
from sklearn.model_selection import train_test_split


# Add code to load in the data.
data = pd.read_csv("./data/census_cleaned.csv", index_col=0)
print(f"data loded: {data.shape}")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
print(f"train shape: {train.shape}")
print(f"test shape: {test.shape}")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="labels", training=True
)
print("train data processed")

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="labels",
    training=False,
    encoder=encoder,
    lb=lb,
)
print("test data processed")

# train model
model = train_model(X_train, y_train)
print("model trained")

# save model
joblib.dump(model, "./model/linear_regression_model.pkl")
joblib.dump(encoder, "./model/encoder.pkl")
joblib.dump(lb, "./model/lb.pkl")
print("model saved")

preds = inference(model, X_test)

precision, recall, fbeta, accuracy = compute_model_metrics(y_test, preds)
print(f"Overall Accuracy: {accuracy}")
print(f"Overall Precision: {precision}")
print(f"Overall Recall: {recall}")
print(f"Overall F1-Score: {fbeta}")
print("")
print("")

compute_model_slice_metrics(test, cat_features, model, encoder, lb)
