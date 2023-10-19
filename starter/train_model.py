# Script to train machine learning model.
# Add the necessary imports for the starter code.

import joblib
import pandas as pd
from ml.data import process_data
from ml.model import train_model
from sklearn.model_selection import train_test_split


# Add code to load in the data.
data = pd.read_csv("data/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="labels",
    training=False,
    encoder=encoder,
    lb=lb,
)

# train model
model = train_model(X_train, y_train)

# save model
joblib.dump(model, "./model/linear_regression_model.pkl")
