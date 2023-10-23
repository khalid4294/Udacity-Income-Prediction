from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from starter.ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : LogisticRegression model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    preds = model.predict(X)
    return preds


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    accuracy = accuracy_score(y, preds)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta, accuracy


def compute_model_slice_metrics(test, cat_features, model, encoder, lb):
    """
    Validates slices of the trained machine learning model using precision, recall, and F1.
    """
    string = ""

    for i in cat_features:
        unique_vals = test[i].unique()

        for j in unique_vals:
            test_slice = test[test[i] == j]
            X_test, y_test, encoder, lb = process_data(
                test_slice,
                categorical_features=cat_features,
                label="labels",
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds = model.predict(X_test)

            string += f"accuracy for feature {i} with value {j}: {accuracy_score(y_test, preds)}\n"
            string += f"Test shape: {test_slice.shape}\n"
            string += (
                f"Test labels (0): {test_slice['labels'].value_counts().values[0]}\n"
            )
            try:
                string += f"Test labels (1): {test_slice['labels'].value_counts().values[1]}\n"
            except:
                string += "Test labels (1): 0\n"

    with open("screenshots/slice_output.txt", "w") as f:
        f.write(string)
