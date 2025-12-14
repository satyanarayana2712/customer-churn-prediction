import joblib
import pandas as pd

# Load trained model
MODEL_PATH = "models/churn_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_churn(input_df):
    """
    Takes a preprocessed dataframe and returns
    churn prediction and probability
    """
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    return prediction[0], probability[0]
