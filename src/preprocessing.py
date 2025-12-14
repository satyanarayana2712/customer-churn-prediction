import pandas as pd

def load_and_preprocess_data(csv_path=None):
    """
    Loads raw data and performs cleaning + encoding
    """
    df= pd.read_csv("data/raw/telco_customer_churn.csv")


    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['TotalCharges'].isnull(), 'TotalCharges'] = 0

    # Clean and encode target
    df['Churn'] = df['Churn'].astype(str).str.strip()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop ID column
    df = df.drop(columns=['customerID'], errors='ignore')

    # Split features & target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # One-hot encode
    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded, y
