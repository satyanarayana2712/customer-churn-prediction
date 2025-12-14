import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from preprocessing import load_and_preprocess_data
def train_and_save_model():
    # Load and preprocess data
    X, y =load_and_preprocess_data("data/raw/telco_customer_churn.csv")


    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, "models/churn_model.pkl")

    print("✅ Model trained and saved successfully!")


if __name__ == "__main__":
    train_and_save_model()
