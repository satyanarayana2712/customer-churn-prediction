import streamlit as st
import pandas as pd

from src.predict import predict_churn
from src.preprocessing import load_and_preprocess_data
# Load training feature columns
X_train_columns, _ = load_and_preprocess_data("data/raw/telco_customer_churn.csv")

FEATURE_COLUMNS = X_train_columns.columns
st.set_page_config(page_title="Customer Churn Predictor")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn.")
st.subheader("Customer Details")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)

monthly_charges = st.number_input(
    "Monthly Charges", min_value=0.0, max_value=200.0, value=70.0
)

total_charges = st.number_input(
    "Total Charges", min_value=0.0, max_value=10000.0, value=800.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "PaymentMethod": payment_method
}


# Convert input to DataFrame
input_df = pd.DataFrame([input_data])
input_df_encoded = pd.get_dummies(input_df)
input_df_encoded = input_df_encoded.reindex(
    columns=FEATURE_COLUMNS,
    fill_value=0
)
if st.button("Predict Churn"):
    prediction, probability = predict_churn(input_df_encoded)

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is unlikely to churn (Probability: {probability:.2f})")
