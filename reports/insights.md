1. Dataset Overview

The dataset contains ~7,000 customer records with 21 features.

Target variable: Churn (Yes / No) indicating whether a customer left the service.

Features include customer demographics, service usage, contract details, and billing information.

2. Data Quality Observations

No duplicate customer records were found.

The TotalCharges column contained 11 empty values, corresponding to customers with zero tenure.

TotalCharges was stored as a string and required type conversion during preprocessing.

3. Churn Distribution

Majority of customers did not churn, indicating class imbalance.

This insight influenced the choice of evaluation metrics beyond accuracy.

4. Key Factors Influencing Churn

Contract Type:
Customers on month-to-month contracts showed significantly higher churn compared to long-term contracts.

Tenure:
Customers with lower tenure were more likely to churn. Most churn occurred within the first year.

Monthly Charges:
Higher monthly charges were associated with increased churn risk.

Payment Method:
Customers using electronic check showed higher churn rates.

Service Usage:
Customers using multiple services churned less, indicating higher engagement.
5. Conclusion

EDA revealed that customer churn is driven by contract flexibility, tenure, pricing, and service engagement.
These insights guided feature engineering and model selection in later stages.
