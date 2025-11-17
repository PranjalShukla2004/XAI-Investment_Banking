# predict_new_companies.py
import joblib
import pandas as pd
from tabulate import tabulate

# ==========================
# 1. Load trained artifacts
# ==========================
model = joblib.load("artifacts/xgb_risk_model.joblib")
le = joblib.load("artifacts/label_encoder.joblib")

with open("artifacts/feature_list.txt") as f:
    feature_cols = [line.strip() for line in f]

# ==========================
# 2. Define sample companies
# ==========================
# Approximate values in USD (billions scaled to scientific notation)
companies = pd.DataFrame([
    # --- Low-risk, large caps ---
    {
        "Company": "Apple Inc.",
        "WorkingCapital": 3.6e10,
        "TotalLiabilities": 2.9e11,
        "StockholdersEquity": 6.8e10,
        "EBIT_proxy": 1.15e11,
        "Revenue": 3.9e11
    },
    {
        "Company": "Microsoft Corp.",
        "WorkingCapital": 1.0e11,
        "TotalLiabilities": 2.1e11,
        "StockholdersEquity": 2.1e11,
        "EBIT_proxy": 1.0e11,
        "Revenue": 2.4e11
    },
    # --- Moderate-risk ---
    {
        "Company": "Ford Motor Co.",
        "WorkingCapital": 2.1e10,
        "TotalLiabilities": 2.0e11,
        "StockholdersEquity": 6.5e10,
        "EBIT_proxy": 1.5e10,
        "Revenue": 1.6e11
    },
    # --- Higher-risk / leveraged firms ---
    {
        "Company": "GameStop Corp.",
        "WorkingCapital": 1.0e9,
        "TotalLiabilities": 3.5e9,
        "StockholdersEquity": 1.2e9,
        "EBIT_proxy": 6.0e7,
        "Revenue": 5.7e9
    },
    {
        "Company": "AMC Entertainment Holdings",
        "WorkingCapital": -2.8e9,   # negative WC common in distress
        "TotalLiabilities": 1.1e10,
        "StockholdersEquity": -2.5e9,
        "EBIT_proxy": 2.0e8,
        "Revenue": 4.8e9
    }
])

# Keep only model features for prediction
X_new = companies[feature_cols]

# ==========================
# 3. Run model predictions
# ==========================
pred_probs = model.predict_proba(X_new)[:, 1]  # probability of high risk
pred_labels = (pred_probs >= 0.5).astype(int)
decoded_labels = le.inverse_transform(pred_labels)

companies["Predicted_Risk_Label"] = decoded_labels
companies["High_Risk_Probability"] = pred_probs.round(3)

# ==========================
# 4. Display results
# ==========================
cols_to_show = ["Company"] + feature_cols + ["Predicted_Risk_Label", "High_Risk_Probability"]
print(tabulate(companies[cols_to_show], headers="keys", tablefmt="pretty", showindex=False))

# ==========================
# 5. Optional: Save to CSV
# ==========================
companies.to_csv("artifacts/predictions_new_companies.csv", index=False)
print("\nPredictions saved to artifacts/predictions_new_companies.csv")
