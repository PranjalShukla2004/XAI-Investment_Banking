# xgb_risk_model.py
import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

# --------------------------
# 1) Load data
# --------------------------
CSV_PATH = "./datasets/company_fins_plus_market.csv"  # adjust path if needed
df = pd.read_csv(CSV_PATH)

# --------------------------
# 2) Target & feature mapping
# --------------------------
# Core signals per your objectives:
# Liquidity: WorkingCapital
# Solvency/Leverage: TotalLiabilities, RetainedEarnings
# Profitability: RetainedEarnings, EBIT
# Activity: Sales (Revenue)

feature_map = {
    "liquidity_working_capital": "WorkingCapital",
    "leverage_total_liabilities": "TotalLiabilities",
    "profitability_ebit": "EBIT_proxy",      # using EBIT_proxy as EBIT
    "activity_sales": "Revenue"
}

# Handle Retained Earnings:
retained_candidates = ["RetainedEarnings", "Retained_Earnings"]
retained_col = next((c for c in retained_candidates if c in df.columns), None)

if retained_col is None:
    # Temporary proxy so you can run end-to-end now
    if "StockholdersEquity" in df.columns:
        warnings.warn(
            "Retained earnings not found. Using StockholdersEquity as a TEMPORARY proxy. "
            "Replace with true retained earnings when available."
        )
        retained_col = "StockholdersEquity"
    else:
        raise ValueError(
            "No RetainedEarnings column found and StockholdersEquity is also missing. "
            "Please add a RetainedEarnings column to fully match the 5-record spec."
        )

feature_map["solvency_profitability_retained"] = retained_col

# Final feature list (five signals)
feature_cols = list(feature_map.values())

# --------------------------
# 3) Build dataset (X, y)
# --------------------------
if "risk_label" not in df.columns:
    raise ValueError("The dataset must include a 'risk_label' column (e.g., 'low_risk'/'high_risk').")

X = df[feature_cols].copy()
y = df["risk_label"].astype(str).copy()

# Encode target to {0,1}
le = LabelEncoder()
y_enc = le.fit_transform(y)  # e.g., {'high_risk':1, 'low_risk':0} depending on fit

# --------------------------
# 4) Train / test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# --------------------------
# 5) Preprocess + Model
# --------------------------
numeric_features = feature_cols
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    # XGBoost doesn't require scaling; keeping it numeric only
])

preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)],
    remainder="drop"
)

# Reasonable starting hyperparams for tabular finance, tweak later
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", xgb)
])

# --------------------------
# 6) Train
# --------------------------
model.fit(X_train, y_train)

# --------------------------
# 7) Evaluate
# --------------------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

# If labels are single-class in test split, ROC-AUC can fail; guard it:
try:
    roc = roc_auc_score(y_test, y_prob)
except ValueError:
    roc = np.nan

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

print("=== Metrics ===")
print(f"Accuracy     : {acc:.3f}")
print(f"Precision    : {prec:.3f}")
print(f"Recall       : {rec:.3f}")
print(f"F1           : {f1:.3f}")
print(f"ROC-AUC      : {roc:.3f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# --------------------------
# 8) Save artifacts
# --------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/xgb_risk_model.joblib")
joblib.dump(le, "artifacts/label_encoder.joblib")
with open("artifacts/feature_list.txt", "w") as f:
    f.write("\n".join(feature_cols))

print("\nSaved: artifacts/xgb_risk_model.joblib, label_encoder.joblib, feature_list.txt")

# --------------------------
# 9) How to predict later
# --------------------------
# Example:
# loaded_model = joblib.load("artifacts/xgb_risk_model.joblib")
# newX = pd.DataFrame([{
#     "WorkingCapital": 1.2e9,
#     "TotalLiabilities": 6.5e9,
#     "EBIT_proxy": 8.1e8,
#     "Revenue": 4.7e9,
#     retained_col: 2.3e9
# }])
# pred_prob = loaded_model.predict_proba(newX)[0,1]
# pred_label = (pred_prob >= 0.5).astype(int)
# print("Risk prob:", pred_prob)
# print("Pred class:", le.inverse_transform([pred_label])[0])
