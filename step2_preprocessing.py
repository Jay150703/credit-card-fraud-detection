# step2_preprocessing.py
"""
Step 2 - Preprocessing for Fraud Detection
- Removes 'Time' column
- Splits data into train/test
- Scales the features using StandardScaler
- Saves processed CSVs and scaler object in './artifacts/'
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

os.makedirs("artifacts", exist_ok=True)

# Load data
df = pd.read_csv("./data/creditcard.csv")

# Drop 'Time' column
df = df.drop(columns=["Time"])

# Split features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaled data and scaler
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("./artifacts/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("./artifacts/X_test.csv", index=False)
y_train.to_csv("./artifacts/y_train.csv", index=False)
y_test.to_csv("./artifacts/y_test.csv", index=False)
joblib.dump(scaler, "./artifacts/scaler.pkl")

print("✅ Preprocessing completed. Scaled data & scaler saved in './artifacts/'")
