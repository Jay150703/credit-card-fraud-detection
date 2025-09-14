# step3_model_training.py
"""
Step 3 - Model Training: Fraud Detection
- Trains Isolation Forest, LOF, and XGBoost
- Uses scaled train/test data from Step 2
- Saves confusion matrices and XGBoost model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import joblib

sns.set(style="whitegrid")
os.makedirs("artifacts", exist_ok=True)

# Load train/test data
X_train = pd.read_csv('./artifacts/X_train.csv')
X_test = pd.read_csv('./artifacts/X_test.csv')
y_train = pd.read_csv('./artifacts/y_train.csv').values.ravel()
y_test = pd.read_csv('./artifacts/y_test.csv').values.ravel()

# 1. Isolation Forest
iso = IsolationForest(n_estimators=100, contamination=0.0017, random_state=42)
iso.fit(X_train)
y_pred_iso = iso.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)

cm_iso = confusion_matrix(y_test, y_pred_iso)
sns.heatmap(cm_iso, annot=True, fmt="d", cmap="Blues")
plt.title("Isolation Forest - Confusion Matrix")
plt.savefig(os.path.join("artifacts", "cm_isolation_forest.png"))
plt.close()

print("[Isolation Forest Results]")
print(classification_report(y_test, y_pred_iso, digits=4))

# 2. Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.0017, novelty=True)
lof.fit(X_train)
y_pred_lof = lof.predict(X_test)
y_pred_lof = np.where(y_pred_lof == -1, 1, 0)

cm_lof = confusion_matrix(y_test, y_pred_lof)
sns.heatmap(cm_lof, annot=True, fmt="d", cmap="Greens")
plt.title("Local Outlier Factor - Confusion Matrix")
plt.savefig(os.path.join("artifacts", "cm_lof.png"))
plt.close()

print("[Local Outlier Factor Results]")
print(classification_report(y_test, y_pred_lof, digits=4))

# 3. XGBoost Classifier
xgb = XGBClassifier(
    n_estimators=200, max_depth=4, scale_pos_weight=10,
    random_state=42, use_label_encoder=False, eval_metric="logloss"
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Oranges")
plt.title("XGBoost - Confusion Matrix")
plt.savefig(os.path.join("artifacts", "cm_xgb.png"))
plt.close()

# Save XGBoost model
joblib.dump(xgb, "./artifacts/XGB_model.pkl")

print("[XGBoost Results]")
print(classification_report(y_test, y_pred_xgb, digits=4))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_xgb))
print("✅ Model training completed. Results & plots saved in './artifacts/'")
