# step1_data_loading_eda.py
"""
Step 1 - Data loading & basic EDA for Fraud Detection project

This script expects the Kaggle 'creditcard.csv' to be placed at ./data/creditcard.csv.
It produces several EDA artifacts in ./artifacts/:
 - eda_summary.txt (text summary)
 - df_head.csv (first rows)
 - class_distribution.png
 - amount_hist.png
 - amount_hist_log.png
 - correlation_matrix.png
 - fraud_samples.csv, nonfraud_samples.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set(style='whitegrid')

DATA_PATH = os.path.join('data', 'creditcard.csv')
if not os.path.exists(DATA_PATH):
    print(f"ERROR: dataset not found at {DATA_PATH}.\nPlease download 'creditcard.csv' from Kaggle (https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the 'data/' folder.")
else:
    df = pd.read_csv(DATA_PATH)
    os.makedirs('artifacts', exist_ok=True)

    # Save head preview
    df.head().to_csv(os.path.join('artifacts', 'df_head.csv'), index=False)

    # Write summary text
    with open(os.path.join('artifacts', 'eda_summary.txt'), 'w') as f:
        f.write('Shape: ' + str(df.shape) + '\n\n')
        f.write('Missing values:\n')
        f.write(df.isnull().sum().to_string())
        f.write('\n\nClass distribution (0 = Normal, 1 = Fraud):\n')
        f.write(df['Class'].value_counts().to_string())
        f.write('\n\nDescribe:\n')
        f.write(df.describe().to_string())

    # Class distribution plot
    plt.figure(figsize=(6,4))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0=Normal, 1=Fraud)')
    plt.savefig(os.path.join('artifacts', 'class_distribution.png'))
    plt.close()

    # Amount distribution (raw)
    plt.figure(figsize=(8,4))
    sns.histplot(df['Amount'], bins=50, kde=False)
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount')
    plt.savefig(os.path.join('artifacts', 'amount_hist.png'))
    plt.close()

    # Amount distribution (log1p)
    plt.figure(figsize=(8,4))
    sns.histplot(np.log1p(df['Amount']), bins=50, kde=False)
    plt.title('Log-Transformed Transaction Amount Distribution (log1p)')
    plt.xlabel('log1p(Amount)')
    plt.savefig(os.path.join('artifacts', 'amount_hist_log.png'))
    plt.close()

    # Correlation heatmap
    corr = df.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.2)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join('artifacts', 'correlation_matrix.png'))
    plt.close()

    # Save small samples of fraud and non-fraud for inspection
    fraud_count = int(df['Class'].sum())
    df[df['Class']==1].sample(min(200, fraud_count), random_state=42).to_csv(os.path.join('artifacts', 'fraud_samples.csv'), index=False)
    df[df['Class']==0].sample(200, random_state=42).to_csv(os.path.join('artifacts', 'nonfraud_samples.csv'), index=False)

    print("EDA completed. Artifacts saved to './artifacts/' folder.")
