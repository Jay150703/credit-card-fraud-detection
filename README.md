# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using Isolation Forest, Local Outlier Factor (LOF), and XGBoost classifier.

---

## Features

- Preprocessing of transaction data
- Training multiple anomaly detection and classification models
- Evaluation of model performance with confusion matrices and ROC-AUC
- Streamlit dashboard for interactive predictions

---

## Project Structure

.
├── data/ # Raw and processed data
├── artifacts/ # Scaled data, models, plots
├── step1_data_loading_eda.py
├── step2_preprocessing.py
├── step3_model_training.py
├── step4_dashboard.py
└── README.md

yaml
Copy code

---

## Requirements

- Python 3.11+
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Streamlit

Install dependencies using:

```bash
pip install -r requirements.txt
Usage
Preprocess data:

bash
Copy code
python3 step2_preprocessing.py
Train models:

bash
Copy code
python3 step3_model_training.py
Run Streamlit dashboard:

bash
Copy code
streamlit run step4_dashboard.py
Notes
Make sure the artifacts/ folder contains the scaled data and trained models.

The dashboard allows you to input transaction features and predict if it is fraudulent.

Author
Jayanthi M
GitHub: Jay150703

yaml
Copy code
