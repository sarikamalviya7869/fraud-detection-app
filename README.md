# 💳 Credit Card Fraud Detection

An end-to-end machine learning project that detects fraudulent credit card transactions using XGBoost, with a deployed Streamlit web app for real-time predictions.

🔗 **Live App:** [sarikamalviya7869-fraud-detection-app-app-bni1ou.streamlit.app](https://sarikamalviya7869-fraud-detection-app-app-bni1ou.streamlit.app/)

## Demo

![App Screenshot](screenshot1.png)
![Prediction Result](screenshot2.png)

*(Add your two screenshots to this folder with these exact filenames, or update the paths above to match.)*

## Problem

The dataset contains 284,807 credit card transactions, of which only **0.17% are fraudulent** — a severe class imbalance that makes this a genuinely challenging classification problem. A naive model that predicts "not fraud" every time would still be 99.8% accurate while catching zero fraud, which is why accuracy alone is the wrong metric here.

## Approach

1. **EDA** — explored class distribution, transaction amount patterns, and checked for missing values
2. **Preprocessing** — scaled `Amount` and `Time` with separate `StandardScaler` instances (the other 28 features are already PCA-transformed)
3. **Class imbalance handling** — used **SMOTE** (Synthetic Minority Oversampling) on the training set only, to avoid leaking synthetic data into the test set
4. **Modeling** — compared Logistic Regression (baseline) against **XGBoost** (final model)
5. **Evaluation** — used precision, recall, F1-score, and ROC-AUC instead of accuracy, since accuracy is meaningless on this imbalanced dataset
6. **Deployment** — packaged the trained model and scalers, built a Streamlit UI, and deployed it on Streamlit Community Cloud

## Tech Stack

- **Python**, pandas, NumPy
- **scikit-learn** (Logistic Regression, StandardScaler, train/test split)
- **XGBoost** (final classifier)
- **imbalanced-learn** (SMOTE)
- **Streamlit** (deployment UI)
- **joblib** (model serialization)

## Project Structure

```
fraud-detection-app/
├── app.py                 # Streamlit app
├── requirements.txt       # Dependencies
├── fraud_model.pkl        # Trained XGBoost model
├── amount_scaler.pkl      # StandardScaler for Amount
├── time_scaler.pkl        # StandardScaler for Time
└── Untitled18 (1).ipynb   # Full training notebook (EDA to model comparison)
```

## Using the App

- **Single Transaction tab** — manually enter Time, Amount, and (optionally) the V1–V28 PCA features to get an instant Fraud / Legitimate prediction with a probability score
- **Batch CSV Upload tab** — upload a CSV of multiple transactions and get predictions for all of them, downloadable as a CSV

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle, by the Machine Learning Group at ULB.

## Author

Built by Sarika Malviya
