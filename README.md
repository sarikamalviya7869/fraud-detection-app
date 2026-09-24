# Fraud Detection — Streamlit Deployment

## Step 1 — Save the model (add this as a NEW cell in your Colab notebook, after Cell 6)

⚠️ **Important fix**: in your original notebook, `Amount` and `Time` were both scaled with the *same* `scaler` object, and `fit_transform` was called twice — the second call (on `Time`) overwrites the fit from the first (on `Amount`). This means the saved scaler would only really be correct for `Time`, not `Amount`. Use **two separate scalers** instead, as below.

Replace your original Cell 3 scaling lines with this:

```python
from sklearn.preprocessing import StandardScaler

amount_scaler = StandardScaler()
time_scaler = StandardScaler()

X['Amount'] = amount_scaler.fit_transform(X[['Amount']])
X['Time'] = time_scaler.fit_transform(X[['Time']])
```

Then add this new cell after training XGBoost (after Cell 6) to save everything:

```python
import joblib

joblib.dump(xgb, "fraud_model.pkl")
joblib.dump(amount_scaler, "amount_scaler.pkl")
joblib.dump(time_scaler, "time_scaler.pkl")

# Download them to your computer
from google.colab import files
files.download("fraud_model.pkl")
files.download("amount_scaler.pkl")
files.download("time_scaler.pkl")
```

This downloads 3 files to your computer: `fraud_model.pkl`, `amount_scaler.pkl`, `time_scaler.pkl`.

**Update:** `app.py` has been updated to use the two separate scalers (`amount_scaler.pkl` and `time_scaler.pkl`) — no further changes needed there.

## Step 2 — Set up your GitHub repo

Create a new repo (or use an existing one) with this structure:

```
fraud-detection-app/
├── app.py
├── requirements.txt
├── fraud_model.pkl
├── amount_scaler.pkl
├── time_scaler.pkl
```

Push all files, including the `.pkl` files (they're small enough for a normal git push here).

## Step 3 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app", select your repo, branch, and `app.py` as the main file
4. Click Deploy

You'll get a live public link like `https://your-app-name.streamlit.app` — put this in your resume/portfolio/GitHub README.

## What the app does

- **Single Transaction tab**: manually enter Time, Amount, and (optionally) the V1–V28 PCA features, get an instant Fraud/Legitimate prediction with probability
- **Batch CSV tab**: upload a CSV of multiple transactions, get predictions for all of them, download results
