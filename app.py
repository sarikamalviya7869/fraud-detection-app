import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")

# ---------- Load model & scalers ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_model.pkl")
    amount_scaler = joblib.load("amount_scaler.pkl")
    time_scaler = joblib.load("time_scaler.pkl")
    return model, amount_scaler, time_scaler

model, amount_scaler, time_scaler = load_artifacts()

FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

st.title("💳 Credit Card Fraud Detector")
st.caption("XGBoost model trained on SMOTE-balanced transaction data")

tab1, tab2 = st.tabs(["🔍 Single Transaction", "📁 Batch CSV Upload"])

# ---------- TAB 1: Single transaction ----------
with tab1:
    st.subheader("Enter Transaction Details")
    st.markdown(
        "The V1–V28 fields are PCA-anonymized features from the original dataset. "
        "For a quick test, you can leave them at 0 and just adjust **Time** and **Amount**."
    )

    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input("Time (seconds since first transaction)", value=0.0)
    with col2:
        amount_val = st.number_input("Amount ($)", value=100.0, min_value=0.0)

    with st.expander("Advanced: set V1–V28 manually"):
        v_cols = st.columns(4)
        v_values = {}
        for i in range(1, 29):
            with v_cols[(i - 1) % 4]:
                v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v_{i}", format="%.4f")

    if st.button("Predict", type="primary"):
        row = {**v_values, "Time": time_val, "Amount": amount_val}
        input_df = pd.DataFrame([row])[FEATURE_COLS]

        # Scale Time and Amount exactly as done in training (separate scalers)
        input_df["Amount"] = amount_scaler.transform(input_df[["Amount"]])
        input_df["Time"] = time_scaler.transform(input_df[["Time"]])

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.divider()
        if pred == 1:
            st.error(f"🚨 Predicted: **FRAUD** (probability: {prob:.2%})")
        else:
            st.success(f"✅ Predicted: **Legitimate** (fraud probability: {prob:.2%})")

        st.progress(min(float(prob), 1.0))

# ---------- TAB 2: Batch CSV ----------
with tab2:
    st.subheader("Upload a CSV of transactions")
    st.markdown("CSV must contain the columns: `Time, V1, V2, ..., V28, Amount` (same format as the original dataset, without the `Class` column).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)

        missing = set(FEATURE_COLS) - set(batch_df.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            scaled_df = batch_df.copy()
            scaled_df["Amount"] = amount_scaler.transform(scaled_df[["Amount"]])
            scaled_df["Time"] = time_scaler.transform(scaled_df[["Time"]])

            preds = model.predict(scaled_df[FEATURE_COLS])
            probs = model.predict_proba(scaled_df[FEATURE_COLS])[:, 1]

            result_df = batch_df.copy()
            result_df["Predicted_Class"] = preds
            result_df["Fraud_Probability"] = probs

            st.write(f"**{int(preds.sum())} flagged as fraud** out of {len(preds)} transactions")
            st.dataframe(
                result_df.sort_values("Fraud_Probability", ascending=False),
                use_container_width=True,
            )

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results as CSV", csv_out, "fraud_predictions.csv", "text/csv")

st.divider()
st.caption("Model: XGBoost | Class imbalance handled with SMOTE | Built by Sarika")
