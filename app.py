import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("fraud_detection_model.joblib")

st.set_page_config(page_title="Fraud Detection App", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.markdown("Upload a CSV or JSON file containing transaction data.")

st.divider()

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV or JSON File",
    type=["csv", "json"]
)

# REQUIRED columns (including Time)
expected_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

if uploaded_file is not None:

    try:
        # ----------------------------
        # Read File
        # ----------------------------
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(".json"):
            data_json = json.load(uploaded_file)

            if isinstance(data_json, dict):
                data = pd.DataFrame([data_json])
            else:
                data = pd.DataFrame(data_json)

        else:
            st.error("Unsupported file type.")
            st.stop()

        # ----------------------------
        # Clean Columns
        # ----------------------------
        data.columns = data.columns.str.strip()

        if "Class" in data.columns:
            data = data.drop(columns=["Class"])

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(data.head())

        # ----------------------------
        # Validate Columns
        # ----------------------------
        if not all(col in data.columns for col in expected_columns):
            st.error("File does not contain required columns.")
            st.write("Columns found in file:", list(data.columns))
            st.write("Required columns:", expected_columns)
            st.stop()

        data = data[expected_columns]

        # ----------------------------
        # Prediction
        # ----------------------------
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]

        result_df = data.copy()
        result_df["Prediction"] = predictions
        result_df["Prediction"] = result_df["Prediction"].map({0: "Legit", 1: "Fraud"})
        result_df["Fraud_Probability"] = probabilities

        st.divider()

        # ----------------------------
        # Single Transaction Result
        # ----------------------------
        if len(result_df) == 1:

            prediction = result_df["Prediction"].iloc[0]
            prob = result_df["Fraud_Probability"].iloc[0]

            st.subheader("🔍 Prediction Result")

            if prediction == "Fraud":
                st.error(f"⚠️ FRAUD DETECTED! (Confidence: {prob:.2%})")
            else:
                st.success(f"✅ LEGITIMATE TRANSACTION (Confidence: {(1 - prob):.2%})")

            st.dataframe(result_df)

        # ----------------------------
        # Multiple Transactions Result
        # ----------------------------
        else:

            fraud_count = (result_df["Prediction"] == "Fraud").sum()
            legit_count = (result_df["Prediction"] == "Legit").sum()
            total = len(result_df)

            st.subheader("📊 Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Transactions", total)
            col2.metric("Legit Transactions", legit_count)
            col3.metric("Fraud Transactions", fraud_count)

            st.divider()

            st.subheader("📑 Detailed Results")
            st.dataframe(result_df)

            # Download button
            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇ Download Results",
                csv,
                "fraud_predictions.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info(" Please upload a CSV or JSON file to begin.")
