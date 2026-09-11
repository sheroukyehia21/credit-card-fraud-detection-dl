"""
Credit Card Fraud Detection - Inference App
=============================================

Reproduces the exact inference pipeline built in the notebook:

  Input transaction (V1..V28, Amount, Hour)
        |
        +--> [scaler.pkl]              --> MLP (mlp_model.keras)              --> MLP probability   --> mlp_threshold.pkl
        +--> [autoencoder_scaler.pkl]  --> Autoencoder (autoencoder_model.keras) --> reconstruction error --> autoencoder_threshold.pkl
                        |
                        v
        [MLP probability, AE reconstruction error]
                        |
              ensemble_meta_scaler.pkl
                        |
              ensemble_meta_model.pkl  (Logistic Regression, stacking)
                        |
              ensemble_threshold.pkl
                        |
                        v
                 FRAUD / LEGITIMATE

This app does NOT load creditcard.csv and does NOT retrain anything.
It only loads the artifacts already saved by the notebook.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")

# ----------------------------------------------------------------------------
# Load artifacts (cached so they only load once per session)
# ----------------------------------------------------------------------------


@st.cache_resource
def load_artifacts():
    artifacts = {}

    # Feature order used to build X during training (V1..V28, Amount, Hour)
    artifacts["feature_columns"] = joblib.load("feature_columns.pkl")

    # MLP pipeline
    artifacts["scaler"] = joblib.load("scaler.pkl")
    artifacts["mlp_model"] = tf.keras.models.load_model("mlp_model.keras")
    artifacts["mlp_threshold"] = joblib.load("mlp_threshold.pkl")

    # Autoencoder pipeline
    artifacts["autoencoder_scaler"] = joblib.load("autoencoder_scaler.pkl")
    artifacts["autoencoder_model"] = tf.keras.models.load_model("autoencoder_model.keras")
    artifacts["autoencoder_threshold"] = joblib.load("autoencoder_threshold.pkl")
    artifacts["autoencoder_feature_weights"] = joblib.load("autoencoder_feature_weights.pkl")

    # Ensemble (stacking) pipeline
    artifacts["ensemble_meta_scaler"] = joblib.load("ensemble_meta_scaler.pkl")
    artifacts["ensemble_meta_model"] = joblib.load("ensemble_meta_model.pkl")
    artifacts["ensemble_threshold"] = joblib.load("ensemble_threshold.pkl")

    return artifacts


artifacts = load_artifacts()
feature_columns = artifacts["feature_columns"]

# ----------------------------------------------------------------------------
# Inference helpers (mirror the notebook's logic exactly)
# ----------------------------------------------------------------------------


def reconstruction_error(model, X_scaled, weights):
    """Mean squared reconstruction error per row, weighted per feature.
    Identical to the reconstruction_error() function in the notebook (AE.5).
    """
    X_pred = model.predict(X_scaled, verbose=0)
    squared_error = np.square(X_scaled - X_pred)
    squared_error = squared_error * weights
    return np.mean(squared_error, axis=1)


def run_pipeline(input_df: pd.DataFrame):
    # Make sure the input DataFrame has exactly the training feature order
    input_df = input_df[feature_columns]

    # ---- MLP branch -----------------------------------------------------
    X_scaled_mlp = artifacts["scaler"].transform(input_df)
    mlp_prob = float(artifacts["mlp_model"].predict(X_scaled_mlp, verbose=0).ravel()[0])
    mlp_pred = int(mlp_prob >= artifacts["mlp_threshold"])

    # ---- Autoencoder branch ----------------------------------------------
    X_scaled_ae = artifacts["autoencoder_scaler"].transform(input_df)
    ae_score = float(
        reconstruction_error(
            artifacts["autoencoder_model"],
            X_scaled_ae,
            artifacts["autoencoder_feature_weights"],
        )[0]
    )
    ae_pred = int(ae_score >= artifacts["autoencoder_threshold"])

    # ---- Ensemble (stacking meta-model) -----------------------------------
    meta_input = np.array([[mlp_prob, ae_score]])
    meta_input_scaled = artifacts["ensemble_meta_scaler"].transform(meta_input)
    ensemble_prob = float(
        artifacts["ensemble_meta_model"].predict_proba(meta_input_scaled)[:, 1][0]
    )
    ensemble_pred = int(ensemble_prob >= artifacts["ensemble_threshold"])

    return {
        "mlp_prob": mlp_prob,
        "mlp_pred": mlp_pred,
        "ae_score": ae_score,
        "ae_pred": ae_pred,
        "ensemble_prob": ensemble_prob,
        "ensemble_pred": ensemble_pred,
    }


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("💳 Credit Card Fraud Detection")
st.caption(
    "MLP + Autoencoder ensemble (stacking) — inference only. "
    "Models were trained separately in the project notebook; this app just applies them."
)

st.markdown("### Enter Transaction Details")

col_a, col_b = st.columns(2)

with col_a:
    amount_raw = st.number_input(
        "Transaction Amount (original currency units)",
        min_value=0.0,
        value=100.0,
        step=1.0,
        help="Entered as the raw amount. The app applies the same log1p transform "
        "used in the notebook (Amount = log1p(raw amount)) before scaling.",
    )

with col_b:
    hour = st.slider(
        "Approximate Hour (0-23)",
        min_value=0,
        max_value=23,
        value=12,
        help="Matches the notebook's engineered 'Hour' feature: "
        "(Time_in_seconds // 3600) % 24.",
    )

st.markdown("### V1 - V28 (anonymized PCA features)")
st.caption("These come from the dataset's PCA-transformed features. Default 0.0 is a reasonable neutral value.")

v_values = {}
v_cols = st.columns(4)
for i in range(1, 29):
    col = v_cols[(i - 1) % 4]
    with col:
        v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v_{i}")

st.markdown("---")

if st.button("🔍 Check Transaction", type="primary"):
    # Build the row in the exact same order as feature_columns (V1..V28, Amount, Hour)
    row = {}
    for col in feature_columns:
        if col == "Amount":
            row["Amount"] = np.log1p(amount_raw)
        elif col == "Hour":
            row["Hour"] = float(hour)
        else:
            row[col] = v_values[col]

    input_df = pd.DataFrame([row], columns=feature_columns)

    result = run_pipeline(input_df)

    st.markdown("## Result")

    if result["ensemble_pred"] == 1:
        st.error(f"### 🚨 FRAUD  (ensemble probability: {result['ensemble_prob']:.4f})")
    else:
        st.success(f"### ✅ LEGITIMATE  (ensemble probability: {result['ensemble_prob']:.4f})")

    st.markdown("### Model Breakdown")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("MLP Probability", f"{result['mlp_prob']:.4f}")
        st.write("Prediction:", "Fraud" if result["mlp_pred"] else "Legitimate")

    with m2:
        st.metric("Autoencoder Score", f"{result['ae_score']:.4f}")
        st.write("Prediction:", "Fraud" if result["ae_pred"] else "Legitimate")

    with m3:
        st.metric("Ensemble Probability", f"{result['ensemble_prob']:.4f}")
        st.write("Prediction:", "Fraud" if result["ensemble_pred"] else "Legitimate")

    with st.expander("Thresholds used"):
        st.write("MLP threshold:", artifacts["mlp_threshold"])
        st.write("Autoencoder threshold:", artifacts["autoencoder_threshold"])
        st.write("Ensemble threshold:", artifacts["ensemble_threshold"])
