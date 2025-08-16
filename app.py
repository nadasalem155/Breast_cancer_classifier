import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model and scaler
model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🩺 Breast Cancer Prediction App")

st.markdown(
    "This app predicts whether a breast tumor is *Malignant (cancerous)* or *Benign (non-cancerous)* "
    "using *tumor cell features*."
)

# Feature info (meaning + effect)
feature_info = {
    "mean_radius": "👩‍⚕ *Radius* → Larger radius often indicates malignant tumors.",
    "mean_texture": "🌐 *Texture* → Variation in gray-scale; higher values may relate to malignancy.",
    "mean_perimeter": "📏 *Perimeter* → Bigger perimeter usually means higher cancer risk.",
    "mean_area": "📐 *Area* → Larger tumor area is linked with malignancy.",
    "mean_smoothness": "✨ *Smoothness* → Less smooth (rough edges) often indicates cancer.",
    "mean_compactness": "📦 *Compactness* → Dense/compact nuclei may suggest malignancy.",
    "mean_concavity": "⬇ *Concavity* → Deeper concave parts in tumor shape = higher cancer probability.",
    "mean_concave_points": "🔻 *Concave Points* → More concave points often mean malignancy.",
    "mean_symmetry": "🔄 *Symmetry* → Tumors with irregular symmetry are more likely malignant.",
    "mean_fractal_dimension": "🌀 *Fractal Dimension* → Complex, irregular borders often relate to malignancy."
}

# Input layout
st.subheader("🔢 Enter Tumor Features")

col1, col2 = st.columns(2)
inputs = {}

features = list(feature_info.keys())

for i, feature in enumerate(features):
    if i % 2 == 0:
        with col1:
            inputs[feature] = st.number_input(
                f"{feature}", min_value=0.0, step=0.01
            )
            st.caption(feature_info[feature])
    else:
        with col2:
            inputs[feature] = st.number_input(
                f"{feature}", min_value=0.0, step=0.01
            )
            st.caption(feature_info[feature])

# Predict button
if st.button("🔍 Predict"):
    # Prepare input
    input_data = np.array(list(inputs.values())).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data_scaled)[0]
    prediction_proba = model.predict_proba(input_data_scaled)[0]

    # Result Box (colored)
    if prediction == 0:
        st.markdown(
            "<div style='padding:15px; border-radius:10px; background-color:#d4edda; color:#155724;'>"
            "✅ The tumor is <b>Benign</b> (non-cancerous)."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='padding:15px; border-radius:10px; background-color:#f8d7da; color:#721c24;'>"
            "⚠ The tumor is <b>Malignant</b> (cancerous)."
            "</div>",
            unsafe_allow_html=True
        )

    # Show probabilities in a bar chart
    st.subheader("📊 Prediction Probability")
    fig, ax = plt.subplots()
    ax.bar(["Benign", "Malignant"], prediction_proba, color=["green", "red"])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)