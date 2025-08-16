import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open("mlp_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# App Title
st.title("🩺 Breast Cancer Classifier")

st.write("This app predicts whether a tumor is *Benign (Non-cancerous)* or *Malignant (Cancerous)* based on cell features.")

st.markdown("---")

# Features in two columns
col1, col2 = st.columns(2)

with col1:
    radius_mean = st.slider("Radius Mean", 6.0, 30.0, 14.0)
    st.caption("🔵 Average distance from center to edges (size of tumor).")

    texture_mean = st.slider("Texture Mean", 9.0, 40.0, 20.0)
    st.caption("🎨 Variation in gray levels (smooth or rough).")

    smoothness_mean = st.slider("Smoothness Mean", 0.05, 0.20, 0.10)
    st.caption("✨ Smoothness of cell boundary.")

    compactness_mean = st.slider("Compactness Mean", 0.02, 0.40, 0.15)
    st.caption("📏 Density of shape (area vs perimeter).")

    symmetry_mean = st.slider("Symmetry Mean", 0.1, 0.4, 0.2)
    st.caption("⚖ Symmetry of the cell shape.")

    fractal_dimension_mean = st.slider("Fractal Dimension Mean", 0.04, 0.10, 0.06)
    st.caption("🌀 Complexity of the boundary.")

with col2:
    radius_se = st.slider("Radius SE", 0.1, 3.0, 0.5)
    st.caption("📐 Variation in radius sizes.")

    texture_se = st.slider("Texture SE", 0.2, 5.0, 1.0)
    st.caption("🎭 Variation in texture roughness.")

    smoothness_se = st.slider("Smoothness SE", 0.001, 0.01, 0.005)
    st.caption("🌊 Variation in smoothness.")

    compactness_se = st.slider("Compactness SE", 0.01, 0.30, 0.05)
    st.caption("🧩 Variation in compactness.")

    symmetry_se = st.slider("Symmetry SE", 0.01, 0.08, 0.02)
    st.caption("🔄 Variation in symmetry.")

    symmetry_worst = st.slider("Symmetry Worst", 0.1, 0.6, 0.3)
    st.caption("⚠ Worst (least symmetric) value.")

    fractal_dimension_worst = st.slider("Fractal Dimension Worst", 0.05, 0.20, 0.12)
    st.caption("🧶 Most irregular boundary value.")

# Prepare input
features = np.array([[radius_mean, texture_mean, smoothness_mean, compactness_mean, 
                      symmetry_mean, fractal_dimension_mean, radius_se, texture_se, 
                      smoothness_se, compactness_se, symmetry_se, symmetry_worst, 
                      fractal_dimension_worst]])

scaled_features = scaler.transform(features)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(scaled_features)[0]

    if prediction == 0:
        st.success("✅ The tumor is *Benign (Non-cancerous)* 🎉")
    else:
        st.error("⚠ The tumor is *Malignant (Cancerous)* 🚨")

    # Bar chart of inputs
    st.subheader("📊 Input Features Visualization")
    feature_names = ["radius_mean","texture_mean","smoothness_mean","compactness_mean",
                     "symmetry_mean","fractal_dimension_mean","radius_se","texture_se",
                     "smoothness_se","compactness_se","symmetry_se","symmetry_worst",
                     "fractal_dimension_worst"]

    df = pd.DataFrame(features, columns=feature_names)

    plt.figure(figsize=(10,5))
    df.iloc[0].plot(kind="bar")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title("Input Feature Values")
    st.pyplot(plt)