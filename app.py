import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("mlp_model.pkl")

# Title with emoji
st.markdown("<h1 style='color:#4B9CD3;'>🩺 Breast Cancer Classifier</h1>", unsafe_allow_html=True)
st.write("Fill in the values below to predict if the tumor is *Benign* or *Malignant*.")

# Feature descriptions
feature_descriptions = {
    'mean radius': "Average distance from the center to points on the perimeter.",
    'mean texture': "Variation in gray-scale values (cell texture).",
    'mean perimeter': "Average size of the cell perimeter.",
    'mean area': "Average area of the cells.",
    'mean smoothness': "How smooth the cell edges are.",
    'mean compactness': "How compact (dense) the cells are.",
    'mean concavity': "Severity of concave portions of the contour.",
    'mean concave points': "Number of concave portions of the contour.",
    'mean symmetry': "Symmetry of the cells.",
    'mean fractal dimension': "Complexity of the cell borders."
}

# Emojis for each feature
emoji_map = {
    'mean radius': "📏🔬",
    'mean texture': "🧵🎨",
    'mean perimeter': "📐🧭",
    'mean area': "📊🧫",
    'mean smoothness': "🌊✨",
    'mean compactness': "🧩📦",
    'mean concavity': "⬇⚫",
    'mean concave points': "🔹📍",
    'mean symmetry': "⚖🔄",
    'mean fractal dimension': "🌀🌐"
}

# User inputs
st.subheader("Enter Feature Values 🔽")
user_data = {}
for feature, description in feature_descriptions.items():
    label = f"{emoji_map[feature]} {feature}"
    user_data[feature] = st.number_input(
        label, 
        min_value=0.0, 
        max_value=100.0, 
        step=0.1, 
        help=description
    )

# Prediction button
if st.button("🔮 Predict"):
    input_df = pd.DataFrame([user_data])
    prediction = model.predict(input_df)[0]
    result = "✅ Benign (Not Cancerous)" if prediction == 1 else "⚠ Malignant (Cancerous)"
    st.markdown(f"<h2 style='color:#2E8B57;'>{result}</h2>", unsafe_allow_html=True)