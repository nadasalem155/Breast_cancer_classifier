import streamlit as st
import pandas as pd
import numpy as np
import joblib  
from sklearn.preprocessing import StandardScaler

# Load saved model and scaler
mlp_clf = joblib.load("mlp_model.pkl")   
scaler = joblib.load("scaler.pkl")         

# Define features and tooltips
feature_info = {
    'radius_mean': ("📏", "Average radius of the cells – larger radius may indicate abnormal growth."),
    'texture_mean': ("🌀", "Variation in texture/gray-scale values – higher texture can be linked to malignancy."),
    'smoothness_mean': ("✨", "Measures how smooth the cell edges are – irregular edges can indicate tumors."),
    'compactness_mean': ("🧩", "Ratio of area to perimeter squared – higher compactness may suggest malignancy."),
    'symmetry_mean': ("⚖", "Symmetry of the cells – tumors often have irregular shapes."),
    'fractal_dimension_mean': ("🔍", "Complexity of the boundary – higher fractal dimension may be malignant."),
    'radius_se': ("📐", "Standard error of cell radius – variability can show irregular growth."),
    'texture_se': ("🌈", "Standard error of texture – measures variation in texture."),
    'smoothness_se': ("💎", "Standard error of smoothness – checks variation in cell edge smoothness."),
    'compactness_se': ("🧱", "Standard error of compactness – irregular compactness may indicate malignancy."),
    'symmetry_se': ("🔄", "Standard error of symmetry – higher values mean irregular cell symmetry."),
    'symmetry_worst': ("⚠", "Worst (largest) symmetry – higher means more irregular tumor shapes."),
    'fractal_dimension_worst': ("🧬", "Worst fractal dimension – higher values may indicate malignant growth.")
}

feature_names = list(feature_info.keys())

# Slider ranges 
feature_ranges = {
    'radius_mean': (6.981, 21.9),
    'texture_mean': (9.71, 30.245),
    'smoothness_mean': (0.057975, 0.133695),
    'compactness_mean': (0.01938, 0.22862),
    'symmetry_mean': (0.1112, 0.2464),
    'fractal_dimension_mean': (0.04996, 0.07875),
    'radius_se': (0.1115, 0.84865),
    'texture_se': (0.3602, 2.43415),
    'smoothness_se': (0.001713, 0.0126115),
    'compactness_se': (0.002252, 0.061505),
    'symmetry_se': (0.007882, 0.03596),
    'symmetry_worst': (0.1565, 0.41915),
    'fractal_dimension_worst': (0.05504, 0.12301)
}

# Streamlit UI
st.title("🩺 Breast Cancer Prediction (MLP Classifier)")
st.markdown("Use the sliders below to input feature values and predict whether the tumor is *Malignant 🔴* or *Benign 🟢*.")

# Input sliders in two columns
user_input = []
cols = st.columns(2)
for i, feature in enumerate(feature_names):
    col = cols[i % 2]
    emoji, tooltip = feature_info[feature]
    min_val, max_val = feature_ranges[feature]
    with col:
        val = st.slider(
            f"{emoji} {feature}",
            float(min_val), float(max_val), float((min_val + max_val) / 2),
            help=tooltip
        )
        user_input.append(val)

# Convert to numpy array and scale
input_array = np.array(user_input).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# Predict using probability threshold
if st.button("🔮 Predict"):
    proba = mlp_clf.predict_proba(input_scaled)[0]  # probabilities for each class
    proba_malignant = proba[1]
    proba_benign = proba[0]

    # Use probability directly to determine prediction
    prediction_label = "Malignant 🔴" if proba_malignant > 0.5 else "Benign 🟢"
    color = "#FF4C4C" if prediction_label == "Malignant 🔴" else "#4CAF50"

    # Show prediction box
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; color:white; font-size:24px; font-weight:bold;">
            Prediction: {prediction_label}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show probabilities
    st.write(f"Probability of Malignant 🔴: {proba_malignant:.2f}")
    st.write(f"Probability of Benign 🟢: {proba_benign:.2f}")
