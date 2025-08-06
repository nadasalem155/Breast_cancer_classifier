import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Load the dataset
df = pd.read_csv("breast_cancer_data.csv")  

# Define features and target
feature_names = [
    'radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
    'smoothness_se', 'compactness_se', 'symmetry_se', 'symmetry_worst',
    'fractal_dimension_worst'
]
X = df[feature_names]
y = df['diagnosis']  

# Fit scaler and scale X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit MLP classifier
mlp_clf = MLPClassifier(random_state=42, max_iter=1000)
mlp_clf.fit(X_scaled, y)

# Min and max values for sliders (you can extract these from your dataset too)
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
st.title("Breast Cancer Prediction (MLP Classifier)")
st.markdown("Enter values for the features to predict whether the tumor is *Malignant* or *Benign*.")

# Input sliders
user_input = []
for feature in feature_names:
    min_val, max_val = feature_ranges[feature]
    val = st.slider(f"{feature}", float(min_val), float(max_val), float((min_val + max_val) / 2))
    user_input.append(val)

# Convert to numpy array and scale
input_array = np.array(user_input).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# Predict
if st.button("Predict"):
    prediction = mlp_clf.predict(input_scaled)[0]
    prediction_label = "Malignant" if prediction == 1 else "Benign"
    st.subheader(f"Prediction: *{prediction_label}*")