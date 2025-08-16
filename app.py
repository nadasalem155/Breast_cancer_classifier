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

# Min and max values for sliders
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

# --- Streamlit UI ---
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")

# Header
st.markdown(
    """
    <div style="text-align:center; padding: 10px; background-color:#f9f9f9; border-radius:10px;">
        <h1 style="color:#d6336c;">🩺 Breast Cancer Prediction App</h1>
        <p style="font-size:18px;">🔧 Adjust the sliders below and click <b>Predict</b> to see the result.</p>
    </div>
    """, unsafe_allow_html=True
)

# Two-column layout
col1, col2 = st.columns(2)

# Add emojis to sliders
emoji_map = {
    'radius_mean': "📏",
    'texture_mean': "🎨",
    'smoothness_mean': "✨",
    'compactness_mean': "🧩",
    'symmetry_mean': "🔄",
    'fractal_dimension_mean': "🌀",
    'radius_se': "📐",
    'texture_se': "🌈",
    'smoothness_se': "💎",
    'compactness_se': "⚙",
    'symmetry_se': "♾",
    'symmetry_worst': "🪞",
    'fractal_dimension_worst': "🔬"
}

user_input = []
for i, feature in enumerate(feature_names):
    min_val, max_val = feature_ranges[feature]
    default_val = (min_val + max_val) / 2
    label = f"{emoji_map[feature]} {feature}"
    if i % 2 == 0:
        val = col1.slider(label, float(min_val), float(max_val), float(default_val))
    else:
        val = col2.slider(label, float(min_val), float(max_val), float(default_val))
    user_input.append(val)

# Convert to numpy array and scale
input_array = np.array(user_input).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# Predict Button
if st.button("🔍 Predict", use_container_width=True):
    prediction = mlp_clf.predict(input_scaled)[0]
    prediction_label = "Malignant" if prediction == 1 else "Benign"
    
    color = "#d6336c" if prediction_label == "Malignant" else "#2f9e44"
    icon = "❌" if prediction_label == "Malignant" else "✅"
    face = "😟" if prediction_label == "Malignant" else "😃"
    
    st.markdown(
        f"""
        <div style="text-align:center; padding:20px; border-radius:15px; background-color:{color}; color:white;">
            <h2>{icon} Prediction: <b>{prediction_label}</b> {face}</h2>
        </div>
        """, unsafe_allow_html=True
    )