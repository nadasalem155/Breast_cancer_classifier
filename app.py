import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# ------------------ Setup ------------------
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("🩺 Breast Cancer Prediction Model")

# Load breast cancer dataset (from sklearn)
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# ------------------ Emoji + Tooltip Info ------------------
emoji_map = {
    'mean radius': "📏",
    'mean texture': "🎨",
    'mean perimeter': "📐",
    'mean area': "📊",
    'mean smoothness': "✨",
    'mean compactness': "🧩",
    'mean concavity': "⬇",
    'mean concave points': "🔹",
    'mean symmetry': "🔄",
    'mean fractal dimension': "🌀"
}

slider_info = {
    'mean radius': "Average cell radius. Larger values may indicate malignancy.",
    'mean texture': "Average variation in cell texture (pixel intensity variations).",
    'mean perimeter': "Average perimeter of the cell nucleus.",
    'mean area': "Average area of the cell nucleus. Bigger nuclei often suggest malignancy.",
    'mean smoothness': "How smooth the edges of the cells are. Rough edges may indicate cancer.",
    'mean compactness': "How tightly packed the cell shapes are.",
    'mean concavity': "The severity of concave portions of the cell contours.",
    'mean concave points': "Number of concave portions in cell shapes.",
    'mean symmetry': "How symmetric the cells are. Benign tumors tend to be more symmetric.",
    'mean fractal dimension': "Complexity of cell borders. Higher values indicate irregular borders."
}

# ------------------ Sliders ------------------
st.subheader("📌 Input Tumor Features")

col1, col2 = st.columns(2)
input_data = {}

for i, feature in enumerate(list(emoji_map.keys())):
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    default_val = float(X[feature].mean())

    label = f"{emoji_map[feature]} {feature}"
    if i % 2 == 0:
        input_data[feature] = col1.slider(
            label, min_val, max_val, default_val, help=slider_info[feature]
        )
    else:
        input_data[feature] = col2.slider(
            label, min_val, max_val, default_val, help=slider_info[feature]
        )

# ------------------ Prediction ------------------
st.subheader("✅ Entered Values")
st.json(input_data)

if st.button("🔮 Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.markdown(
            f"""
            <div style="background-color:#d4edda; padding:20px; border-radius:15px; border:2px solid #28a745;">
                <h3 style="color:#155724;">🟢 Prediction: Benign (Not Cancerous)</h3>
                <p style="color:#155724; font-size:18px;">Confidence: {probability:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#f8d7da; padding:20px; border-radius:15px; border:2px solid #dc3545;">
                <h3 style="color:#721c24;">🔴 Prediction: Malignant (Cancerous)</h3>
                <p style="color:#721c24; font-size:18px;">Confidence: {probability:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )