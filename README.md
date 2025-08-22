# 🩺 Breast Cancer Prediction (MLP Classifier)

This is a **Breast Cancer Classification project** using multiple classifiers (MLP, Random Forest, Decision Tree, XGBoost, Gradient Boosting, etc.) with a deployed **Streamlit web app** based on the MLP model.  

🌐 **Try the App Online:** [Breast Cancer Prediction App](https://breast-cancer-app55.streamlit.app/)

The project includes data preprocessing, exploratory data analysis (EDA), feature selection, training multiple models, and deploying the best model using Streamlit.

---

## 📊 Dataset

- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset  
- **Features:** 30 numeric features including radius, texture, smoothness, compactness, symmetry, and fractal dimension  
- **Target:** `diagnosis` (Malignant = 🔴 M, Benign = 🟢 B)

---

## 🛠️ Project Workflow

### 1️⃣ Data Cleaning
- Removed unnecessary columns (`id` and `Unnamed:32`) ❌  
- Checked for missing values (none found) ✅  
- Verified no duplicates exist 🔍  

### 2️⃣ Outlier Handling
- Detected outliers using **Boxplots** 📦  
- Replaced extreme values with boundary values to reduce skewness 🔄  

### 3️⃣ Exploratory Data Analysis (EDA)
- Correlation Heatmap 🌡️ for numeric features  
- Review of Data Types and Unique Values 📝  
- Count Plot for target distribution: 🟢 B = 357, 🔴 M = 212  

### 4️⃣ Feature Selection
- Removed highly correlated features (correlation ≥ 0.8) ⚡ to reduce multicollinearity  
- Selected 13 final features for modeling ✨  

### 5️⃣ Normalization
- Applied Min-Max Scaling to all features (range [0,1]) 🔧  

### 6️⃣ Train/Test Split
- 80% training, 20% testing 🏋️  
- `random_state=42` for reproducibility 🔄  

### 7️⃣ Model Training & Evaluation
- Trained multiple classifiers:
  - **Random Forest 🌲** → Test Accuracy: 93.86%, F1 Score: 0.916  
  - **Decision Tree 🌳** → Test Accuracy: 92.11%, F1 Score: 0.897  
  - **XGBoost 🚀** → Test Accuracy: 95.61%, F1 Score: 0.941  
  - **Gradient Boosting ⬆️** → Test Accuracy: 94.74%, F1 Score: 0.930  
  - **Perceptron (Single-layer NN 🧠)** → Test Accuracy: 94.74%, F1 Score: 0.929  
  - **MLP (Multi-layer NN 🏆)** → Test Accuracy: 96.49%, F1 Score: 0.953 (Best Model)  

### 8️⃣ Model Deployment
- Deployed using **Streamlit 💻**  
- Features input via **sliders 🎚️** with descriptive tooltips  
- Output shows predicted class 🔴🟢 and probabilities 📊  

---

## 🚀 How to Run the App

1. Clone the repository 📂  
2. Install dependencies:
```bash
pip install -r requirements.txt

3. Run the Streamlit app:



streamlit run app.py

4. Use the sliders to input feature values and predict the tumor type 🎯




---

💾 Saved Models

mlp_model.pkl → Trained MLP classifier 🧠

scaler.pkl → MinMaxScaler for normalization 🔧



---
## 📄 Presentation

🌐 **View Project Presentation:** [Breast Cancer Prediction Presentation](https://your-presentation-link.com)