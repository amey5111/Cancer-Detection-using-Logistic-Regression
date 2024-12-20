import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load dataset to get feature metadata (column names and preprocessing details)
@st.cache_data
def load_dataset():
    data = pd.read_csv("survey lung cancer.csv")
    X = data.drop(columns=["LUNG_CANCER"])
    return X

# Load trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("logistic_regression_model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'logistic_regression_model.pkl' is in the same directory.")
        st.stop()
    return model

# Preprocess user input
def preprocess_input(user_inputs, categorical_features, numerical_features, preprocessor):
    input_data = pd.DataFrame([user_inputs])
    # Apply preprocessing
    input_data_transformed = preprocessor.transform(input_data)
    return input_data_transformed

# Streamlit app
st.title("Lung Cancer Prediction App")
st.markdown("This application predicts the likelihood of **Lung Cancer** based on clinical and lifestyle data.")

# Load dataset and model
X = load_dataset()
categorical_features = X.select_dtypes(include=["object"]).columns
numerical_features = X.select_dtypes(exclude=["object"]).columns
model = load_model()

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
    ]
)
# Fit the preprocessor using the original dataset
preprocessor.fit(X)

# Input fields
st.header("Input Patient Data")

user_inputs = {
    "GENDER": st.selectbox("Gender", ["M", "F"]),
    "AGE": st.number_input("Age", min_value=0, max_value=120, step=1, value=30),
    "SMOKING": st.selectbox("Smoking", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
    "YELLOW_FINGERS": st.selectbox("Yellow Fingers", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
    "ANXIETY": st.selectbox("Anxiety", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
    "PEER_PRESSURE": st.selectbox("Peer Pressure", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
    "CHRONIC DISEASE": st.selectbox("Chronic Disease", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
    "FATIGUE ": st.selectbox("Fatigue", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
    "ALLERGY ": st.selectbox("Allergy", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
    "WHEEZING": st.selectbox("Wheezing", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
    "ALCOHOL CONSUMING": st.selectbox("Alcohol Consuming", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
    "COUGHING": st.selectbox("Coughing", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
    "SHORTNESS OF BREATH": st.selectbox("Shortness of Breath", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
    "SWALLOWING DIFFICULTY": st.selectbox("Swallowing Difficulty", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
    "CHEST PAIN": st.selectbox("Chest Pain", [2, 1], format_func=lambda x: "YES" if x == 2 else "NO"),
}

# Predict button
if st.button("Predict"):
    input_data = preprocess_input(user_inputs, categorical_features, numerical_features, preprocessor)
    try:
        prediction = model.predict(input_data)[0]
        st.write(f"Prediction: **{'Lung Cancer Detected' if prediction == 'YES' else 'No Lung Cancer Detected'}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
