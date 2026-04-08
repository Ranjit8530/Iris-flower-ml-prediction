import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Iris Classifier", layout="centered")

# Title
st.title("🌸 Iris Flower Prediction App")

st.write("Enter the flower measurements below:")

# Inputs (same order as training)
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Prediction button
if st.button("Predict"):
    
    # Create input array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Apply scaling
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # Class labels
    classes = ["Setosa", "Versicolor", "Virginica"]

    # Output
    st.success(f"Prediction: {classes[prediction]}")
    
    st.write("### Prediction Probabilities")
    for i, prob in enumerate(probabilities):
        st.write(f"{classes[i]}: {prob:.2f}")