
import streamlit as st
import pandas as pd
import joblib
import os

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load models
rf_model = joblib.load(os.path.join(current_dir, "random_forest_classifier_model.pkl"))
lin_model = joblib.load(os.path.join(current_dir, "linear_model.pkl"))

# Title
st.title("Diabetes Outcome Prediction App")
st.write("This app predicts the likelihood of diabetes based on personal health metrics.")

# Model selection
model_option = st.selectbox("Choose a Model", ["Random Forest Classifier", "Logistic Regression"])

# Input features
Age = st.number_input("Age", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
Glucose = st.number_input("Glucose", min_value=0.0)
BloodPressure = st.number_input("Blood Pressure", min_value=0.0)
Insulin = st.number_input("Insulin", min_value=0.0)
SkinThickness = st.number_input("Skin Thickness", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Pregnancies = st.number_input("Pregnancies", min_value=0)
HbA1c = st.number_input("HbA1c", min_value=0.0)
WaistCircumference = st.number_input("Waist Circumference", min_value=0.0)
PhysicalActivityLevel = st.number_input("Physical Activity Level (0–1)", min_value=0.0, max_value=1.0)
SmokingStatus = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
AlcoholConsumption = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
DietQuality = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
FamilyHistoryOfDiabetes = st.selectbox("Family History of Diabetes", ["No", "Yes"])

# Encode categorical features
SmokingStatus_encoded = {"Never": 0, "Former": 1, "Current": 2}[SmokingStatus]
AlcoholConsumption_encoded = {"None": 0, "Moderate": 1, "High": 2}[AlcoholConsumption]
DietQuality_encoded = {"Poor": 0, "Average": 1, "Good": 2}[DietQuality]
FamilyHistory_encoded = {"No": 0, "Yes": 1}[FamilyHistoryOfDiabetes]

# Create input DataFrame
input_data = pd.DataFrame([{
    "Age": Age,
    "BMI": BMI,
    "Glucose": Glucose,
    "BloodPressure": BloodPressure,
    "Insulin": Insulin,
    "SkinThickness": SkinThickness,
    "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
    "Pregnancies": Pregnancies,
    "HbA1c": HbA1c,
    "WaistCircumference": WaistCircumference,
    "PhysicalActivityLevel": PhysicalActivityLevel,
    "SmokingStatus": SmokingStatus_encoded,
    "AlcoholConsumption": AlcoholConsumption_encoded,
    "DietQuality": DietQuality_encoded,
    "FamilyHistoryOfDiabetes": FamilyHistory_encoded
}])

# Make prediction
if st.button("Predict Outcome"):
    model = rf_model if model_option == "Random Forest Classifier" else lin_model
    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0][1]
        st.success(f"Predicted Class: {prediction}")
        st.info(f"Probability of Class 1 (Positive Outcome): {proba:.2f}")
    else:
        st.success(f"Predicted Class: {prediction}")
        st.warning("Probability not available for this model.")
