import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="vinaykumartv/tourism-project-model", filename="best_tourism_project_model.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Visit With Us")
st.write("The Visit with Us App is an internal tool for staff that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them")
st.write("Kindly enter the customer details to check whether they are likely to purchase the newly introduced Wellness Tourism Package.")

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact(The method by which the customer was contacted)", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier (The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3))", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("Occupation (Customer's occupation)", ["Salaried", "Freelancer"])
Gender = st.selectbox("Gender (Customer's gender)", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of People Visiting (Total number of people accompanying the customer on the trip)", min_value=1, value=1)
PreferredPropertyStar = st.number_input("Preferred Property Star (Customer's preferred hotel rating)", min_value=1, max_value=5, value=3)
MaritalStatus = st.selectbox("Marital Status (Customer's marital status)", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips (Average number of trips the customer takes annually)", min_value=1, value=1)
Passport = st.selectbox("Passport (Whether the customer holds a valid passport)", ["Yes", "No"])
OwnCar = st.selectbox("Own Car (Whether the customer owns a car)", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (Number of children below age 5 accompanying the customer)", min_value=0, value=0)
Tenure = st.number_input("Tenure (number of years the customer has been with the bank)", value=12)
Designation = st.selectbox("Designation (Customer's designation in their current organization)", ["Executive", "Managerial", "Professional", "Other"])
MonthlyIncome = st.number_input("Monthly Income (Customer's gross monthly income)", min_value=0.0, value=5000.0)

# Customer Interaction Data
PitchSatisfactionScore= st.number_input("Pitch Satisfaction Score (Customer's satisfaction with the sales pitch)", min_value=1, max_value=5, value=3)
ProductPitched = st.selectbox("Product Pitched (Customer's type of product pitched)", ["Basic", "Standard", "Premium"]) 
NumberOfFollowups = st.number_input("Number of Followups (Total number of follow-ups by the salesperson after the sales pitch)", min_value=0, value=0)
DurationOfPitch = st.number_input("Duration of Pitch (Duration of the sales pitch delivered to the customer)", min_value=1, value=1)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch,
    'Tenure': Tenure
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase a package" if prediction == 1 else "not purchase a package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
