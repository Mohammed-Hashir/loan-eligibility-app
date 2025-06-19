import streamlit as st
import pickle
import numpy as np
import shap
import pandas as pd

# Load model and training data
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
train_df = pd.read_csv("train.csv")
train_df.dropna(inplace=True)
train_df['Dependents'] = train_df['Dependents'].replace('3+', 3).astype(int)
train_df.replace({'Gender': {'Male': 1, 'Female': 0},
                  'Married': {'Yes': 1, 'No': 0},
                  'Education': {'Graduate': 0, 'Not Graduate': 1},
                  'Self_Employed': {'Yes': 1, 'No': 0},
                  'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}}, inplace=True)
X_train = train_df.drop(columns=['Loan_ID', 'Loan_Status'])

# SHAP Explainer
explainer = shap.Explainer(model.predict, X_train)

st.title("🏦 Loan Eligibility Prediction")
st.write("Fill the form to check your loan status and see why.")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in ₹ thousands)", min_value=0)
loan_term = st.selectbox("Loan Amount Term (in months)", [360, 120, 180, 240, 300, 84, 60, 36, 12])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encoding
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
property_area = property_map[property_area]

# Feature array
features = np.array([[gender, married, dependents, education, self_employed,
                      applicant_income, coapplicant_income, loan_amount,
                      loan_term, credit_history, property_area]])

# Prediction
if st.button("Check Eligibility"):
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Denied.")

    # SHAP explanation
    input_df = pd.DataFrame(features, columns=X_train.columns)
    shap_values = explainer(input_df)

    st.subheader("📊 Why this decision was made:")
    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Input Value": input_df.iloc[0].values,
        "Impact": shap_values.values[0]
    }).sort_values(by="Impact", key=abs, ascending=False)

    st.write(shap_df.head(3))  # Top 3 influencing features
