import streamlit as st
import pickle
import numpy as np

# Loading the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Loan Predictor", page_icon="💰")
st.title("🏦 Real-Time Loan Eligibility Checker")

# Input form
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
ApplicantIncome = st.number_input("Applicant Income (₹)", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income (₹)", min_value=0)
LoanAmount = st.number_input("Loan Amount (₹)", min_value=10000)
Loan_Amount_Term = st.selectbox("Loan Term (in months)", [120, 180, 240, 300, 360])
Credit_History = st.selectbox("Credit History", ["Good (1.0)", "Bad (0.0)"])

# Encoding the inputs
gender = 1 if Gender == "Male" else 0
married = 1 if Married == "Yes" else 0
dependents = {"0": 0, "1": 1, "2": 2, "3+": 3}[Dependents]
credit = 1.0 if Credit_History == "Good (1.0)" else 0.0

# Featuring array for prediction
features = np.array([[gender, married, dependents,
                      ApplicantIncome, CoapplicantIncome,
                      LoanAmount, Loan_Amount_Term,
                      credit]])

# Predicting
if st.button("Check Loan Eligibility"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    total_income = ApplicantIncome + CoapplicantIncome

    if prediction == 1:
        st.success(f"✅ Loan Approved! (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Loan Denied. (Confidence: {probability:.2f})")
        st.subheader("🔍 Explanation:")
        
        if total_income <= 60000:
            st.warning("📉 Your total income is too low. Try increasing income or adding a coapplicant.")
        if credit == 0.0:
            st.warning("🚫 Your credit history is poor. A good credit history greatly improves approval chances.")
        if LoanAmount >= 200000:
            st.warning("💸 The loan amount you requested is too high based on your income.")
        if Loan_Amount_Term < 180:
            st.warning("⏳ Loan term is too short. Longer terms reduce monthly burden.")

        # Catch-all if all looks good but still denied
        if total_income > 60000 and credit == 1.0 and LoanAmount < 200000 and Loan_Amount_Term >= 180:
            st.info("ℹ️ You're close to being approved. A slight increase in income or term could help.")

