### 📘 Loan Eligibility Predictor (Streamlit App)

This is a real-time loan approval prediction app built using Streamlit and Machine Learning on a dataset of key financial indicators.

The app allows users to check if their loan application would be approved based on inputs like income, loan amount, credit history, etc. It also provides clear SHAP explanations for approvals or rejections — showing exactly which features impacted the result most.

### 🚀 Live Demo

### 🌐 [Click to Use the App](https://loan-eligibility-app-000.streamlit.app/)

### 💡 Features
- ✅ Predict loan approval using a trained Random Forest model
- 📋 Clean form input with Streamlit UI
- 📊 SHAP-based interpretability — see the top 3 factors influencing your approval decision
- 🔒 Logical rules based on income, credit score, and loan term
- 🧮 Loan amount entered in ₹ thousands (e.g., 120 = ₹120,000)

### 📊 Prediction Logic
The model was trained on real dataset constraints and performs based on:
- ✅ Total income > ₹60,000
- ✅ Credit history = 1.0 (good)
- ✅ Loan amount < ₹200,000
- ✅ Loan term ≥ 180 months
  
### 🗂 Files Included
File   	Description     
app.py	Main Streamlit web app
loan_model.py	Trains model using cleaned dataset, saves model.pkl
model.pkl	Final trained ML model
train.csv	Original training data from real dataset
requirements.txt	All Python dependencies for deployment

### 🧠 SHAP Explainability (New!)
This version integrates SHAP (SHapley Additive exPlanations) to help users understand their predictions:
> After each prediction, the app displays the top 3 features that influenced your approval/rejection.
This improves transparency and makes the app ideal for projects, reports, or demos.

### 👨‍💻 Built With
- Python
- scikit-learn
- Pandas & NumPy
- Streamlit
- SHAP

### 📬 Contact
Feel free to connect with me on [LinkedIn](www.linkedin.com/in/mohammed-hashir-99793428a) or \[[ smdhashir2006@gmail.com](mailto:smdhashir2006@gmail.com)]
