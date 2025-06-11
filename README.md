## 📘 Loan Eligibility Predictor (Streamlit App)

This is a real-time loan approval prediction app built using **Streamlit** and **Machine Learning** with a custom logic-based dataset.

The app allows users to check if their loan application would be approved based on key inputs like income, loan amount, credit history, etc. It also provides clear explanations for approvals or rejections.

---

### 🚀 Live Demo

🌐 [Click to Use the App](https://loan-eligibility-app-123.streamlit.app/)

---

### 💡 Features

* Predict loan approval using a trained Random Forest model
* Custom dummy dataset with logical approval rules
* Input form with clean UI (Streamlit)
* Detailed explanations for loan rejection
* Confidence score shown with every result

---

### 📊 Prediction Logic

The model was trained using logical conditions:

* ✅ Total income > ₹60,000
* ✅ Credit history is good (1.0)
* ✅ Loan amount < ₹200,000
* ✅ Loan term ≥ 180 months

---

### 🗂 Files Included

| File               | Description                                               |
| ------------------ | --------------------------------------------------------- |
| `app.py`           | Main Streamlit web app                                    |
| `loan_model.py`    | Generates dummy data, trains model, and saves `model.pkl` |
| `model.pkl`        | Trained machine learning model                            |
| `requirements.txt` | Python dependencies for deployment                        |

---

### 👨‍💻 Built With

* Python
* Scikit-learn
* Pandas & NumPy
* Streamlit

---

### 📬 Contact

Feel free to connect with me on [LinkedIn](www.linkedin.com/in/mohammed-hashir-99793428a) or \[[email@example.com](mailto:smdhashir2006@gmail.com)]


