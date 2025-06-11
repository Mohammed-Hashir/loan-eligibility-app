import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Setting random seed for consistent results
np.random.seed(42)
n = 500  # number of records

# Creating dummy dataset with only 8 features
data = pd.DataFrame({
    'Gender': np.random.randint(0, 2, n),
    'Married': np.random.randint(0, 2, n),
    'Dependents': np.random.randint(0, 4, n),
    'ApplicantIncome': np.random.randint(20000, 120000, n),
    'CoapplicantIncome': np.random.randint(0, 50000, n),
    'LoanAmount': np.random.randint(50000, 300000, n),
    'Loan_Amount_Term': np.random.choice([120, 180, 240, 300, 360], n),
    'Credit_History': np.random.choice([0.0, 1.0], n, p=[0.3, 0.7]),
})

# Defining custom approval logic
def approve(row):
    total_income = row['ApplicantIncome'] + row['CoapplicantIncome']
    return int(
        total_income > 60000 and
        row['Credit_History'] == 1.0 and
        row['LoanAmount'] < 200000 and
        row['Loan_Amount_Term'] >= 180
    )

# Applying logic
data['Loan_Status'] = data.apply(approve, axis=1)

# Splitting features and target
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Saving the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained on simplified dummy data and saved as model.pkl")
