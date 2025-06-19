import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Loading dataset
data = pd.read_csv("train.csv")

# Dropping rows with missing values
data.dropna(inplace=True)

# Dropping Loan_ID
if 'Loan_ID' in data.columns:
    data.drop("Loan_ID", axis=1, inplace=True)

# Replacing '3+' with 3 in Dependents
data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)

# Encoding categorical columns
label_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Splitting features and target
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training RandomForest model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluating
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Saving the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")
