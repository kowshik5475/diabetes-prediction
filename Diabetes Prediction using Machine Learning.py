import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r"D:\pro\pima-indians-diabetes.csv", header=None)
data.columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Features & target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test input
sample =  pd.DataFrame([[2,120,70,20,79,25.0,0.5,30]], columns=X.columns)
result = model.predict(sample)

if result[0] == 1:
    print("Diabetes Detected")
else:
    print("No Diabetes")