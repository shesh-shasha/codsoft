import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("fraudTest.csv")

# Select useful columns
df = df[['amt', 'city_pop', 'is_fraud']]

# Features and labels
X = df[['amt', 'city_pop']]
y = df['is_fraud']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Fraud Detection Accuracy:", accuracy)

# Sample prediction
sample = [[2500, 50000]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Prediction: Fraud Transaction")
else:
    print("Prediction: Legitimate Transaction")