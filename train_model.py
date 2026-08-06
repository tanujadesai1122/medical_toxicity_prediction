import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/toxicity.csv")

# Features
X = data[[
    "MolecularWeight",
    "LogP",
    "HBD",
    "HBA",
    "RotatableBonds",
    "TPSA"
]]

# Target
y = data["Toxicity"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model
with open("model/toxicity_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")