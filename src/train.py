import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

os.makedirs("models", exist_ok=True)

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"Accuracy: {accuracy:.4f}")
