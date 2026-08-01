"""
Trains a model on the heart disease dataset and saves it to model.pkl.
Run this with: python train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---- load + look at the data ----

df = pd.read_csv("heart.csv")

print("First 5 rows:")
print(df.head(), "\n")

target_col = "target"
feature_cols = [c for c in df.columns if c != target_col]

print(f"Features: {feature_cols}")
print(f"Target: {target_col}\n")

print("Missing values:")
print(df.isnull().sum(), "\n")

# this dataset (the kaggle version) has a ton of exact duplicate rows.
# if you don't drop them, the same row can land in both train and test,
# which basically leaks the answer and gives a fake 100% accuracy.
# found this out the hard way when my first run gave 1.0 accuracy lol
print(f"Duplicate rows: {df.duplicated().sum()}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after dropping duplicates: {df.shape}\n")

# ---- split ----

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

# ---- train ----

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred))

# save model + the feature order it expects, app.py needs both
joblib.dump({"model": model, "features": feature_cols}, "model.pkl")
print("Saved model.pkl")
