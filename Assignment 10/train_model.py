"""
train_model.py
----------------
Loads the Heart Disease dataset, trains a Random Forest classifier,
evaluates it, and saves the trained model (+ feature list) to model.pkl
using Joblib.

Run:
    python train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------------------------------------------------------------
# Task 1: Data Understanding and Preprocessing
# ---------------------------------------------------------------

# 1. Load the dataset
df = pd.read_csv("heart.csv")

print("First five records:")
print(df.head(), "\n")

# 2. Identify numerical features and target variable
target_col = "target"
feature_cols = [c for c in df.columns if c != target_col]

print(f"Numerical features ({len(feature_cols)}): {feature_cols}")
print(f"Target variable: '{target_col}'\n")

# 3. Check for missing values
print("Missing values per column:")
print(df.isnull().sum(), "\n")

# NOTE: This Kaggle release of the dataset contains a large number of exact
# duplicate rows (723 out of 1025). Leaving them in causes the same patient
# record to appear in both the train and test split, which leaks information
# and produces an artificially perfect accuracy. We drop duplicates before
# splitting so the reported accuracy reflects real generalization.
print(f"Duplicate rows found: {df.duplicated().sum()}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after removing duplicates: {df.shape}\n")

# 4. Split into 80% train / 20% test
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

# ---------------------------------------------------------------
# Task 2: Model Development
# ---------------------------------------------------------------

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy Score: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model + feature order using Joblib
joblib.dump({"model": model, "features": feature_cols}, "model.pkl")
print("Saved trained model to model.pkl")
