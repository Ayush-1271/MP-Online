# Assignment 2: Customer Churn Prediction using Logistic Regression

## 🎯 Objective

A telecommunications company wants to predict whether a customer is likely to leave (churn) based on demographic information and service usage. This project develops a **Logistic Regression** model to predict customer churn, helping the company identify at-risk customers and take proactive retention measures.

## 📊 Dataset

**Telco Customer Churn Dataset**

- **Kaggle Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Size:** 7,043 customers × 21 attributes
- **Target Variable:** `Churn` (Yes/No)

> ⚠️ The dataset is **not** included in this repository due to licensing restrictions. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the Kaggle link above and place it in the same folder as the notebook before running it.

## 📚 Libraries Used

| Library | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib / seaborn | Visualization (churn distribution, coefficients, confusion matrix) |
| scikit-learn | Preprocessing, train/test split, Logistic Regression, evaluation metrics |

## 🔬 Methodology

1. **Data Understanding** — Loaded the dataset with Pandas, displayed the first five records, and identified numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`), categorical features (gender, contract, internet service, payment method, etc.) and the target variable (`Churn`).
2. **Data Preprocessing** —
   - Detected 11 hidden missing values in `TotalCharges` (blank strings for customers with tenure = 0) and imputed them with 0.
   - Encoded binary categorical variables with label encoding and multi-category variables with one-hot encoding; dropped the `customerID` identifier.
   - Split the data into **80% training / 20% testing** (stratified) and standardized numerical features with `StandardScaler`.
3. **Model Development** — Trained a `LogisticRegression` model (`max_iter=1000`, `random_state=42`) on the training set and predicted churn on the test set.
4. **Model Evaluation** — Evaluated with Accuracy, Precision, Recall, F1-Score, a Confusion Matrix, and analyzed model coefficients to identify the drivers of churn.

## 📈 Results

| Metric | Score |
|---|---|
| **Accuracy** | 80.62% |
| **Precision (Churn)** | 0.6593 |
| **Recall (Churn)** | 0.5588 |
| **F1-Score (Churn)** | 0.6049 |

**Confusion Matrix (test set, n = 1,409):**

| | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 927 (TN) | 108 (FP) |
| **Actual: Churn** | 165 (FN) | 209 (TP) |

**Key observations:**

- Overall accuracy is good (~80%), but the class imbalance (~27% churners) means the model is weaker on the churn class — recall of ~56% shows many actual churners are missed.
- The most influential churn drivers (positive coefficients) are **month-to-month contracts**, **short tenure**, **fibre-optic internet**, and **electronic-check payment**; **two-year contracts** and **long tenure** strongly reduce churn probability.
- Recall could be improved with `class_weight='balanced'`, resampling (SMOTE), or threshold tuning — a worthwhile trade-off since missing a churner is costlier than a false alarm.

## 🏁 Conclusion

The Logistic Regression model predicts telecom customer churn with about 80% accuracy and an F1-score of ~0.60 for the churn class. Churn is driven primarily by contract flexibility and customer lifetime: customers on month-to-month contracts with short tenure, fibre-optic internet, and electronic-check payments are most likely to leave, while long-tenured customers on two-year contracts rarely churn. Retention efforts should therefore focus on new, flexible-contract customers early in their lifecycle. A key limitation of Logistic Regression here is its assumption of a linear relationship between features and the log-odds of churn — it cannot capture non-linear effects or feature interactions, which (together with class imbalance) limits recall on actual churners. Tree-based ensembles such as Random Forest or XGBoost would likely perform better.

## 📁 Repository Structure

```
Assignment 2/
├── Assignment-2.ipynb   # Complete notebook (Tasks 1–5, with outputs)
└── README.md            # This file
```

## ▶️ How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` next to the notebook.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open and run `Assignment-2.ipynb` in Jupyter.

---

*Submitted by: Ayush (23MIP10135), VIT Bhopal*
