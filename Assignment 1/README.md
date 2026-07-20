# Assignment 1 – Medical Insurance Cost Prediction (Multiple Linear Regression)

Hi! This is my first assignment for the internship. The task was to build a Multiple
Linear Regression model that predicts medical insurance charges based on a person's
details. I did the whole thing in Python (mainly pandas and scikit-learn) and I've
written down my steps and results below.

## Objective
An insurance company wants to estimate how much a customer's medical insurance will
cost based on their personal and health information. So the goal here is to build a
Multiple Linear Regression model that takes inputs like age, sex, BMI, number of
children, smoker status and region, and predicts the **charges** (the insurance cost).

## Dataset Link
Medical Cost Personal Insurance Dataset (from Kaggle):
https://www.kaggle.com/datasets/mirichoi0218/insurance

It has 1,338 rows and 7 columns:

- `age`, `bmi`, `children` → numbers
- `sex`, `smoker`, `region` → categories
- `charges` → the value we want to predict (target)

I did not upload the CSV to GitHub because the assignment said not to, so I added it
to `.gitignore`. To run my code, just download `insurance.csv` from the Kaggle link
above and keep it in the same folder as the notebook.

## Libraries Used
- **pandas** – for reading the CSV and handling the data
- **numpy** – for some basic number stuff
- **scikit-learn** – for the model (`LinearRegression`), splitting the data and the
  evaluation metrics
- **matplotlib** – for plotting actual vs predicted charges

## Methodology
Here's basically what I did, step by step:

1. **Understanding the data** – loaded the dataset with pandas, looked at the first 5
   rows with `.head()`, and figured out which columns are numerical, which are
   categorical, and which one is the target.
2. **Preprocessing** – checked for missing values (there were none), then converted the
   categorical columns (`sex`, `smoker`, `region`) into numbers using one-hot encoding
   (`pd.get_dummies` with `drop_first=True` so I don't get the dummy variable trap).
   After that I split the data into 80% training and 20% testing.
3. **Building the model** – trained a `LinearRegression` model on all six features and
   used it to predict the charges for the test data.
4. **Evaluating** – checked how good the model is using MAE, MSE and R² score, and made
   an actual vs predicted scatter plot.
5. **Conclusion** – wrote down what I learned from the results.

## Results
These are the numbers I got on the test set (268 records):

| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | ~4,181 |
| MSE (Mean Squared Error)  | ~33,596,916 |
| RMSE (Root Mean Squared Error) | ~5,796 |
| R² Score | ~0.78 |

So the R² is about **0.78**, which means the model explains around 78% of the variation
in insurance charges. That's actually pretty decent for a simple linear model.

When I looked at the model's coefficients, **smoker** had by far the biggest effect on
the cost (being a smoker adds a huge amount), followed by **age** and **BMI**. Things
like number of children, sex and region didn't matter as much.

In the `actual_vs_predicted.png` plot, most points sit close to the red diagonal line
(which would be a perfect prediction). But you can clearly see two groups of points – the
smokers are the higher ones and the model struggles a bit more with them.

## Conclusion
Overall the Multiple Linear Regression model works reasonably well and gives an R² of
around 0.78. The most important factors affecting insurance charges came out to be
smoking status, age and BMI – which honestly makes sense, since older people and smokers
usually have higher medical costs.

One limitation I noticed is that linear regression assumes every feature affects the cost
independently and in a straight line. But in real life, being a smoker AND having a high
BMI together pushes the cost up much more than either one alone (this is an "interaction"
that a plain linear model can't capture). Because of this, the model under-predicts for
some of the very high-cost customers. To improve it, I could try adding interaction terms
or use a more advanced model like Random Forest or Gradient Boosting.

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib
# put insurance.csv (from Kaggle) in this folder, then:
jupyter notebook Assignment-1.ipynb
# or just run the script:
python Assignment-1.py
```
