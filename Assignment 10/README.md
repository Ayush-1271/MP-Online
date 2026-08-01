# Heart Disease Prediction

Assignment project - trained a model to predict heart disease risk from clinical data, wrapped it in a Flask API, and deployed it on Render.

**Live app:** https://heart-disease-prediction-6wln.onrender.com
(there's a `/health` route too, and `/predict` for the actual API call)

Note: Render's free tier sleeps the app after a while if nobody hits it, so the first request after that might take 30-40 seconds to wake up. That's normal, not a bug.

## Dataset

Used the Heart Disease dataset from Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

1025 rows, 13 input columns + a `target` column (1 = disease, 0 = no disease).

One thing I found while poking around the data - this CSV has 723 duplicate rows in it. If you don't drop them before splitting into train/test, you end up training and testing on the same rows basically, and the model "predicts" with 100% accuracy which is obviously fake. Took me a bit to figure out why my first run gave a perfect score. Dropped duplicates first, then split, and got a more believable ~75%.

Columns:
- age, sex, cp (chest pain type), trestbps (resting BP), chol (cholesterol)
- fbs (fasting blood sugar), restecg, thalach (max heart rate)
- exang (exercise induced angina), oldpeak, slope, ca, thal
- target (what we're predicting)

## What's in here

```
app.py                       - the Flask API
train_model.py                - trains the model, run this to regenerate model.pkl
HeartDiseaseDeployment.ipynb  - notebook with the EDA/preprocessing/training walkthrough
model.pkl                     - the saved model (joblib)
requirements.txt
Procfile                      - tells Render how to start the app
heart.csv                     - the dataset
templates/index.html          - simple form UI so you don't need Postman to test it
```

## Model

Went with Random Forest since it usually does well on this kind of tabular data without much tuning. Split 80/20, stratified on target so both sets have a similar mix of positive/negative cases.

Accuracy on the test set: ~0.75

Not going to pretend that's amazing, but it's a real number after fixing the duplicate-row issue, not the fake 1.0 you get if you skip that step.

## Running it locally

## Running Locally

```bash
git clone https://github.com/Ayush-1271/MP-Online.git
cd MP-Online/"Assignment 10"
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in a browser, fill the form, hit predict.

If you want to retrain the model yourself: `python train_model.py`

## API

`GET /health` -> `{"status": "ok"}`, mostly just so Render/uptime checks have something to ping.

`POST /predict` - send patient values as JSON, get a prediction back.

```bash
curl -X POST https://heart-disease-prediction-6wln.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```

returns something like:

```json
{
  "prediction": "No Heart Disease Detected",
  "prediction_label": 0,
  "probability": 0.88
}
```

## Deploying on Render (in case I need to redo it)

- New Web Service, connect the GitHub repo
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Render sets a PORT env var automatically, app.py already reads it

## Conclusion

The model ends up around 75% accuracy after cleaning up the duplicate rows, which is a decent result for a basic Random Forest with no real tuning. Features like chest pain type, number of major vessels (ca), and max heart rate showed up as the most important ones, which lines up with what you'd expect medically.

Deployment was honestly more annoying than the ML part. Getting the same column order in the API as during training, making sure requirements.txt versions actually matched what Render installs, and remembering that Render assigns its own PORT instead of using 5000 all tripped me up at some point. Also the free tier spinning down when idle threw me off at first since it looked broken when it was just asleep.

Doing this end to end (not just training a model in a notebook and calling it done) made it pretty clear why MLOps is its own thing - a model sitting in a notebook doesn't help anyone, it actually has to run somewhere reliably and be callable by other things.

---
Built for a college assignment. Not an actual medical tool, obviously.
