# Heart Disease Prediction — End-to-End ML Deployment

A machine learning project that predicts whether a patient is at risk of heart disease
based on clinical parameters, exposed as a REST API using Flask and deployed live on Render.

**Live API URL:** `<< PASTE YOUR RENDER URL HERE AFTER DEPLOYMENT, e.g. https://heart-disease-api.onrender.com >>`

---

## Dataset

[Heart Disease Prediction Dataset — Kaggle (johnsmith88)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

1025 patient records, 13 clinical input features, and a binary `target` column
(`1` = heart disease present, `0` = no heart disease). Note: the raw CSV contains 723
duplicate rows; these are dropped during preprocessing to avoid train/test data leakage
(see `HeartDiseaseDeployment.ipynb` / `train_model.py`).

| Feature | Description |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy (0–3) |
| thal | Thalassemia (0–3) |

---

## 🗂 Repository Structure

```
HeartDiseaseDeployment/
│
├── app.py                      # Flask REST API
├── train_model.py              # Model training script
├── HeartDiseaseDeployment.ipynb  # Notebook: EDA, preprocessing, training, evaluation
├── model.pkl                   # Trained model (Random Forest) saved with Joblib
├── requirements.txt            # Python dependencies
├── Procfile                    # Start command for Render/Gunicorn
├── heart.csv                   # Dataset
├── README.md
├── templates/
│   └── index.html              # Simple info page served at "/"
└── static/                     # (unused / reserved for static assets)
```

---

## Model

- **Algorithm:** Random Forest Classifier (`scikit-learn`)
- **Split:** 80% train / 20% test, stratified on the target
- **Evaluation metric:** Accuracy Score (see notebook for full classification report)
- **Result:** ~0.75 accuracy on the held-out, de-duplicated test set

The model and the exact feature order it expects are bundled together in `model.pkl`
(saved with `joblib.dump({'model': ..., 'features': ...}, 'model.pkl')`) so the API can
reconstruct the correct input format at inference time.

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/HeartDiseaseDeployment.git
cd HeartDiseaseDeployment

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Retrain the model
python train_model.py

# 5. Run the API
python app.py
```

The API will start on `http://127.0.0.1:5000`.

---

## API Reference

### `GET /health`
Simple health check.

```json
{ "status": "ok" }
```

### `POST /predict`
Accepts a patient's clinical details as JSON and returns a prediction.

**Request**

```bash
curl -X POST https://<your-render-url>/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
      }'
```

**Response**

```json
{
  "prediction": "Heart Disease Detected",
  "prediction_label": 1,
  "probability": 0.83
}
```

---

## Deployment (Render)

This app is deployed on [Render](https://render.com) as a **Web Service**:

1. Push this repository to a public GitHub repo.
2. On Render: **New → Web Service** → connect the GitHub repo.
3. Settings:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (matches the `Procfile`)
4. Deploy. Render assigns a public URL and sets the `PORT` environment variable,
   which `app.py` reads automatically (`os.environ.get("PORT", 5000)`).
5. Once live, verify with:
   ```bash
   curl https://<your-render-url>/health
   ```
6. Paste the live URL at the top of this README.

> **Note:** Render's free tier spins down idle services; the first request after
> inactivity may take ~30–50 seconds to respond while the instance wakes up.

---

## Conclusion

The Random Forest model achieved solid accuracy (~0.75) on a held-out test set after
removing duplicate records from the dataset, which eliminated a data-leakage issue
present in the raw Kaggle CSV — without that fix, accuracy appeared as a misleading
100%. Key predictors of heart disease risk in this dataset include chest pain type
(`cp`), number of major vessels colored by fluoroscopy (`ca`), thalassemia status
(`thal`), and maximum heart rate achieved (`thalach`), consistent with established
clinical understanding of cardiovascular risk factors.

The main challenges during deployment were less about the model itself and more about
productionizing it: keeping the exact same feature order used in training consistent
inside the Flask API, handling malformed or incomplete JSON input gracefully, pinning
dependency versions in `requirements.txt` so the Render build environment matches the
local one, and configuring the app to bind to the `PORT` environment variable Render
assigns at runtime rather than a hardcoded port.

This highlights why MLOps matters: a model is only useful once it can be reliably
packaged, versioned, served, and monitored in a real environment. Practices such as
saving models with Joblib, pinning dependencies, exposing health-check endpoints, and
using reproducible deployment platforms like Render turn a one-off notebook experiment
into a dependable, maintainable service a healthcare organization could actually put in
front of clinicians.

---

## Disclaimer

This project is for educational purposes only and is **not** a certified medical
diagnostic tool. Predictions should not be used for real clinical decision-making.
