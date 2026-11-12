# Credit Card Fraud Detection — Streamlit App

## Run locally
```bash
cd credit_fraud_app
python -m venv venv
# Windows: venv\Scripts\activate    |  Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

# (optional) pre-train from terminal — otherwise use the "Train" page in the app
python pipeline.py --fast          # quick: notebook's best RF params
python pipeline.py                 # full GridSearch (slower)
python pipeline.py --csv my.csv    # use a local CSV instead of Kaggle

streamlit run app.py               # opens http://localhost:8501
```

## Dataset
Auto-downloaded via `kagglehub` (miadul/credit-card-fraud-detection-dataset) and cached to `data/`.
If download fails: set up `~/.kaggle/kaggle.json`, OR download the CSV manually and upload it in the app sidebar.

## Pages
- **Data & EDA** – stats, missing values, skew, class balance, distributions, boxplots, correlation
- **Train & Evaluate** – 4-model CV comparison, tuned Random Forest, confusion matrix, feature importance
- **Predict Transaction** – form + fraud-probability gauge with adjustable threshold
- **Batch Prediction** – upload CSV, score all rows, download results

## Files
- `pipeline.py` – data load, outlier clipping, preprocessing, training, saving, prediction
- `app.py` – Streamlit UI
- `model_pipeline.pkl` / `model_meta.json` – created after training
