"""
Shared ML logic for the Credit Card Fraud project.
Mirrors the notebook: clip outliers (1st-99th pct) -> stratified split ->
ColumnTransformer (StandardScaler + OneHotEncoder) -> model comparison ->
GridSearch on RandomForest -> save pipeline.

Run directly to train:
    python pipeline.py                     # downloads dataset via kagglehub
    python pipeline.py --csv path/to.csv   # use a local CSV
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model_pipeline.pkl")
META_PATH = os.path.join(HERE, "model_meta.json")
DATA_CACHE = os.path.join(HERE, "data", "credit_card_fraud.csv")

TARGET = "is_fraud"
DROP_COLS = ["transaction_id"]
CLIP_COLS = ["amount", "transaction_hour", "device_trust_score",
             "velocity_last_24h", "cardholder_age"]
NUM_COLS = ["amount", "transaction_hour", "foreign_transaction",
            "location_mismatch", "device_trust_score", "velocity_last_24h",
            "cardholder_age"]
CAT_COLS = ["merchant_category"]
FEATURES = NUM_COLS + CAT_COLS


# ---------------------------------------------------------------- data
def load_data(csv_path: str | None = None) -> pd.DataFrame:
    """Load from given CSV, else local cache, else download via kagglehub."""
    if csv_path:
        return pd.read_csv(csv_path)
    if os.path.exists(DATA_CACHE):
        return pd.read_csv(DATA_CACHE)
    import kagglehub
    path = kagglehub.dataset_download("miadul/credit-card-fraud-detection-dataset")
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(path, csv_file))
    os.makedirs(os.path.dirname(DATA_CACHE), exist_ok=True)
    df.to_csv(DATA_CACHE, index=False)
    return df


def clip_bounds(df: pd.DataFrame) -> dict:
    return {c: (float(df[c].quantile(0.01)), float(df[c].quantile(0.99)))
            for c in CLIP_COLS}


def apply_clip(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    df = df.copy()
    for c, (lo, hi) in bounds.items():
        if c in df:
            df[c] = np.clip(df[c], lo, hi)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), NUM_COLS),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


# ---------------------------------------------------------------- train
def train(df: pd.DataFrame, run_grid: bool = True, log=print) -> dict:
    df = df.drop(columns=[c for c in DROP_COLS if c in df])
    bounds = clip_bounds(df)
    df = apply_clip(df, bounds)

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y)

    # 1) compare models with 3-fold CV
    comparison = {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", model)])
        acc = cross_val_score(pipe, X_tr, y_tr, cv=3, scoring="accuracy").mean()
        f1 = cross_val_score(pipe, X_tr, y_tr, cv=3, scoring="f1_macro").mean()
        comparison[name] = {"accuracy": round(float(acc), 4), "f1_macro": round(float(f1), 4)}
        log(f"{name:20s} acc={acc:.3f}  f1_macro={f1:.3f}")

    # 2) tune Random Forest (as in notebook)
    base = Pipeline([("preprocessor", build_preprocessor()),
                     ("classifier", RandomForestClassifier(class_weight="balanced",
                                                           random_state=42, n_jobs=-1))])
    if run_grid:
        grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [8, 12, 18, None],
            "classifier__min_samples_split": [2, 5, 10],
        }
        search = GridSearchCV(base, grid, cv=3, scoring="f1_macro", n_jobs=-1)
        search.fit(X_tr, y_tr)
        best, best_params = search.best_estimator_, search.best_params_
    else:  # fast path: notebook's best params
        best_params = {"classifier__max_depth": 8, "classifier__min_samples_split": 10,
                       "classifier__n_estimators": 200}
        best = base.set_params(**best_params).fit(X_tr, y_tr)
    log(f"Best params: {best_params}")

    # 3) evaluate on held-out test set
    pred = best.predict(X_te)
    proba = best.predict_proba(X_te)[:, 1]
    test_metrics = {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "report": classification_report(y_te, pred, output_dict=True),
    }
    log(f"Test: acc={test_metrics['accuracy']}  f1_macro={test_metrics['f1_macro']}  "
        f"roc_auc={test_metrics['roc_auc']}")

    # 4) save
    importances = _feature_importance(best)
    meta = {"clip_bounds": bounds, "cv_comparison": comparison,
            "best_params": {k: (v if v is not None else "None") for k, v in best_params.items()},
            "test_metrics": test_metrics, "feature_importance": importances,
            "merchant_categories": sorted(df["merchant_category"].unique().tolist())}
    joblib.dump(best, MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Saved -> {MODEL_PATH}")
    return meta


def _feature_importance(pipe: Pipeline) -> dict:
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    imp = pipe.named_steps["classifier"].feature_importances_
    names = [n.split("__", 1)[-1] for n in names]
    return dict(sorted(zip(names, map(float, imp)), key=lambda t: -t[1]))


def load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        return None, None
    with open(META_PATH) as f:
        return joblib.load(MODEL_PATH), json.load(f)


def predict(model, meta, X: pd.DataFrame) -> pd.DataFrame:
    Xc = apply_clip(X[FEATURES], meta["clip_bounds"])
    out = X.copy()
    out["fraud_probability"] = model.predict_proba(Xc)[:, 1]
    out["prediction"] = model.predict(Xc)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Path to local dataset CSV")
    ap.add_argument("--fast", action="store_true", help="Skip GridSearch, use notebook best params")
    a = ap.parse_args()
    train(load_data(a.csv), run_grid=not a.fast)
