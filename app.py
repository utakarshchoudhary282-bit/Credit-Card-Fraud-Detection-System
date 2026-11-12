"""
Credit Card Fraud Detection — Streamlit app
Run:  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pipeline as P

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")


# ---------------------------------------------------------------- cache
@st.cache_data(show_spinner="Loading dataset...")
def get_data(file_bytes: bytes | None = None) -> pd.DataFrame:
    if file_bytes is not None:
        import io
        return pd.read_csv(io.BytesIO(file_bytes))
    return P.load_data()


@st.cache_resource(show_spinner="Loading model...")
def get_model(_stamp: float):
    return P.load_model()


def model_stamp() -> float:
    import os
    return os.path.getmtime(P.MODEL_PATH) if os.path.exists(P.MODEL_PATH) else 0.0


# ---------------------------------------------------------------- sidebar
st.sidebar.title("💳 Fraud Detection")
page = st.sidebar.radio("Go to", ["📊 Data & EDA", "🧠 Train & Evaluate",
                                  "🔍 Predict Transaction", "📁 Batch Prediction"])
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Optional: upload dataset CSV", type="csv",
                                    help="Leave empty to auto-download from Kaggle")

try:
    df = get_data(uploaded.getvalue() if uploaded else None)
except Exception as e:
    st.error(f"Could not load dataset: {e}\n\nUpload the CSV from the sidebar "
             "or set up Kaggle credentials (`~/.kaggle/kaggle.json`).")
    st.stop()

model, meta = get_model(model_stamp())
st.sidebar.success("Model loaded ✅") if model else st.sidebar.warning("No model yet — train one")


# ================================================================ EDA
if page == "📊 Data & EDA":
    st.title("📊 Credit Card Transactions — EDA")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Fraud cases", f"{int(df[P.TARGET].sum()):,}")
    c4.metric("Fraud rate", f"{df[P.TARGET].mean():.2%}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Preview", "Target", "Distributions", "Category vs Amount", "Correlation"])

    with tab1:
        st.dataframe(df.head(50), width="stretch")
        a, b = st.columns(2)
        a.subheader("Statistics"); a.dataframe(df.describe().T, width="stretch")
        b.subheader("Missing values"); b.dataframe(df.isnull().sum().rename("missing"),
                                                   width="stretch")
        st.subheader("Skewness")
        st.dataframe(df[P.NUM_COLS + [P.TARGET]].skew().rename("skew").to_frame().T,
                     width="stretch")

    with tab2:
        vc = df[P.TARGET].map({0: "Legit", 1: "Fraud"}).value_counts().reset_index()
        vc.columns = ["class", "count"]
        st.plotly_chart(px.bar(vc, x="class", y="count", color="class", text="count",
                               title="Class balance (is_fraud)"), width="stretch")
        st.info("Dataset is heavily imbalanced → models use `class_weight='balanced'` "
                "and are scored with macro-F1, not just accuracy.")

    with tab3:
        col = st.selectbox("Feature", P.NUM_COLS)
        split = st.checkbox("Split by is_fraud", value=True)
        fig = px.histogram(df, x=col, nbins=50,
                           color=df[P.TARGET].map({0: "Legit", 1: "Fraud"}) if split else None,
                           barmode="overlay", opacity=0.7, title=f"Distribution of {col}")
        st.plotly_chart(fig, width="stretch")
        st.plotly_chart(px.histogram(df, x="merchant_category", title="Merchant category count"),
                        width="stretch")

    with tab4:
        st.plotly_chart(px.box(df, x="merchant_category", y="amount",
                               color=df[P.TARGET].map({0: "Legit", 1: "Fraud"}),
                               title="Amount by merchant category & fraud"),
                        width="stretch")
        rate = df.groupby("merchant_category")[P.TARGET].mean().sort_values().reset_index()
        st.plotly_chart(px.bar(rate, x="merchant_category", y=P.TARGET,
                               title="Fraud rate by merchant category"), width="stretch")

    with tab5:
        corr = df[P.NUM_COLS + [P.TARGET]].corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, title="Correlation matrix", height=650),
                        width="stretch")


# ================================================================ TRAIN
elif page == "🧠 Train & Evaluate":
    st.title("🧠 Train & Evaluate")
    st.write("Pipeline: **clip outliers (1–99 pct) → stratified split (67/33) → "
             "StandardScaler + OneHotEncoder → compare 4 models (3-fold CV) → "
             "GridSearch Random Forest → save pipeline**")

    fast = st.toggle("Fast mode (skip GridSearch, use notebook's best params)", value=True)
    if st.button("🚀 Train model", type="primary"):
        logbox = st.empty(); lines = []
        def log(m):
            lines.append(m); logbox.code("\n".join(lines))
        with st.spinner("Training... (GridSearch can take a few minutes)"):
            P.train(df, run_grid=not fast, log=log)
        get_model.clear()
        st.success("Training complete!")
        st.rerun()

    if meta:
        st.subheader("Model comparison (3-fold CV on train set)")
        comp = pd.DataFrame(meta["cv_comparison"]).T.reset_index(names="model")
        st.plotly_chart(px.bar(comp.melt(id_vars="model"), x="model", y="value",
                               color="variable", barmode="group", text_auto=".3f"),
                        width="stretch")

        st.subheader("Tuned Random Forest — test set")
        tm = meta["test_metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{tm['accuracy']:.3f}")
        c2.metric("Macro F1", f"{tm['f1_macro']:.3f}")
        c3.metric("ROC AUC", f"{tm['roc_auc']:.3f}")
        st.caption(f"Best params: `{meta['best_params']}`")

        a, b = st.columns(2)
        cm = tm["confusion_matrix"]
        a.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                 x=["Pred Legit", "Pred Fraud"], y=["Legit", "Fraud"],
                                 title="Confusion matrix"), width="stretch")
        fi = pd.Series(meta["feature_importance"]).sort_values()
        b.plotly_chart(px.bar(fi, orientation="h", title="Feature importance"),
                       width="stretch")

        rep = pd.DataFrame(tm["report"]).T
        st.dataframe(rep.style.format("{:.3f}"), width="stretch")
    else:
        st.info("No trained model found. Click **Train model**.")


# ================================================================ PREDICT ONE
elif page == "🔍 Predict Transaction":
    st.title("🔍 Score a single transaction")
    if not model:
        st.warning("Train a model first (🧠 Train & Evaluate)."); st.stop()

    with st.form("txn"):
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input("Amount", 0.0, 100000.0, 120.0, step=10.0)
        hour = c1.slider("Transaction hour", 0, 23, 14)
        category = c1.selectbox("Merchant category", meta["merchant_categories"])
        foreign = c2.selectbox("Foreign transaction", [0, 1], format_func=lambda v: "Yes" if v else "No")
        mismatch = c2.selectbox("Location mismatch", [0, 1], format_func=lambda v: "Yes" if v else "No")
        trust = c2.slider("Device trust score", 0, 100, 70)
        velocity = c3.number_input("Transactions in last 24h", 0, 50, 1)
        age = c3.slider("Cardholder age", 18, 90, 35)
        threshold = c3.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "amount": amount, "transaction_hour": hour, "merchant_category": category,
            "foreign_transaction": foreign, "location_mismatch": mismatch,
            "device_trust_score": trust, "velocity_last_24h": velocity, "cardholder_age": age}])
        p = float(P.predict(model, meta, row)["fraud_probability"].iloc[0])

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": "Fraud probability"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "crimson" if p >= threshold else "seagreen"},
                   "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100}}))
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")
        if p >= threshold:
            st.error(f"🚨 FRAUD — probability {p:.1%} ≥ threshold {threshold:.0%}")
        else:
            st.success(f"✅ LEGIT — probability {p:.1%} < threshold {threshold:.0%}")


# ================================================================ BATCH
elif page == "📁 Batch Prediction":
    st.title("📁 Batch prediction")
    if not model:
        st.warning("Train a model first (🧠 Train & Evaluate)."); st.stop()

    st.write("Upload a CSV with columns: `" + "`, `".join(P.FEATURES) + "`")
    f = st.file_uploader("Transactions CSV", type="csv", key="batch")
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    src = pd.read_csv(f) if f else None
    if src is None and st.checkbox("Use 200 random rows from the dataset as a demo"):
        src = df.sample(200, random_state=0)

    if src is not None:
        missing = [c for c in P.FEATURES if c not in src.columns]
        if missing:
            st.error(f"Missing columns: {missing}"); st.stop()
        res = P.predict(model, meta, src)
        res["prediction"] = (res["fraud_probability"] >= threshold).astype(int)
        c1, c2 = st.columns(2)
        c1.metric("Transactions", len(res))
        c2.metric("Flagged as fraud", int(res["prediction"].sum()))
        st.dataframe(res.sort_values("fraud_probability", ascending=False)
                     .style.background_gradient(subset=["fraud_probability"], cmap="Reds"),
                     width="stretch")
        st.download_button("⬇️ Download results", res.to_csv(index=False),
                           "fraud_predictions.csv", "text/csv")
