"""
Credit Card Fraud Detection — Live Dashboard
Run:  streamlit run app.py
"""
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pipeline as P

st.set_page_config(page_title="FraudShield · Live Detector", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {padding: 18px 24px; border-radius: 16px; color: #fff; margin-bottom: 14px;
       background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #7c3aed 100%);}
.hero h1 {margin: 0; font-size: 1.9rem;}
.hero p {margin: 4px 0 0; opacity: .85;}
.card {border-radius: 14px; padding: 14px 18px; border: 1px solid rgba(128,128,128,.25);
       background: rgba(128,128,128,.06); height: 100%;}
.card .lbl {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7;}
.card .val {font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.verdict {border-radius: 14px; padding: 14px 18px; color: #fff; font-weight: 700;
          font-size: 1.5rem; text-align: center;}
.chip {display: inline-block; padding: 5px 12px; margin: 3px 4px 3px 0; border-radius: 999px;
       font-size: .85rem; font-weight: 600;}
.chip.bad {background: rgba(220,38,38,.15); color: #dc2626; border: 1px solid rgba(220,38,38,.4);}
.chip.ok  {background: rgba(22,163,74,.12); color: #16a34a; border: 1px solid rgba(22,163,74,.35);}
</style>
""", unsafe_allow_html=True)

GREEN, AMBER, RED = "#16a34a", "#f59e0b", "#dc2626"


# ---------------------------------------------------------------- data / model
@st.cache_data(show_spinner="Loading dataset...")
def get_data(file_bytes: bytes | None = None) -> pd.DataFrame:
    if file_bytes is not None:
        return pd.read_csv(io.BytesIO(file_bytes))
    return P.load_data()


@st.cache_resource(show_spinner="Loading model...")
def get_model(_stamp: float):
    return P.load_model()


def model_stamp() -> float:
    return os.path.getmtime(P.MODEL_PATH) if os.path.exists(P.MODEL_PATH) else 0.0


@st.cache_data
def reference_stats(df: pd.DataFrame) -> dict:
    """Typical legit profile + fraud rates used for explanations."""
    legit = df[df[P.TARGET] == 0]
    typical = {c: float(legit[c].median()) for c in P.NUM_COLS}
    typical["merchant_category"] = legit["merchant_category"].mode()[0]
    return {
        "typical": typical,
        "cat_rate": df.groupby("merchant_category")[P.TARGET].mean().to_dict(),
        "hour_rate": df.groupby("transaction_hour")[P.TARGET].mean().to_dict(),
        "overall_rate": float(df[P.TARGET].mean()),
        "amount_p95": float(df["amount"].quantile(0.95)),
        "velocity_p95": float(df["velocity_last_24h"].quantile(0.95)),
    }


# ---------------------------------------------------------------- sidebar: data source
st.sidebar.markdown("## 🛡️ FraudShield")
page = st.sidebar.radio("Navigate", ["⚡ Live Detector", "📊 Data Insights",
                                     "🧠 Model Performance", "📁 Batch Scoring"])
st.sidebar.markdown("---")

try:
    with st.sidebar.expander("Dataset source"):
        uploaded = st.file_uploader("Upload dataset CSV (optional)", type="csv")
    df = get_data(uploaded.getvalue() if uploaded else None)
except Exception as e:
    st.error(f"Could not load dataset: {e}\n\nUpload the CSV from the sidebar, or add it "
             "to the repo as `data/credit_card_fraud.csv`.")
    st.stop()

model, meta = get_model(model_stamp())
if model is None:
    with st.spinner("No saved model found — training one (first run only, ~30s)..."):
        P.train(df, run_grid=False, log=lambda m: None)
    get_model.clear()
    model, meta = get_model(model_stamp())

REF = reference_stats(df)
CATS = meta["merchant_categories"]


def score(rows: pd.DataFrame) -> np.ndarray:
    return P.predict(model, meta, rows)["fraud_probability"].to_numpy()


def risk_level(p: float, thr: float):
    if p >= thr:
        return "HIGH RISK", RED, "🚨"
    if p >= thr * 0.5:
        return "REVIEW", AMBER, "⚠️"
    return "SAFE", GREEN, "✅"


def card(col, label, value, color=None):
    style = f"color:{color};" if color else ""
    col.markdown(f'<div class="card"><div class="lbl">{label}</div>'
                 f'<div class="val" style="{style}">{value}</div></div>', unsafe_allow_html=True)


# ================================================================ LIVE DETECTOR
if page == "⚡ Live Detector":
    PRESETS = {
        "🛒 Normal": dict(amount=45.0, transaction_hour=13, merchant_category=CATS[0],
                                    foreign_transaction="No", location_mismatch="No",
                                    device_trust_score=85, velocity_last_24h=1, cardholder_age=34),
        "🤔 Unusual": dict(amount=420.0, transaction_hour=23, merchant_category=CATS[-1],
                          foreign_transaction="Yes", location_mismatch="No",
                          device_trust_score=45, velocity_last_24h=4, cardholder_age=27),
        "🚨 Fraud": dict(amount=780.0, transaction_hour=2, merchant_category=CATS[-1],
                                foreign_transaction="Yes", location_mismatch="Yes",
                                device_trust_score=15, velocity_last_24h=8, cardholder_age=22),
    }
    DEFAULT = PRESETS["🛒 Normal"]
    for k, v in DEFAULT.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("history", [])

    def apply_preset(name):
        for k, v in PRESETS[name].items():
            st.session_state[k] = v
        st.session_state["scored"] = dict(PRESETS[name])  # auto-predict

    def randomize():
        r = df.sample(1).iloc[0]
        for c in P.NUM_COLS:
            v = r[c]
            if c in ("foreign_transaction", "location_mismatch"):
                v = "Yes" if int(v) else "No"
            elif c == "amount":
                v = float(round(min(v, 5000.0), 2))
            elif c == "velocity_last_24h":
                v = int(min(v, 20))
            else:
                v = int(v)
            st.session_state[c] = v
        st.session_state["merchant_category"] = r["merchant_category"]
        st.session_state["scored"] = {k: st.session_state[k] for k in PRESETS["🛒 Normal"]}

    thr = st.sidebar.slider("🎯 Alert threshold", 0.05, 0.95, 0.50, 0.05)
    ss = st.session_state

    st.markdown('<div class="hero"><h1>🛡️ FraudShield — Fraud Detector</h1>'
                '<p>Enter the transaction details below and click <b>Predict</b>.</p></div>',
                unsafe_allow_html=True)

    # ---- quick-fill buttons
    st.markdown("##### ⚡ Quick fill (optional)")
    q = st.columns(4)
    for col, name in zip(q, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), width="stretch")
    q[3].button("🎲 Random real row", on_click=randomize, width="stretch")

    # ---- input form: type values, then Predict
    with st.form("txn_form", border=True):
        st.markdown("#### 📝 Enter transaction details")
        r1 = st.columns(4)
        r1[0].number_input("💰 Amount", 0.0, 5000.0, step=10.0, key="amount")
        r1[1].number_input("🕐 Hour of day (0–23)", 0, 23, step=1, key="transaction_hour")
        r1[2].selectbox("🏬 Merchant category", CATS, key="merchant_category")
        r1[3].number_input("👤 Cardholder age", 18, 90, step=1, key="cardholder_age")
        r2 = st.columns(4)
        r2[0].selectbox("🌍 Foreign transaction", ["No", "Yes"], key="foreign_transaction")
        r2[1].selectbox("📍 Location mismatch", ["No", "Yes"], key="location_mismatch")
        r2[2].number_input("📱 Device trust score (0–100)", 0, 100, step=1, key="device_trust_score")
        r2[3].number_input("⚡ Transactions in last 24h", 0, 20, step=1, key="velocity_last_24h")
        submitted = st.form_submit_button("🔍 Predict", type="primary", width="stretch")

    fields = ["amount", "transaction_hour", "merchant_category", "foreign_transaction",
              "location_mismatch", "device_trust_score", "velocity_last_24h", "cardholder_age"]
    if submitted:
        ss["scored"] = {k: ss[k] for k in fields}
    if "scored" not in ss:
        st.info("👆 Enter values (or use a quick-fill button) and click **Predict** to see the result.")
        st.stop()

    sv = ss["scored"]
    txn = {
        "amount": float(sv["amount"]), "transaction_hour": int(sv["transaction_hour"]),
        "merchant_category": sv["merchant_category"],
        "foreign_transaction": 1 if sv["foreign_transaction"] == "Yes" else 0,
        "location_mismatch": 1 if sv["location_mismatch"] == "Yes" else 0,
        "device_trust_score": int(sv["device_trust_score"]),
        "velocity_last_24h": int(sv["velocity_last_24h"]), "cardholder_age": int(sv["cardholder_age"]),
    }
    row = pd.DataFrame([txn])
    p = float(score(row)[0])
    level, color, icon = risk_level(p, thr)

    st.markdown("## 🎯 Prediction result")
    k1, k2, k3, k4 = st.columns([1.4, 1, 1, 1])
    k1.markdown(f'<div class="verdict" style="background:{color}">{icon} {level}</div>',
                unsafe_allow_html=True)
    card(k2, "Fraud probability", f"{p:.1%}", color)
    card(k3, "vs. base fraud rate", f"{p / max(REF['overall_rate'], 1e-9):.1f}×")
    card(k4, "Alert threshold", f"{thr:.0%}")
    st.write("")

    # ---- gauge + red flags
    g, f = st.columns([1.1, 1])
    with g:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100,
            number={"suffix": "%", "font": {"size": 48, "color": color}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.3},
                   "steps": [{"range": [0, thr * 50], "color": "rgba(22,163,74,.18)"},
                             {"range": [thr * 50, thr * 100], "color": "rgba(245,158,11,.20)"},
                             {"range": [thr * 100, 100], "color": "rgba(220,38,38,.20)"}],
                   "threshold": {"line": {"color": "black", "width": 4}, "value": thr * 100}}))
        fig.update_layout(height=290, margin=dict(t=30, b=10, l=30, r=30),
                          title={"text": "Fraud risk meter", "x": 0.5})
        st.plotly_chart(fig, width="stretch")

    with f:
        st.markdown("#### 🚩 Red-flag checklist")
        flags = [
            ("Foreign transaction", txn["foreign_transaction"] == 1),
            ("Location mismatch", txn["location_mismatch"] == 1),
            ("Low device trust (< 40)", txn["device_trust_score"] < 40),
            (f"High velocity (> {REF['velocity_p95']:.0f}/day)",
             txn["velocity_last_24h"] > REF["velocity_p95"]),
            (f"Large amount (> {REF['amount_p95']:.0f})", txn["amount"] > REF["amount_p95"]),
            ("Late-night (00:00–05:00)", txn["transaction_hour"] <= 5),
        ]
        st.markdown("".join(f'<span class="chip {"bad" if hit else "ok"}">'
                            f'{"✖" if hit else "✔"} {name}</span>' for name, hit in flags),
                    unsafe_allow_html=True)
        n_hit = sum(h for _, h in flags)
        st.progress(n_hit / len(flags), text=f"{n_hit} of {len(flags)} red flags raised")
        cat_rate = REF["cat_rate"].get(txn["merchant_category"], 0)
        hr_rate = REF["hour_rate"].get(txn["transaction_hour"], 0)
        m1, m2 = st.columns(2)
        m1.metric(f"Fraud rate · {txn['merchant_category']}", f"{cat_rate:.2%}",
                  f"{(cat_rate - REF['overall_rate']) * 100:+.2f} pp vs avg", delta_color="inverse")
        m2.metric(f"Fraud rate · {txn['transaction_hour']:02d}:00", f"{hr_rate:.2%}",
                  f"{(hr_rate - REF['overall_rate']) * 100:+.2f} pp vs avg", delta_color="inverse")

    # ---- explanation: what drove this score
    st.markdown("### 🔬 What drove this score?")
    st.caption("Each bar = how much the probability changes if that feature is reset to a typical "
               "legitimate value. Red pushes toward fraud, green pulls toward legit.")
    variants = []
    for c in P.FEATURES:
        v = dict(txn); v[c] = REF["typical"][c]
        if c in ("foreign_transaction", "location_mismatch"):
            v[c] = 0
        variants.append(v)
    contrib = p - score(pd.DataFrame(variants))
    cdf = pd.DataFrame({"feature": P.FEATURES, "impact": contrib * 100})
    cdf = cdf.reindex(cdf["impact"].abs().sort_values().index)
    fig = go.Figure(go.Bar(x=cdf["impact"], y=cdf["feature"], orientation="h",
                           marker_color=[RED if v > 0 else GREEN for v in cdf["impact"]],
                           text=[f"{v:+.1f} pp" for v in cdf["impact"]], textposition="outside"))
    span = max(5.0, float(cdf["impact"].abs().max()) * 1.25)
    fig.update_layout(height=330, margin=dict(t=10, b=10, l=10, r=40), xaxis_range=[-span, span],
                      xaxis_title="Impact on fraud probability (percentage points)")
    st.plotly_chart(fig, width="stretch")

    # ---- what-if explorer
    st.markdown("### 🧪 What-if explorer")
    w1, w2 = st.columns([1, 3])
    feat = w1.selectbox("Vary feature", P.FEATURES, index=P.FEATURES.index("device_trust_score"))
    w1.caption("All other inputs stay fixed at your current transaction.")
    if feat == "merchant_category":
        grid = CATS
    elif feat in ("foreign_transaction", "location_mismatch"):
        grid = [0, 1]
    else:
        lo = min(float(df[feat].min()), float(txn[feat]))
        hi = max(float(df[feat].max()), float(txn[feat]))
        grid = np.linspace(lo, hi, 60)
        if feat != "amount":
            grid = np.unique(grid.round().astype(int))
    sweep = pd.DataFrame([{**txn, feat: g} for g in grid])
    sweep["prob"] = score(sweep) * 100
    if feat in ("merchant_category", "foreign_transaction", "location_mismatch"):
        fig = px.bar(sweep, x=feat, y="prob", text_auto=".1f",
                     color="prob", color_continuous_scale=["#16a34a", "#f59e0b", "#dc2626"],
                     range_color=[0, 100])
    else:
        fig = px.area(sweep, x=feat, y="prob")
        fig.update_traces(line_color="#7c3aed", fillcolor="rgba(124,58,237,.15)")
        fig.add_vline(x=txn[feat], line_dash="dash", line_color=color,
                      annotation_text="current", annotation_position="top")
    fig.add_hline(y=thr * 100, line_dash="dot", line_color=RED, annotation_text="threshold")
    fig.update_layout(height=320, yaxis_title="Fraud probability (%)", yaxis_range=[0, 105],
                      margin=dict(t=20, b=10), coloraxis_showscale=False)
    w2.plotly_chart(fig, width="stretch")

    # ---- where it sits vs population
    st.markdown("### 📍 How this transaction compares to all transactions")
    cols = st.columns(4)
    for col, c, lbl in zip(cols, ["amount", "device_trust_score", "velocity_last_24h", "cardholder_age"],
                           ["Amount", "Device trust", "Velocity (24h)", "Age"]):
        pct = float((df[c] <= txn[c]).mean())
        fig = px.histogram(df, x=c, nbins=40, color_discrete_sequence=["#94a3b8"])
        fig.add_vline(x=txn[c], line_color=color, line_width=3)
        fig.update_layout(height=200, margin=dict(t=30, b=0, l=0, r=0), showlegend=False,
                          title={"text": f"{lbl} · {pct:.0%} percentile", "font": {"size": 13}},
                          xaxis_title=None, yaxis_title=None)
        col.plotly_chart(fig, width="stretch")

    # ---- session log
    st.markdown("### 🗂️ Session log")
    l1, l2, _ = st.columns([1, 1, 4])
    if l1.button("📌 Log this transaction", type="primary"):
        ss.history.append({**txn, "fraud_probability": round(p, 4), "verdict": level})
    if l2.button("🧹 Clear log"):
        ss.history = []
    if ss.history:
        hist = pd.DataFrame(ss.history)
        st.dataframe(hist.style.background_gradient(subset=["fraud_probability"], cmap="RdYlGn_r",
                                                    vmin=0, vmax=1),
                     width="stretch", hide_index=True)
        st.download_button("⬇️ Download log", hist.to_csv(index=False), "fraud_log.csv", "text/csv")
    else:
        st.caption("Log transactions to compare scenarios side by side.")


# ================================================================ DATA INSIGHTS
elif page == "📊 Data Insights":
    st.markdown('<div class="hero"><h1>📊 Data Insights</h1>'
                '<p>Exploratory analysis of the transaction dataset.</p></div>', unsafe_allow_html=True)
    k = st.columns(4)
    card(k[0], "Transactions", f"{len(df):,}")
    card(k[1], "Fraud cases", f"{int(df[P.TARGET].sum()):,}", RED)
    card(k[2], "Fraud rate", f"{df[P.TARGET].mean():.2%}", RED)
    card(k[3], "Avg amount", f"{df['amount'].mean():,.0f}")
    st.write("")

    lab = df[P.TARGET].map({0: "Legit", 1: "Fraud"})
    cmap = {"Legit": "#64748b", "Fraud": RED}
    t1, t2, t3, t4 = st.tabs(["Fraud patterns", "Distributions", "Correlation", "Raw data"])
    with t1:
        a, b = st.columns(2)
        hr = df.groupby("transaction_hour")[P.TARGET].mean().reset_index()
        a.plotly_chart(px.bar(hr, x="transaction_hour", y=P.TARGET, title="Fraud rate by hour",
                              color=P.TARGET, color_continuous_scale="Reds"), width="stretch")
        cr = df.groupby("merchant_category")[P.TARGET].mean().sort_values().reset_index()
        b.plotly_chart(px.bar(cr, x=P.TARGET, y="merchant_category", orientation="h",
                              title="Fraud rate by merchant category", color=P.TARGET,
                              color_continuous_scale="Reds"), width="stretch")
        a, b = st.columns(2)
        flags = pd.DataFrame({
            "flag": ["Foreign", "Location mismatch"],
            "fraud rate when YES": [df[df.foreign_transaction == 1][P.TARGET].mean(),
                                    df[df.location_mismatch == 1][P.TARGET].mean()],
            "fraud rate when NO": [df[df.foreign_transaction == 0][P.TARGET].mean(),
                                   df[df.location_mismatch == 0][P.TARGET].mean()]})
        a.plotly_chart(px.bar(flags.melt(id_vars="flag"), x="flag", y="value", color="variable",
                              barmode="group", title="Binary flags vs fraud rate",
                              color_discrete_sequence=[RED, "#64748b"]), width="stretch")
        b.plotly_chart(px.box(df, x=lab, y="device_trust_score", color=lab, color_discrete_map=cmap,
                              title="Device trust score: legit vs fraud"), width="stretch")
    with t2:
        col = st.selectbox("Feature", P.NUM_COLS)
        st.plotly_chart(px.histogram(df, x=col, color=lab, nbins=50, barmode="overlay",
                                     opacity=0.7, color_discrete_map=cmap, histnorm="percent",
                                     title=f"{col} (normalised per class)"), width="stretch")
        st.plotly_chart(px.box(df, x="merchant_category", y="amount", color=lab,
                               color_discrete_map=cmap, title="Amount by category"), width="stretch")
    with t3:
        st.plotly_chart(px.imshow(df[P.NUM_COLS + [P.TARGET]].corr(), text_auto=".2f",
                                  color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=620),
                        width="stretch")
    with t4:
        st.dataframe(df.head(200), width="stretch")
        st.dataframe(df.describe().T, width="stretch")


# ================================================================ MODEL
elif page == "🧠 Model Performance":
    st.markdown('<div class="hero"><h1>🧠 Model Performance</h1>'
                '<p>Tuned Random Forest · StandardScaler + OneHotEncoder pipeline</p></div>',
                unsafe_allow_html=True)
    tm = meta["test_metrics"]
    k = st.columns(4)
    card(k[0], "Accuracy", f"{tm['accuracy']:.3f}")
    card(k[1], "Macro F1", f"{tm['f1_macro']:.3f}")
    card(k[2], "ROC AUC", f"{tm['roc_auc']:.3f}")
    card(k[3], "Fraud recall", f"{tm['report']['1']['recall']:.3f}")
    st.write("")

    a, b = st.columns(2)
    comp = pd.DataFrame(meta["cv_comparison"]).T.reset_index(names="model")
    a.plotly_chart(px.bar(comp.melt(id_vars="model"), x="model", y="value", color="variable",
                          barmode="group", text_auto=".3f", title="Model comparison (3-fold CV)"),
                   width="stretch")
    fi = pd.Series(meta["feature_importance"]).sort_values()
    b.plotly_chart(px.bar(fi, orientation="h", title="Feature importance",
                          color=fi.values, color_continuous_scale="Purples"), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.imshow(tm["confusion_matrix"], text_auto=True, color_continuous_scale="Blues",
                             x=["Pred Legit", "Pred Fraud"], y=["Legit", "Fraud"],
                             title="Confusion matrix (test set)"), width="stretch")
    b.markdown("#### Classification report")
    b.dataframe(pd.DataFrame(tm["report"]).T.style.format("{:.3f}"), width="stretch")
    b.caption(f"Best params: `{meta['best_params']}`")

    st.markdown("---")
    fast = st.toggle("Fast retrain (skip GridSearch)", value=True)
    if st.button("🔁 Retrain model", type="primary"):
        logbox = st.empty(); lines = []

        def log(m):
            lines.append(m); logbox.code("\n".join(lines))
        with st.spinner("Training..."):
            P.train(df, run_grid=not fast, log=log)
        get_model.clear()
        st.rerun()


# ================================================================ BATCH
elif page == "📁 Batch Scoring":
    st.markdown('<div class="hero"><h1>📁 Batch Scoring</h1>'
                '<p>Upload many transactions and score them at once.</p></div>', unsafe_allow_html=True)
    st.write("Required columns: `" + "`, `".join(P.FEATURES) + "`")
    f = st.file_uploader("Transactions CSV", type="csv", key="batch")
    thr = st.slider("Alert threshold", 0.05, 0.95, 0.50, 0.05)
    src = pd.read_csv(f) if f else None
    if src is None and st.checkbox("Use 500 random rows from the dataset as a demo"):
        src = df.sample(500, random_state=0)

    if src is not None:
        missing = [c for c in P.FEATURES if c not in src.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
        res = P.predict(model, meta, src)
        res["prediction"] = (res["fraud_probability"] >= thr).astype(int)
        k = st.columns(3)
        card(k[0], "Transactions", f"{len(res):,}")
        card(k[1], "Flagged", f"{int(res['prediction'].sum()):,}", RED)
        card(k[2], "Flag rate", f"{res['prediction'].mean():.2%}")
        st.write("")
        st.plotly_chart(px.histogram(res, x="fraud_probability", nbins=40,
                                     title="Score distribution", color_discrete_sequence=["#7c3aed"]),
                        width="stretch")
        st.dataframe(res.sort_values("fraud_probability", ascending=False)
                     .style.background_gradient(subset=["fraud_probability"], cmap="RdYlGn_r",
                                                vmin=0, vmax=1),
                     width="stretch", hide_index=True)
        st.download_button("⬇️ Download results", res.to_csv(index=False),
                           "fraud_predictions.csv", "text/csv")
