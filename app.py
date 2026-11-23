"""SmartEdge AI - Streamlit backend.

Run:  streamlit run app.py

Everything marked MEASURED is read from this machine at run time.
Everything marked SIMULATED is an assumption you can change in the UI.
No energy savings are claimed: the app reports CPU time and memory only.
"""
import pickle
import re
import time

import numpy as np
import pandas as pd
import psutil
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="SmartEdge AI", page_icon="🛰️", layout="wide")

# --------------------------------------------------------------------------
# Data and models
# --------------------------------------------------------------------------
POS_W = ["great", "excellent", "fast", "helpful", "friendly", "smooth", "reliable",
         "amazing", "easy", "perfect", "lovely", "quick", "polite", "solid"]
NEG_W = ["terrible", "awful", "slow", "rude", "confusing", "unreliable", "broken",
         "poor", "useless", "horrible", "late", "buggy", "damaged", "frustrating"]
THINGS = ["delivery", "support team", "app", "checkout", "packaging", "battery",
          "service", "update", "camera", "setup", "website", "seller"]
FRAMES = ["The {t} was {w}.", "I found the {t} {w}.", "Honestly the {t} was {w} for me.",
          "{w} {t}, would say the same again.", "My experience with the {t} was {w}."]


def make_dataset(n=400, seed=0):
    """Template-generated reviews. Replace with a real labelled dataset."""
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for _ in range(n):
        y = int(rng.random() < 0.5)
        words = POS_W if y else NEG_W
        other = NEG_W if y else POS_W
        sent = rng.choice(FRAMES).format(t=rng.choice(THINGS), w=rng.choice(words))
        if rng.random() < 0.25:  # mixed review: the earlier clause points the other way
            sent = rng.choice(FRAMES).format(t=rng.choice(THINGS), w=rng.choice(other)) + " But " + sent[0].lower() + sent[1:]
        texts.append(sent)
        labels.append(y)
    return texts, labels


TEXTS, LABELS = make_dataset()
LABEL_NAME = {1: "Positive", 0: "Negative"}


@st.cache_resource(show_spinner="Training local models...")
def train_models():
    """Train three models of increasing size and measure them."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        TEXTS, LABELS, test_size=0.3, random_state=42, stratify=LABELS
    )
    specs = {
        "Small - Naive Bayes": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 1)), MultinomialNB()
        ),
        "Medium - Logistic Regression": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000)
        ),
        "Large - Random Forest (300 trees)": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2)),
            RandomForestClassifier(n_estimators=300, random_state=42),
        ),
    }
    out = {}
    for name, pipe in specs.items():
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t0
        acc = accuracy_score(y_te, pipe.predict(X_te))
        out[name] = {
            "pipe": pipe,
            "accuracy": acc,
            "train_s": train_s,
            "size_kb": len(pickle.dumps(pipe)) / 1024,
            "n_test": len(y_te),
        }
    return out


def benchmark(pipe, text, runs=200):
    """Measure wall time and CPU time per prediction."""
    pipe.predict([text])  # warm up
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(runs):
        pipe.predict([text])
    w1, c1 = time.perf_counter(), time.process_time()
    return (w1 - w0) / runs * 1000, (c1 - c0) / runs * 1000


def log_event(kind, detail):
    st.session_state.setdefault("log", []).append(
        {"time": time.strftime("%H:%M:%S"), "module": kind, **detail}
    )


# --------------------------------------------------------------------------
# Privacy Guardian
# --------------------------------------------------------------------------
PATTERNS = [
    ("Email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("Card number", re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD]"),
    ("Aadhaar", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"), "[AADHAAR]"),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    ("Phone", re.compile(r"(?:\+\d{1,3}[ -]?)?\b\d{5}[ -]?\d{5}\b"), "[PHONE]"),
]


def mask_text(text):
    found, masked = [], text
    for name, rx, token in PATTERNS:
        hits = rx.findall(masked)
        if hits:
            found.append({"Type": name, "Found": len(hits), "Replaced with": token})
            masked = rx.sub(token, masked)
    return masked, found


# --------------------------------------------------------------------------
# Fairness
# --------------------------------------------------------------------------
def synthetic_data(n, skew, minority_share, seed=0):
    rng = np.random.default_rng(seed)
    group = np.where(rng.random(n) < minority_share, "B", "A")
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    shift = np.where(group == "B", skew, 0.0)
    noise = np.where(group == "B", 0.8, 0.0) * rng.normal(size=n)
    return pd.DataFrame({"x1": x1 - shift + noise, "x2": x2, "group": group, "target": y})


def fairness_report(df, target, group_col):
    feats = pd.get_dummies(df.drop(columns=[target, group_col])).astype(float)
    y, g = df[target].astype(int), df[group_col].astype(str)
    X_tr, X_te, y_tr, y_te, _, g_te = train_test_split(
        feats, y, g, test_size=0.3, random_state=1, stratify=y
    )
    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    pred = pd.Series(model.predict(X_te), index=y_te.index)
    rows = []
    for grp in sorted(g_te.unique()):
        m = g_te == grp
        pos = (y_te[m] == 1)
        rows.append({
            "Group": grp,
            "Test rows": int(m.sum()),
            "Accuracy": accuracy_score(y_te[m], pred[m]),
            "Approval rate": float((pred[m] == 1).mean()),
            "True positive rate": float((pred[m][pos] == 1).mean()) if pos.any() else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("SmartEdge AI")
st.caption("Think Smart. Compute Local. Build Responsible AI.")

models = train_models()
tabs = st.tabs([
    "Overview", "Edge AI", "Resource Monitor", "Privacy Guardian",
    "Fairness Checker", "Smart Optimizer", "Analytics",
])

# ---- Overview -----------------------------------------------------------
with tabs[0]:
    st.subheader("This machine (measured)")
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    c = st.columns(5)
    c[0].metric("CPU cores", psutil.cpu_count(logical=True))
    c[1].metric("Total RAM", f"{vm.total / 1e9:.1f} GB")
    c[2].metric("CPU now", f"{psutil.cpu_percent(interval=0.3):.0f}%")
    c[3].metric("RAM used", f"{vm.percent:.0f}%")
    c[4].metric("This app's memory", f"{proc.memory_info().rss / 1e6:.0f} MB")
    st.markdown(
        "**Flow:** input → Privacy Guardian → task analysis → Edge AI Engine → "
        "Resource Monitor → Fairness check → Smart Optimizer → Analytics"
    )
    st.info(
        "Models are trained on template-generated reviews (SIMULATED data), so accuracy "
        "figures only show the workflow. Replace TEXTS and LABELS with a real labelled dataset."
    )

# ---- Edge AI ------------------------------------------------------------
with tabs[1]:
    st.subheader("Edge AI prediction")
    name = st.selectbox("Model", list(models), index=1)
    text = st.text_area(
        "Text to classify",
        "The delivery was quick but the packaging was poor and the app keeps crashing.",
    )
    mask_first = st.checkbox("Mask sensitive data before processing", value=True)
    rtt = st.slider("Assumed network round trip to a cloud API (ms) - SIMULATED", 20, 800, 280)
    if st.button("Run locally", type="primary"):
        clean, found = mask_text(text) if mask_first else (text, [])
        pipe = models[name]["pipe"]
        wall, cpu = benchmark(pipe, clean)
        proba = pipe.predict_proba([clean])[0]
        label = int(pipe.predict([clean])[0])
        a, b, d = st.columns(3)
        a.metric("Prediction", LABEL_NAME[label], f"{proba[label] * 100:.0f}% confidence")
        b.metric("Local time per request (measured)", f"{wall:.2f} ms")
        d.metric("Cloud estimate (simulated)", f"{wall + rtt:.0f} ms")
        st.caption(f"CPU time per request (measured): {cpu:.2f} ms. Masked items: {sum(f['Found'] for f in found)}.")
        log_event("Edge AI", {"model": name, "local_ms": round(wall, 3),
                              "cloud_ms_simulated": round(wall + rtt, 1),
                              "masked": sum(f["Found"] for f in found)})

# ---- Resource Monitor ---------------------------------------------------
with tabs[2]:
    st.subheader("Resource monitor")
    secs = st.slider("Sampling time (seconds)", 5, 30, 10)
    load = st.checkbox("Run model predictions during sampling to create load")
    if st.button("Start sampling"):
        chart, rows = st.empty(), []
        proc = psutil.Process()
        pipe = models["Large - Random Forest (300 trees)"]["pipe"]
        psutil.cpu_percent(interval=None)
        t_start = time.time()
        while time.time() - t_start < secs:
            if load:
                for _ in range(30):
                    pipe.predict([TEXTS[0]])
            time.sleep(0.4)
            rows.append({
                "CPU %": psutil.cpu_percent(interval=None),
                "RAM %": psutil.virtual_memory().percent,
                "App memory (MB)": proc.memory_info().rss / 1e6,
            })
            chart.line_chart(pd.DataFrame(rows))
        df = pd.DataFrame(rows)
        st.success(f"Average CPU {df['CPU %'].mean():.0f}%, peak CPU {df['CPU %'].max():.0f}%, "
                   f"peak app memory {df['App memory (MB)'].max():.0f} MB (all measured).")
        log_event("Resource Monitor", {"avg_cpu": round(df["CPU %"].mean(), 1),
                                       "peak_cpu": round(df["CPU %"].max(), 1),
                                       "load": load})

# ---- Privacy Guardian ---------------------------------------------------
with tabs[3]:
    st.subheader("Privacy Guardian")
    sample = ("Hi, I'm Priya. Email priya.k@example.com, phone +91 98765 43210, "
              "Aadhaar 2345 6789 0123, PAN ABCDE1234F, card 4111 1111 1111 1111.")
    t = st.text_area("Text to scan", sample, key="priv_in")
    if st.button("Scan and mask"):
        masked, found = mask_text(t)
        st.markdown("**Safe to share**")
        st.code(masked, language=None)
        if found:
            st.dataframe(pd.DataFrame(found), hide_index=True)
        else:
            st.success("No sensitive items found.")
        log_event("Privacy Guardian", {"masked": sum(f["Found"] for f in found)})
    st.caption("Rule-based detection. It can miss names and free-text details, so treat it as a first filter.")

# ---- Fairness Checker ---------------------------------------------------
with tabs[4]:
    st.subheader("Fairness checker")
    up = st.file_uploader("Upload a CSV (optional)", type="csv")
    if up:
        df = pd.read_csv(up)
        target = st.selectbox("Target column (0/1)", df.columns)
        group_col = st.selectbox("Group column", [c for c in df.columns if c != target])
    else:
        c1, c2 = st.columns(2)
        skew = c1.slider("Data skew against Group B", 0.0, 2.0, 0.8, 0.1)
        share = c2.slider("Group B share of data", 0.05, 0.5, 0.2, 0.05)
        df = synthetic_data(3000, skew, share)
        target, group_col = "target", "group"
        st.caption("Using generated data (SIMULATED). Metrics below are measured on that data.")
    if st.button("Evaluate fairness"):
        rep = fairness_report(df, target, group_col)
        st.dataframe(rep.style.format({"Accuracy": "{:.1%}", "Approval rate": "{:.1%}",
                                       "True positive rate": "{:.1%}"}), hide_index=True)
        gap = (rep["Accuracy"].max() - rep["Accuracy"].min()) * 100
        di = rep["Approval rate"].min() / max(rep["Approval rate"].max(), 1e-9)
        if gap > 5 or di < 0.8:
            st.error(f"Gap found: accuracy gap {gap:.1f} points (limit 5), approval ratio {di:.2f} (limit 0.80). "
                     "Try rebalancing the data or using per-group thresholds, then test again.")
        else:
            st.success(f"Within limits: accuracy gap {gap:.1f} points, approval ratio {di:.2f}.")
        log_event("Fairness", {"accuracy_gap_pts": round(gap, 1), "approval_ratio": round(di, 2)})

# ---- Smart Optimizer ----------------------------------------------------
with tabs[5]:
    st.subheader("Smart optimizer")
    st.write("Benchmarks every local model and picks the smallest one that is accurate enough.")
    tol = st.slider("Accuracy you can give up compared with the best model (points)", 0, 20, 5)
    probe = st.text_input("Probe sentence", "The support team was helpful but the app was buggy.")
    if st.button("Benchmark and recommend"):
        rows = []
        for n, m in models.items():
            wall, cpu = benchmark(m["pipe"], probe, runs=100)
            rows.append({"Model": n, "Accuracy": m["accuracy"], "Size (KB)": m["size_kb"],
                         "Latency (ms)": wall, "CPU time (ms)": cpu})
        res = pd.DataFrame(rows)
        best = res["Accuracy"].max()
        ok = res[res["Accuracy"] >= best - tol / 100].sort_values("Size (KB)")
        pick = ok.iloc[0]
        st.dataframe(res.style.format({"Accuracy": "{:.0%}", "Size (KB)": "{:.0f}",
                                       "Latency (ms)": "{:.2f}", "CPU time (ms)": "{:.2f}"}),
                     hide_index=True)
        big = res.sort_values("Size (KB)").iloc[-1]
        st.success(f"Recommended: {pick['Model']}. It is {big['Size (KB)'] / pick['Size (KB)']:.0f}x smaller and "
                   f"{big['Latency (ms)'] / pick['Latency (ms)']:.1f}x faster than the largest model, "
                   f"with accuracy {pick['Accuracy']:.0%} versus the best {best:.0%}.")
        st.caption(f"Test set is only {models[pick['Model']]['n_test']} examples, so accuracy differences are noisy.")
        log_event("Smart Optimizer", {"recommended": pick["Model"], "tolerance_pts": tol})

# ---- Analytics ----------------------------------------------------------
with tabs[6]:
    st.subheader("Session analytics")
    log = st.session_state.get("log", [])
    if log:
        ldf = pd.DataFrame(log)
        st.dataframe(ldf, hide_index=True)
        st.bar_chart(ldf["module"].value_counts())
        st.download_button("Download report (CSV)", ldf.to_csv(index=False), "smartedge_report.csv")
    else:
        st.write("Run a module in another tab and its results will appear here.")
