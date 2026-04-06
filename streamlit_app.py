# =========================
# FILE 2: streamlit_app.py
# =========================

import streamlit as st
import pandas as pd
import tempfile
import os
from core_training import run_training

st.set_page_config(page_title="FL Fraud System", layout="wide")

st.title("Secure Federated Learning Fraud Detection")

# Upload dataset
st.header("1. Upload Dataset")
file = st.file_uploader("Upload creditcard.csv", type=["csv"])

# Sidebar config
st.sidebar.header("Model Settings")

rounds = st.sidebar.slider("Rounds", 1, 20, 10)
clients = st.sidebar.slider("Clients", 1, 20, 10)

run_button = st.button("Start Training")

if file and run_button:
    temp_dir = tempfile.mkdtemp()
    data_path = os.path.join(temp_dir, file.name)

    with open(data_path, "wb") as f:
        f.write(file.read())

    config = {
        "DATA_PATH": data_path,
        "ROUNDS": rounds,
        "NUM_CLIENTS": clients
    }

    st.success("Dataset uploaded successfully")

    progress = st.progress(0)
    log_placeholder = st.empty()

    results, summary = run_training(config)

    for i, r in enumerate(results):
        progress.progress((i + 1) / len(results))
        log_placeholder.write(f"Round {r['round']} → PR-AUC={r['pr_auc']} ROC-AUC={r['roc_auc']} F1={r['f1']}")

    st.subheader("Final Results")
    st.json(summary)

    df = pd.DataFrame(results)
    st.line_chart(df.set_index("round"))

else:
    st.info("Upload dataset and press Start Training")

