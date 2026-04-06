import os
import io
import json
import time
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Secure FL Fraud App", layout="wide")

st.title("Secure FL Fraud Detection - Streamlit")
st.write("Upload your dataset at runtime, then start training with your chosen settings.")


def save_uploaded_file(uploaded_file, folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def preview_csv(uploaded_file, label: str):
    if uploaded_file is None:
        return
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader(f"Preview: {label}")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
        uploaded_file.seek(0)
    except Exception as e:
        st.error(f"Could not read {label}: {e}")


with st.sidebar:
    st.header("1) Upload files")
    dataset_file = st.file_uploader("Main dataset CSV", type=["csv"], key="dataset")
    shap_bg_file = st.file_uploader("SHAP background CSV (optional)", type=["csv"], key="shap_bg")
    shap_explain_file = st.file_uploader("SHAP explain CSV (optional)", type=["csv"], key="shap_explain")

    st.header("2) Training settings")
    rounds = st.number_input("Rounds", min_value=1, max_value=200, value=10, step=1)
    local_epochs = st.number_input("Local epochs", min_value=1, max_value=20, value=3, step=1)
    num_clients = st.number_input("Number of clients", min_value=2, max_value=50, value=10, step=1)
    batch_size = st.number_input("Batch size", min_value=16, max_value=4096, value=512, step=16)
    learning_rate = st.number_input("Learning rate", min_value=0.00001, max_value=1.0, value=0.0002, format="%.5f")
    d_model = st.number_input("D_MODEL", min_value=8, max_value=512, value=128, step=8)
    n_heads = st.number_input("N_HEADS", min_value=1, max_value=16, value=8, step=1)
    n_layers = st.number_input("N_LAYERS", min_value=1, max_value=12, value=3, step=1)
    dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.10, step=0.01)
    mlp_hidden = st.number_input("MLP hidden", min_value=16, max_value=1024, value=256, step=16)

    st.header("3) Security and XAI")
    enable_dp = st.checkbox("Enable DP", value=True)
    enable_signing = st.checkbox("Enable signing", value=True)
    enable_encryption = st.checkbox("Enable encryption", value=True)
    enable_shap = st.checkbox("Enable SHAP", value=True)
    enable_perm = st.checkbox("Enable permutation importance", value=True)


col1, col2 = st.columns(2)
with col1:
    preview_csv(dataset_file, "Dataset")
with col2:
    preview_csv(shap_bg_file, "SHAP background")
    preview_csv(shap_explain_file, "SHAP explain")


st.markdown("---")
st.subheader("Run")

if dataset_file is None:
    st.info("Upload the main dataset CSV first.")

run_btn = st.button("Start training")

if run_btn:
    if dataset_file is None:
        st.error("Please upload the main dataset first.")
        st.stop()

    if d_model % n_heads != 0:
        st.error("D_MODEL must be divisible by N_HEADS.")
        st.stop()

    work_dir = Path(tempfile.mkdtemp(prefix="secure_fl_app_"))
    uploads_dir = work_dir / "uploads"
    outputs_dir = work_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = save_uploaded_file(dataset_file, uploads_dir)

    shap_bg_path = ""
    shap_explain_path = ""

    if enable_shap and shap_bg_file is not None and shap_explain_file is not None:
        shap_bg_path = save_uploaded_file(shap_bg_file, uploads_dir)
        shap_explain_path = save_uploaded_file(shap_explain_file, uploads_dir)
    elif enable_shap:
        st.warning("SHAP is enabled, but optional SHAP files were not uploaded. In your full version, you can auto-sample them from the dataset.")

    config_for_run = {
        "DATA_PATH": dataset_path,
        "SHAP_BACKGROUND_PATH": shap_bg_path,
        "SHAP_EXPLAIN_PATH": shap_explain_path,
        "OUTPUT_DIR": str(outputs_dir),
        "BATCH_SIZE": int(batch_size),
        "LR": float(learning_rate),
        "NUM_CLIENTS": int(num_clients),
        "ROUNDS": int(rounds),
        "LOCAL_EPOCHS": int(local_epochs),
        "D_MODEL": int(d_model),
        "N_HEADS": int(n_heads),
        "N_LAYERS": int(n_layers),
        "DROPOUT": float(dropout),
        "MLP_HIDDEN": int(mlp_hidden),
        "ENABLE_DP": bool(enable_dp),
        "ENABLE_SIGNING": bool(enable_signing),
        "ENABLE_ENCRYPTION": bool(enable_encryption),
        "ENABLE_SHAP": bool(enable_shap),
        "ENABLE_PERMUTATION_IMPORTANCE": bool(enable_perm),
    }

    st.success("Files uploaded and configuration prepared.")
    st.json(config_for_run)

    st.info(
        "Next step: connect this Streamlit wrapper to your training functions from GUI_TEST.py. "
        "You will replace the old Windows file paths with these uploaded file paths."
    )

    progress = st.progress(0)
    status = st.empty()

    for i in range(1, int(rounds) + 1):
        time.sleep(0.05)
        progress.progress(i / int(rounds))
        status.write(f"Simulated round {i}/{int(rounds)}")

    summary = {
        "status": "wrapper_ready",
        "output_dir": str(outputs_dir),
        "dataset_file": os.path.basename(dataset_path),
        "rounds": int(rounds),
        "num_clients": int(num_clients),
    }

    st.subheader("Summary")
    st.json(summary)

    summary_path = outputs_dir / "streamlit_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(summary_path, "rb") as f:
        st.download_button(
            "Download summary JSON",
            data=f,
            file_name="streamlit_run_summary.json",
            mime="application/json",
        )
