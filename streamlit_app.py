import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from core_training import run_training

st.set_page_config(page_title="FL Fraud System", layout="wide")

st.title("Secure Federated Learning Fraud Detection")

if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None
if "last_dataset_name" not in st.session_state:
    st.session_state.last_dataset_name = None


def save_uploaded_file(uploaded_file, folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(path)


def preview_csv(uploaded_file, title: str):
    if uploaded_file is None:
        st.info(f"{title}: no file uploaded")
        return
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader(title)
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
        uploaded_file.seek(0)
    except Exception as e:
        st.error(f"Could not preview {title}: {e}")


with st.sidebar:
    st.header("Model Settings")

    rounds = st.number_input("Rounds", min_value=1, max_value=150, value=20, step=1)
    clients = st.number_input("Clients", min_value=2, max_value=50, value=10, step=1)

    with st.expander("Training / FL", expanded=True):
        batch_size = st.number_input("Batch size", min_value=16, max_value=4096, value=512, step=16)
        learning_rate = st.number_input("Learning rate", min_value=0.00001, max_value=1.0, value=0.0002, format="%.5f")
        weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=0.0001, format="%.5f")
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)
        local_epochs = st.number_input("Local epochs", min_value=1, max_value=20, value=3, step=1)
        client_fraction = st.slider("Client fraction", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
        early_stopping_patience = st.number_input("Early stopping patience", min_value=1, max_value=50, value=10, step=1)
        noniid_dominance = st.slider("Non-IID dominance", min_value=0.1, max_value=1.0, value=0.6, step=0.05)

    with st.expander("Model", expanded=True):
        d_model = st.number_input("D_MODEL", min_value=8, max_value=512, value=128, step=8)
        n_heads = st.number_input("N_HEADS", min_value=1, max_value=16, value=8, step=1)
        n_layers = st.number_input("N_LAYERS", min_value=1, max_value=12, value=3, step=1)
        dropout = st.slider("Dropout", min_value=0.0, max_value=0.7, value=0.10, step=0.01)
        mlp_hidden = st.number_input("MLP hidden", min_value=16, max_value=2048, value=256, step=16)
        loss_type = st.selectbox("Loss type", ["focal", "bce"], index=0)
        aggregation_method = st.selectbox("Aggregation method", ["fedavg", "trimmed_mean", "coordinate_median"], index=0)
        selection_metric = st.selectbox("Selection metric", ["pr_auc", "roc_auc", "f1"], index=0)

    with st.expander("Security / XAI", expanded=True):
        enable_dp = st.checkbox("Enable DP", value=True)
        enable_encryption = st.checkbox("Enable encryption", value=True)
        enable_signing = st.checkbox("Enable signing", value=True)
        enable_trusted_updates = st.checkbox("Trusted updates only", value=True)
        enable_replay_protection = st.checkbox("Replay protection", value=True)
        enable_fedprox = st.checkbox("Enable FedProx", value=True)
        enable_feature_engineering = st.checkbox("Feature engineering", value=True)
        enable_log1p = st.checkbox("Log1p skewed features", value=True)
        enable_binary_no_standardize = st.checkbox("Binary no standardize", value=True)
        enable_update_anomaly_filter = st.checkbox("Update anomaly filter", value=False)
        enable_shap = st.checkbox("Enable SHAP", value=True)
        enable_tee_independent_shap = st.checkbox("TEE independent SHAP", value=True)
        enable_permutation_importance = st.checkbox("Permutation importance", value=True)
        enable_ablation = st.checkbox("Ablation study", value=False)

    st.caption("High values may take longer on cloud deployment.")


tab1, tab2, tab3 = st.tabs(["Configuration", "Run", "Results"])

with tab1:
    st.header("Files")

    col_up_1, col_up_2 = st.columns(2)

    with col_up_1:
        dataset_file = st.file_uploader("Upload main dataset CSV", type=["csv"], key="dataset_file")
        shap_bg_file = st.file_uploader("Upload SHAP background CSV", type=["csv"], key="shap_bg_file")

    with col_up_2:
        shap_explain_file = st.file_uploader("Upload SHAP explain CSV", type=["csv"], key="shap_explain_file")

    st.markdown("---")

    col_prev_1, col_prev_2 = st.columns(2)
    with col_prev_1:
        preview_csv(dataset_file, "Dataset Preview")
    with col_prev_2:
        preview_csv(shap_bg_file, "SHAP Background Preview")
        preview_csv(shap_explain_file, "SHAP Explain Preview")

    st.markdown("---")
    st.subheader("Current Configuration")

    config_preview = {
        "ROUNDS": int(rounds),
        "NUM_CLIENTS": int(clients),
        "BATCH_SIZE": int(batch_size),
        "LR": float(learning_rate),
        "WEIGHT_DECAY": float(weight_decay),
        "SEED": int(seed),
        "LOCAL_EPOCHS": int(local_epochs),
        "CLIENT_FRACTION": float(client_fraction),
        "EARLY_STOPPING_PATIENCE": int(early_stopping_patience),
        "NONIID_LABEL_DOMINANCE": float(noniid_dominance),
        "D_MODEL": int(d_model),
        "N_HEADS": int(n_heads),
        "N_LAYERS": int(n_layers),
        "DROPOUT": float(dropout),
        "MLP_HIDDEN": int(mlp_hidden),
        "LOSS_TYPE": loss_type,
        "AGGREGATION_METHOD": aggregation_method,
        "MODEL_SELECTION_METRIC": selection_metric,
        "ENABLE_DP": bool(enable_dp),
        "ENABLE_ENCRYPTION": bool(enable_encryption),
        "ENABLE_SIGNING": bool(enable_signing),
        "ENABLE_TRUSTED_UPDATES_ONLY": bool(enable_trusted_updates),
        "ENABLE_REPLAY_PROTECTION": bool(enable_replay_protection),
        "ENABLE_FEDPROX": bool(enable_fedprox),
        "ENABLE_FEATURE_ENGINEERING": bool(enable_feature_engineering),
        "ENABLE_LOG1P_SKEWED_FEATURES": bool(enable_log1p),
        "ENABLE_BINARY_NO_STANDARDIZE": bool(enable_binary_no_standardize),
        "ENABLE_UPDATE_ANOMALY_FILTER": bool(enable_update_anomaly_filter),
        "ENABLE_SHAP": bool(enable_shap),
        "ENABLE_TEE_INDEPENDENT_SHAP": bool(enable_tee_independent_shap),
        "ENABLE_PERMUTATION_IMPORTANCE": bool(enable_permutation_importance),
        "ENABLE_ABLATION_STUDY": bool(enable_ablation),
    }
    st.json(config_preview)

with tab2:
    st.header("Run Training")

    if enable_shap and (shap_bg_file is None or shap_explain_file is None):
        st.warning("SHAP is enabled, but SHAP background/explain files are not uploaded yet.")

    if d_model % n_heads != 0:
        st.error("D_MODEL must be divisible by N_HEADS.")

    start_training = st.button("Start Training", type="primary", use_container_width=True)

    if start_training:
        if dataset_file is None:
            st.error("Please upload the main dataset CSV first.")
            st.stop()

        if d_model % n_heads != 0:
            st.stop()

        work_dir = Path(tempfile.mkdtemp(prefix="fl_fraud_app_"))
        uploads_dir = work_dir / "uploads"
        outputs_dir = work_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = save_uploaded_file(dataset_file, uploads_dir)
        shap_bg_path = ""
        shap_explain_path = ""

        if shap_bg_file is not None:
            shap_bg_path = save_uploaded_file(shap_bg_file, uploads_dir)
        if shap_explain_file is not None:
            shap_explain_path = save_uploaded_file(shap_explain_file, uploads_dir)

        config = {
            "DATA_PATH": dataset_path,
            "SHAP_BACKGROUND_PATH": shap_bg_path,
            "SHAP_EXPLAIN_PATH": shap_explain_path,
            "OUTPUT_DIR": str(outputs_dir),
            "ROUNDS": int(rounds),
            "NUM_CLIENTS": int(clients),
            "BATCH_SIZE": int(batch_size),
            "LR": float(learning_rate),
            "WEIGHT_DECAY": float(weight_decay),
            "SEED": int(seed),
            "LOCAL_EPOCHS": int(local_epochs),
            "CLIENT_FRACTION": float(client_fraction),
            "EARLY_STOPPING_PATIENCE": int(early_stopping_patience),
            "NONIID_LABEL_DOMINANCE": float(noniid_dominance),
            "D_MODEL": int(d_model),
            "N_HEADS": int(n_heads),
            "N_LAYERS": int(n_layers),
            "DROPOUT": float(dropout),
            "MLP_HIDDEN": int(mlp_hidden),
            "LOSS_TYPE": loss_type,
            "AGGREGATION_METHOD": aggregation_method,
            "MODEL_SELECTION_METRIC": selection_metric,
            "ENABLE_DP": bool(enable_dp),
            "ENABLE_ENCRYPTION": bool(enable_encryption),
            "ENABLE_SIGNING": bool(enable_signing),
            "ENABLE_TRUSTED_UPDATES_ONLY": bool(enable_trusted_updates),
            "ENABLE_REPLAY_PROTECTION": bool(enable_replay_protection),
            "ENABLE_FEDPROX": bool(enable_fedprox),
            "ENABLE_FEATURE_ENGINEERING": bool(enable_feature_engineering),
            "ENABLE_LOG1P_SKEWED_FEATURES": bool(enable_log1p),
            "ENABLE_BINARY_NO_STANDARDIZE": bool(enable_binary_no_standardize),
            "ENABLE_UPDATE_ANOMALY_FILTER": bool(enable_update_anomaly_filter),
            "ENABLE_SHAP": bool(enable_shap),
            "ENABLE_TEE_INDEPENDENT_SHAP": bool(enable_tee_independent_shap),
            "ENABLE_PERMUTATION_IMPORTANCE": bool(enable_permutation_importance),
            "ENABLE_ABLATION_STUDY": bool(enable_ablation),
        }

        st.success("Files uploaded successfully.")
        st.write("Run configuration:")
        st.json(config)

        progress = st.progress(0)
        log_box = st.empty()

        results, summary = run_training(config)

        for i, row in enumerate(results):
            progress.progress((i + 1) / len(results))
            log_box.write(
                f"Round {row['round']} → PR-AUC={row['pr_auc']} ROC-AUC={row['roc_auc']} F1={row['f1']}"
            )

        st.session_state.last_results = results
        st.session_state.last_summary = summary
        st.session_state.last_dataset_name = os.path.basename(dataset_path)

        st.success("Training completed. Open the Results tab.")

with tab3:
    st.header("Results")

    if st.session_state.last_results is None:
        st.info("No results yet. Go to the Run tab and start training.")
    else:
        results = st.session_state.last_results
        summary = st.session_state.last_summary

        st.subheader("Final Results")
        st.json(summary)

        df_results = pd.DataFrame(results)
        st.line_chart(df_results.set_index("round")[["pr_auc", "roc_auc", "f1"]])

        st.subheader("Round-by-Round Metrics")
        st.dataframe(df_results, use_container_width=True)

        col_dl_1, col_dl_2 = st.columns(2)

        with col_dl_1:
            csv_data = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download metrics CSV",
                data=csv_data,
                file_name="round_metrics.csv",
                mime="text/csv",
            )

        with col_dl_2:
            st.download_button(
                "Download summary JSON",
                data=pd.Series(summary).to_json(indent=2),
                file_name="training_summary.json",
                mime="application/json",
            )

        st.subheader("SHAP / XAI")
        if enable_shap:
            if summary.get("shap_status") == "completed":
                st.success("SHAP analysis completed.")
                if "shap_top_features" in summary:
                    st.dataframe(
                        pd.DataFrame(summary["shap_top_features"]),
                        use_container_width=True,
                    )
            else:
                st.info("SHAP is enabled, but real SHAP output is not connected yet in core_training.py.")
        else:
            st.info("SHAP is disabled for this run.")
