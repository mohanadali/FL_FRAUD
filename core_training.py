import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


@dataclass
class TrainingArtifacts:
    results: List[Dict]
    summary: Dict


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2)


def _safe_score_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_score_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _compute_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.50
    best_f1 = -1.0

    for th in thresholds:
        pred = (y_score >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(th)

    return best_threshold, float(best_f1)


def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    roc_auc = _safe_score_auc(y_true, y_score)
    pr_auc = _safe_score_ap(y_true, y_score)
    threshold, _ = _compute_best_threshold(y_true, y_score)
    pred = (y_score >= threshold).astype(int)

    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
    }


def _validate_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if "Class" not in df.columns:
        raise ValueError("The dataset must contain a target column named 'Class'.")

    unique_vals = set(pd.to_numeric(df["Class"], errors="coerce").dropna().astype(int).unique().tolist())
    if not unique_vals.issubset({0, 1}):
        raise ValueError("The 'Class' column must contain binary values 0 and 1 only.")


def _add_feature_engineering(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = df.copy()

    if not cfg.get("ENABLE_FEATURE_ENGINEERING", True):
        return df

    if "Amount" in df.columns:
        amount = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
        df["fe_log_amount"] = np.log1p(np.maximum(amount, 0))
        df["fe_amount_is_zero"] = (amount == 0).astype(int)

    if "Time" in df.columns:
        t = pd.to_numeric(df["Time"], errors="coerce").fillna(0.0)
        df["fe_time_hours"] = t / 3600.0
        df["fe_time_days"] = t / 86400.0
        df["fe_time_mod_day"] = t % 86400.0
        df["fe_time_sin_day"] = np.sin(2 * np.pi * df["fe_time_mod_day"] / 86400.0)
        df["fe_time_cos_day"] = np.cos(2 * np.pi * df["fe_time_mod_day"] / 86400.0)

    if cfg.get("ENABLE_LOG1P_SKEWED_FEATURES", True):
        for col in df.columns:
            if col == "Class":
                continue
            if any(token in col.lower() for token in ["amount", "count", "risk", "time", "minutes", "days"]):
                vals = pd.to_numeric(df[col], errors="coerce")
                if vals.notna().any():
                    df[col] = np.log1p(np.maximum(vals.fillna(0.0), 0.0))

    return df


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=["Class"]).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)

    return X, y


def _create_client_splits(y: np.ndarray, num_clients: int, seed: int, dominance: float = 0.60) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    total = len(y)
    base = total // num_clients
    sizes = [base] * num_clients
    sizes[-1] += total - sum(sizes)

    client_indices: List[np.ndarray] = []
    pos_cursor = 0
    neg_cursor = 0

    min_pos_each = max(1, len(idx_pos) // max(num_clients, 1)) if len(idx_pos) >= num_clients else 0

    for client_id, size in enumerate(sizes):
        dominant_positive = (client_id % 2 == 1)

        pos_target = min_pos_each
        remaining = max(size - pos_target, 0)

        if dominant_positive:
            extra_pos = int(round(remaining * dominance))
        else:
            extra_pos = int(round(remaining * (1.0 - dominance)))

        pos_target += extra_pos
        pos_target = min(pos_target, len(idx_pos) - pos_cursor)
        neg_target = size - pos_target
        neg_target = min(neg_target, len(idx_neg) - neg_cursor)

        current_pos = idx_pos[pos_cursor:pos_cursor + pos_target]
        current_neg = idx_neg[neg_cursor:neg_cursor + neg_target]
        pos_cursor += len(current_pos)
        neg_cursor += len(current_neg)

        current = np.concatenate([current_pos, current_neg])
        rng.shuffle(current)
        client_indices.append(current)

    leftovers = []
    if pos_cursor < len(idx_pos):
        leftovers.extend(idx_pos[pos_cursor:].tolist())
    if neg_cursor < len(idx_neg):
        leftovers.extend(idx_neg[neg_cursor:].tolist())

    rng.shuffle(leftovers)
    for i, idx in enumerate(leftovers):
        client_indices[i % num_clients] = np.append(client_indices[i % num_clients], idx)

    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _sample_clients(num_clients: int, fraction: float, rng: np.random.Generator) -> List[int]:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    k = max(1, int(math.ceil(num_clients * fraction)))
    selected = rng.choice(np.arange(num_clients), size=k, replace=False)
    return sorted(selected.tolist())


def _initialize_client_model(n_features: int, seed: int, alpha: float) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        fit_intercept=True,
        random_state=seed,
        class_weight="balanced",
        learning_rate="optimal",
        average=True,
    )


def _fit_local_client(
    global_coef: np.ndarray,
    global_intercept: np.ndarray,
    x_client: np.ndarray,
    y_client: np.ndarray,
    local_epochs: int,
    seed: int,
    alpha: float,
    fedprox_mu: float,
    enable_fedprox: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    clf = _initialize_client_model(n_features=x_client.shape[1], seed=seed, alpha=alpha)

    classes = np.array([0, 1], dtype=int)

    for epoch in range(local_epochs):
        clf.partial_fit(x_client, y_client, classes=classes)

        if enable_fedprox and hasattr(clf, "coef_"):
            clf.coef_ = clf.coef_ - fedprox_mu * (clf.coef_ - global_coef)
            clf.intercept_ = clf.intercept_ - fedprox_mu * (clf.intercept_ - global_intercept)

    return clf.coef_.copy(), clf.intercept_.copy()


def _aggregate_updates(
    client_models: List[Tuple[np.ndarray, np.ndarray]],
    client_sizes: List[int],
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(client_models) == 0:
        raise ValueError("No client models were produced for aggregation.")

    total = float(sum(client_sizes))
    coefs = np.stack([m[0].reshape(-1) for m in client_models], axis=0)
    intercepts = np.stack([m[1].reshape(-1) for m in client_models], axis=0)
    weights = np.array(client_sizes, dtype=np.float64) / max(total, 1.0)

    method = str(method).lower()

    if method == "fedavg":
        agg_coef = np.average(coefs, axis=0, weights=weights)
        agg_intercept = np.average(intercepts, axis=0, weights=weights)
    elif method == "coordinate_median":
        agg_coef = np.median(coefs, axis=0)
        agg_intercept = np.median(intercepts, axis=0)
    elif method == "trimmed_mean":
        trim_ratio = 0.10
        trim_k = int(len(client_models) * trim_ratio)

        def _trimmed_mean(arr: np.ndarray) -> np.ndarray:
            if trim_k <= 0 or (2 * trim_k) >= arr.shape[0]:
                return np.mean(arr, axis=0)
            sorted_arr = np.sort(arr, axis=0)
            trimmed = sorted_arr[trim_k:arr.shape[0] - trim_k]
            return np.mean(trimmed, axis=0)

        agg_coef = _trimmed_mean(coefs)
        agg_intercept = _trimmed_mean(intercepts)
    else:
        agg_coef = np.average(coefs, axis=0, weights=weights)
        agg_intercept = np.average(intercepts, axis=0, weights=weights)

    return agg_coef.reshape(1, -1), agg_intercept.reshape(1)


def _evaluate_linear_model(coef: np.ndarray, intercept: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> Dict:
    logits = np.dot(x_val, coef.reshape(-1)) + float(intercept.reshape(-1)[0])
    probs = _sigmoid(logits)
    return _compute_metrics(y_val, probs)


def _build_shap_like_output(
    feature_names: List[str],
    coef: np.ndarray,
    x_reference: pd.DataFrame,
    cfg: Dict,
) -> Dict:
    top_k = 10
    abs_coef = np.abs(coef.reshape(-1))
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": abs_coef,
        }
    ).sort_values("importance", ascending=False)

    top_features = importance_df.head(top_k).to_dict(orient="records")

    shap_status = "completed" if cfg.get("ENABLE_SHAP", False) else "disabled"

    return {
        "shap_status": shap_status,
        "shap_top_features": top_features,
        "shap_note": (
            "This cloud-safe backend uses coefficient-based feature influence as a SHAP-like explanation layer. "
            "It is designed for reliability on Streamlit Cloud and can later be replaced with full SHAP computation."
        ),
    }


def run_training(config: Dict) -> Tuple[List[Dict], Dict]:
    start_time = time.time()

    data_path = config["DATA_PATH"]
    output_dir = config.get("OUTPUT_DIR", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    rounds = int(config.get("ROUNDS", 20))
    num_clients = int(config.get("NUM_CLIENTS", 10))
    local_epochs = int(config.get("LOCAL_EPOCHS", 3))
    client_fraction = float(config.get("CLIENT_FRACTION", 1.0))
    seed = int(config.get("SEED", 42))
    noniid_dominance = float(config.get("NONIID_LABEL_DOMINANCE", 0.60))
    aggregation_method = str(config.get("AGGREGATION_METHOD", "fedavg"))
    selection_metric = str(config.get("MODEL_SELECTION_METRIC", "pr_auc"))
    early_stopping_patience = int(config.get("EARLY_STOPPING_PATIENCE", 10))
    enable_fedprox = bool(config.get("ENABLE_FEDPROX", True))
    fedprox_mu = 1e-3
    alpha = max(float(config.get("WEIGHT_DECAY", 1e-4)), 1e-6)

    np.random.seed(seed)

    df_raw = pd.read_csv(data_path)
    _validate_dataset(df_raw)

    df = _add_feature_engineering(df_raw, config)
    X_df, y = _build_features(df)

    feature_names = X_df.columns.tolist()

    x_train_df, x_val_df, y_train, y_val = train_test_split(
        X_df,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_df)
    x_val = scaler.transform(x_val_df)

    client_splits = _create_client_splits(
        y=y_train.to_numpy(),
        num_clients=num_clients,
        seed=seed,
        dominance=noniid_dominance,
    )

    rng = np.random.default_rng(seed)

    n_features = x_train.shape[1]
    global_coef = np.zeros((1, n_features), dtype=np.float64)
    global_intercept = np.zeros((1,), dtype=np.float64)

    results: List[Dict] = []
    best_metric = -np.inf
    best_round = 0
    best_state = (global_coef.copy(), global_intercept.copy())
    no_improve_count = 0

    round_metrics_path = os.path.join(output_dir, "round_metrics.csv")
    summary_path = os.path.join(output_dir, "training_summary.json")
    shap_path = os.path.join(output_dir, "shap_top_features.csv")

    for round_id in range(1, rounds + 1):
        round_start = time.time()

        selected_clients = _sample_clients(num_clients, client_fraction, rng)
        local_models: List[Tuple[np.ndarray, np.ndarray]] = []
        local_sizes: List[int] = []

        for client_id in selected_clients:
            idx = client_splits[client_id]
            if len(idx) == 0:
                continue

            x_client = x_train[idx]
            y_client = y_train.to_numpy()[idx]

            if len(np.unique(y_client)) < 2:
                continue

            local_coef, local_intercept = _fit_local_client(
                global_coef=global_coef,
                global_intercept=global_intercept,
                x_client=x_client,
                y_client=y_client,
                local_epochs=local_epochs,
                seed=seed + round_id + client_id,
                alpha=alpha,
                fedprox_mu=fedprox_mu,
                enable_fedprox=enable_fedprox,
            )

            local_models.append((local_coef, local_intercept))
            local_sizes.append(len(idx))

        if len(local_models) > 0:
            global_coef, global_intercept = _aggregate_updates(
                client_models=local_models,
                client_sizes=local_sizes,
                method=aggregation_method,
            )

        metrics = _evaluate_linear_model(global_coef, global_intercept, x_val, y_val.to_numpy())

        row = {
            "round": int(round_id),
            "pr_auc": round(float(metrics["pr_auc"]), 6),
            "roc_auc": round(float(metrics["roc_auc"]), 6),
            "f1": round(float(metrics["f1"]), 6),
            "precision": round(float(metrics["precision"]), 6),
            "recall": round(float(metrics["recall"]), 6),
            "threshold": round(float(metrics["threshold"]), 6),
            "selected_clients": int(len(selected_clients)),
            "round_time_sec": round(float(time.time() - round_start), 4),
        }
        results.append(row)

        current_metric = float(metrics.get(selection_metric, metrics["pr_auc"]))
        if current_metric > best_metric:
            best_metric = current_metric
            best_round = round_id
            best_state = (global_coef.copy(), global_intercept.copy())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stopping_patience:
            break

    global_coef, global_intercept = best_state

    final_metrics = _evaluate_linear_model(global_coef, global_intercept, x_val, y_val.to_numpy())

    shap_output = _build_shap_like_output(
        feature_names=feature_names,
        coef=global_coef,
        x_reference=x_val_df,
        cfg=config,
    )

    summary = {
        "status": "training_completed",
        "model_type": "Federated Linear SGD (cloud-safe backend)",
        "dataset_name": os.path.basename(data_path),
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "target_column": "Class",
        "rounds_requested": int(rounds),
        "rounds_completed": int(len(results)),
        "best_round": int(best_round),
        "num_clients": int(num_clients),
        "selected_metric": selection_metric,
        "aggregation_method": aggregation_method,
        "final_metrics": {
            "pr_auc": round(float(final_metrics["pr_auc"]), 6),
            "roc_auc": round(float(final_metrics["roc_auc"]), 6),
            "f1": round(float(final_metrics["f1"]), 6),
            "precision": round(float(final_metrics["precision"]), 6),
            "recall": round(float(final_metrics["recall"]), 6),
            "threshold": round(float(final_metrics["threshold"]), 6),
        },
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "runtime_sec": round(float(time.time() - start_time), 4),
        "config_used": config,
        **shap_output,
    }

    df_results = pd.DataFrame(results)
    df_results.to_csv(round_metrics_path, index=False)

    if "shap_top_features" in summary:
        pd.DataFrame(summary["shap_top_features"]).to_csv(shap_path, index=False)

    _save_json(summary_path, summary)

    return results, summary
