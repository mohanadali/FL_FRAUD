import os
import io
import math
import json
import time
import hmac
import hashlib
import warnings
import random
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================
@dataclass
class Config:
    # -------------------------
    # Data
    # -------------------------
    DATA_PATH: str = r"C:\tanya thesis\dataset\archive\dataset2\creditcard.csv"
    TARGET_COL: str = "Class"

    SHAP_BACKGROUND_PATH: str = r"C:\tanya thesis\dataset\archive\dataset2\shap_background_256.csv"
    SHAP_EXPLAIN_PATH: str = r"C:\tanya thesis\dataset\archive\dataset2\shap_explain_100.csv"

    OUTPUT_DIR: str = r"C:\Users\ASUS\PycharmProjects\PythonProject3paper1tanya\XAI1944_creditcard"

    # -------------------------
    # Training
    # -------------------------
    BATCH_SIZE: int = 512
    LR: float = 2e-4
    WEIGHT_DECAY: float = 1e-4
    SEED: int = 42
    CLIP_NORM: float = 1.0
    NUM_RUNS: int = 1
    LR_WARMUP_RATIO: float = 0.10

    # -------------------------
    # Loss
    # -------------------------
    LOSS_TYPE: str = "focal"   # "focal" or "bce"
    FOCAL_ALPHA: float = 0.75
    FOCAL_GAMMA: float = 2.0
    LABEL_SMOOTHING: float = 0.0

    # -------------------------
    # FL
    # -------------------------
    NUM_CLIENTS: int = 10
    ROUNDS: int = 50
    LOCAL_EPOCHS: int = 3
    CLIENT_FRACTION: float = 1.0
    EARLY_STOPPING_PATIENCE: int = 10
    NONIID_LABEL_DOMINANCE: float = 0.60

    # -------------------------
    # Server momentum (FedAvgM)
    # -------------------------
    SERVER_USE_MOMENTUM: bool = True
    SERVER_MOMENTUM: float = 0.9

    # -------------------------
    # Model
    # -------------------------
    D_MODEL: int = 128
    N_HEADS: int = 8
    N_LAYERS: int = 3
    DROPOUT: float = 0.10
    MLP_HIDDEN: int = 256

    # -------------------------
    # Security / privacy
    # -------------------------
    ENABLE_TEE_INSPIRED_PARTITION: bool = True
    ENABLE_DP: bool = True
    ENABLE_SIGNING: bool = True
    ENABLE_ENCRYPTION: bool = True
    ENABLE_TRUSTED_UPDATES_ONLY: bool = True
    ENABLE_REPLAY_PROTECTION: bool = True
    ENABLE_UPDATE_ANOMALY_FILTER: bool = False

    # robust aggregation
    AGGREGATION_METHOD: str = "fedavg"  # "fedavg", "trimmed_mean", "coordinate_median"
    TRIM_RATIO: float = 0.1

    # anomaly filtering
    MAX_ROBUST_ZSCORE: float = 5.0
    MIN_COSINE_TO_CENTROID: float = -0.20

    # DP
    DP_CLIP_NORM_SECURE: float = 1.2
    DP_CLIP_NORM_PUBLIC: float = 1.8
    DP_NOISE_MULTIPLIER_SECURE: float = 0.01
    DP_NOISE_MULTIPLIER_PUBLIC: float = 0.005
    DP_DELTA: float = 1e-5

    # FedProx
    ENABLE_FEDPROX: bool = True
    FEDPROX_MU: float = 1e-3

    # 32 bytes hex string for AES-256-GCM
    TRANSPORT_KEY_HEX: str = "7f4c7b2d0b5f8c13344f54c7d2e1a911be2890d2f60a2f30d5a8ee7dfe1ab8c1"

    # -------------------------
    # Feature engineering
    # -------------------------
    ENABLE_FEATURE_ENGINEERING: bool = True
    ENABLE_LOG1P_SKEWED_FEATURES: bool = True
    ENABLE_BINARY_NO_STANDARDIZE: bool = True

    # -------------------------
    # XAI
    # -------------------------
    ENABLE_SHAP: bool = True
    ENABLE_TEE_INDEPENDENT_SHAP: bool = True
    ENABLE_PERMUTATION_IMPORTANCE: bool = True
    SHAP_NSAMPLES: int = 100
    SHAP_TOPK_FEATURES: int = 5
    SHAP_BACKGROUND_LIMIT: int = 128
    SHAP_EXPLAIN_LIMIT: int = 100

    # secure SHAP target
    SECURE_SHAP_MODE: str = "cls_norm"   # "cls_norm", "cls_mean", "cls_dim"
    SECURE_SHAP_DIM_IDX: int = 0

    # -------------------------
    # Model selection
    # -------------------------
    MODEL_SELECTION_METRIC: str = "pr_auc"
    CHECKPOINT_NAME: str = "best_checkpoint.pt"

    # -------------------------
    # Ablations
    # -------------------------
    ENABLE_ABLATION_STUDY: bool = False

    def clone_with(self, **kwargs):
        cfg = Config(**asdict(self))
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg


CFG = Config()


# =========================================================
# PATHS
# =========================================================
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

RUNS_SUMMARY_JSON = os.path.join(CFG.OUTPUT_DIR, "runs_summary.json")
PREPROCESS_META_JSON = os.path.join(CFG.OUTPUT_DIR, "preprocess_meta.json")
SECURITY_RESULTS_CSV = os.path.join(CFG.OUTPUT_DIR, "security_round_results.csv")
SECURITY_SUMMARY_JSON = os.path.join(CFG.OUTPUT_DIR, "security_summary.json")
ROUND_METRICS_CSV = os.path.join(CFG.OUTPUT_DIR, "round_metrics.csv")
CLIENT_METRICS_CSV = os.path.join(CFG.OUTPUT_DIR, "client_metrics.csv")
BEST_MODEL_PATH = os.path.join(CFG.OUTPUT_DIR, "best_global_model.pt")
FINAL_MODEL_PATH = os.path.join(CFG.OUTPUT_DIR, "final_global_model.pt")
BEST_CHECKPOINT_PATH = os.path.join(CFG.OUTPUT_DIR, CFG.CHECKPOINT_NAME)
LOG_PATH = os.path.join(CFG.OUTPUT_DIR, "training.log")
ABLATION_RESULTS_JSON = os.path.join(CFG.OUTPUT_DIR, "ablation_results.json")

CLIENT_KEYS = {
    f"client_{i}": f"client_key_{i}_secure_demo".encode("utf-8")
    for i in range(CFG.NUM_CLIENTS)
}


# =========================================================
# LOGGING
# =========================================================
def setup_logging():
    logger = logging.getLogger("FL_SECURE_XAI")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


LOGGER = setup_logging()


# =========================================================
# UTILS
# =========================================================
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, bytes):
        return obj.hex()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    else:
        return obj


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, indent=2)


# =========================================================
# REPRODUCIBILITY
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# DATASET
# =========================================================
class FraudDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.from_numpy(X_num).float()
        self.X_cat = torch.from_numpy(X_cat).long()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def add_engineered_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not cfg.ENABLE_FEATURE_ENGINEERING:
        return df

    df = df.copy()

    if "Amount" in df.columns:
        amount_nonneg = np.maximum(pd.to_numeric(df["Amount"], errors="coerce").fillna(0), 0)
        df["fe_log_amount"] = np.log1p(amount_nonneg)
        df["fe_amount_is_zero"] = (amount_nonneg == 0).astype(int)

    if "Time" in df.columns:
        time_num = pd.to_numeric(df["Time"], errors="coerce").fillna(0)
        df["fe_time_hours"] = time_num / 3600.0
        df["fe_time_days"] = time_num / 86400.0
        df["fe_time_mod_day"] = time_num % 86400.0
        df["fe_time_sin_day"] = np.sin(2 * np.pi * df["fe_time_mod_day"] / 86400.0)
        df["fe_time_cos_day"] = np.cos(2 * np.pi * df["fe_time_mod_day"] / 86400.0)

    if "Amount" in df.columns and "Time" in df.columns:
        amount_num = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
        time_num = pd.to_numeric(df["Time"], errors="coerce").fillna(0)
        df["fe_amount_over_time"] = amount_num / (time_num + 1.0)

    return df


# =========================================================
# MODEL
# =========================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms) * self.weight


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "D_MODEL must be divisible by N_HEADS"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_flash:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = torch.softmax(scores.float(), dim=-1).type_as(q)
            scores = self.attn_dropout(scores)
            out = torch.matmul(scores, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        out = self.wo(out)
        out = self.resid_dropout(out)
        return out


class GatedFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.0, multiple_of: int = 32):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * d_model
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ff_multiple_of: int = 32, norm_eps: float = 1e-6):
        super().__init__()
        self.attention_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attention = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.feed_forward = GatedFeedForward(
            d_model=d_model,
            hidden_dim=None,
            dropout=dropout,
            multiple_of=ff_multiple_of
        )

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class TransformerStack(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                ff_multiple_of=32,
                norm_eps=1e-6
            )
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model, eps=1e-6)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x


class SecureFeatureExtractor(nn.Module):
    def __init__(self, n_num, cat_sizes, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_num = n_num
        self.n_cat = len(cat_sizes)

        self.num_tokenizers = nn.ModuleList(
            [nn.Linear(1, cfg.D_MODEL) for _ in range(n_num)]
        )
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(size + 1, cfg.D_MODEL) for size in cat_sizes]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.D_MODEL))
        self.input_dropout = nn.Dropout(cfg.DROPOUT)

        secure_layers = 1 if cfg.N_LAYERS >= 1 else 0
        if secure_layers > 0:
            self.transformer_secure = TransformerStack(
                n_layers=secure_layers,
                d_model=cfg.D_MODEL,
                n_heads=cfg.N_HEADS,
                dropout=cfg.DROPOUT
            )
        else:
            self.transformer_secure = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x_num, x_cat):
        tokens = []

        for i in range(self.n_num):
            tok = self.num_tokenizers[i](x_num[:, i:i + 1])
            tokens.append(tok)

        for i in range(self.n_cat):
            tok = self.cat_embeddings[i](x_cat[:, i])
            tokens.append(tok)

        x = torch.stack(tokens, dim=1)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.input_dropout(x)
        x = self.transformer_secure(x)
        return x


class PublicPredictor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        public_layers = max(cfg.N_LAYERS - 1, 0)

        if public_layers > 0:
            self.transformer_public = TransformerStack(
                n_layers=public_layers,
                d_model=cfg.D_MODEL,
                n_heads=cfg.N_HEADS,
                dropout=cfg.DROPOUT
            )
        else:
            self.transformer_public = nn.Identity()

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.D_MODEL),
            nn.Linear(cfg.D_MODEL, cfg.MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.MLP_HIDDEN, cfg.MLP_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.MLP_HIDDEN // 2, 1)
        )

    def forward(self, x):
        x = self.transformer_public(x)
        logits = self.head(x[:, 0]).squeeze(-1)
        return logits


class HybridFraudTransformer(nn.Module):
    """
    TEE-inspired prototype partition only.
    This is NOT a real hardware TEE deployment.
    """
    def __init__(self, n_num, cat_sizes, cfg: Config):
        super().__init__()
        self.secure_part = SecureFeatureExtractor(n_num=n_num, cat_sizes=cat_sizes, cfg=cfg)
        self.public_part = PublicPredictor(cfg=cfg)

    def forward(self, x_num, x_cat):
        z = self.secure_part(x_num, x_cat)
        logits = self.public_part(z)
        return logits


# =========================================================
# LOSS
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = self.alpha * ((1.0 - p_t) ** self.gamma)
        return (focal_weight * bce).mean()


def build_criterion(cfg: Config, pos_weight: torch.Tensor):
    if cfg.LOSS_TYPE.lower() == "focal":
        return FocalLoss(
            alpha=cfg.FOCAL_ALPHA,
            gamma=cfg.FOCAL_GAMMA,
            label_smoothing=cfg.LABEL_SMOOTHING
        )
    else:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# =========================================================
# SCHEDULER
# =========================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(warmup_steps, 1))
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================================================
# PREPROCESSING
# =========================================================
def validate_schema(df: pd.DataFrame, cfg: Config):
    if not os.path.exists(cfg.DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {cfg.DATA_PATH}")

    if cfg.TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{cfg.TARGET_COL}' not found in dataset.")

    if len(df) == 0:
        raise ValueError("Dataset is empty.")

    if df[cfg.TARGET_COL].isna().all():
        raise ValueError("Target column contains only NaN values.")

    LOGGER.info(f"Dataset shape: {df.shape}")
    LOGGER.info(f"Columns count: {len(df.columns)}")


def detect_columns(df, cfg: Config):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cfg.TARGET_COL in num_cols:
        num_cols.remove(cfg.TARGET_COL)

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if cfg.TARGET_COL in cat_cols:
        cat_cols.remove(cfg.TARGET_COL)

    return num_cols, cat_cols


def detect_binary_numeric_columns(df: pd.DataFrame, num_cols: List[str]) -> List[str]:
    binary_cols = []
    for c in num_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
        if len(vals) <= 2 and set(np.round(vals, 6).tolist()).issubset({0.0, 1.0}):
            binary_cols.append(c)
    return binary_cols


def select_log1p_columns(num_cols: List[str]) -> List[str]:
    keywords = [
        "amount", "count", "income", "limit", "risk", "minutes", "days", "time"
    ]
    out = []
    for c in num_cols:
        cl = c.lower()
        if any(k in cl for k in keywords):
            out.append(c)
    return out


def preprocess_train_fit(train_df, num_cols, cat_cols, cfg: Config):
    X_num_df = train_df[num_cols].copy()
    missing_num_flags = pd.DataFrame(index=train_df.index)

    binary_num_cols = detect_binary_numeric_columns(train_df, num_cols) if cfg.ENABLE_BINARY_NO_STANDARDIZE else []
    log1p_cols = select_log1p_columns(num_cols) if cfg.ENABLE_LOG1P_SKEWED_FEATURES else []

    for c in num_cols:
        X_num_df[c] = pd.to_numeric(X_num_df[c], errors="coerce")
        X_num_df[c] = X_num_df[c].replace([np.inf, -np.inf], np.nan)

        if c in log1p_cols:
            X_num_df[c] = np.log1p(np.maximum(X_num_df[c], 0))

        missing_num_flags[f"{c}_was_missing"] = X_num_df[c].isna().astype(int)

    X_num_aug = pd.concat([X_num_df, missing_num_flags], axis=1)

    medians = X_num_aug.median(numeric_only=True)
    X_num_aug = X_num_aug.fillna(medians)

    means = X_num_aug.mean()
    stds = X_num_aug.std().replace(0, 1.0)

    for c in binary_num_cols:
        if c in X_num_aug.columns:
            means[c] = 0.0
            stds[c] = 1.0

    for c in missing_num_flags.columns:
        means[c] = 0.0
        stds[c] = 1.0

    X_num = ((X_num_aug - means) / stds).values.astype(np.float32)

    cat_maps = {}
    cat_sizes = []
    X_cat_list = []
    for c in cat_cols:
        vals = train_df[c].fillna("__MISSING__").astype(str)
        uniq = vals.unique().tolist()
        mapping = {v: i + 1 for i, v in enumerate(uniq)}
        cat_maps[c] = mapping
        enc = vals.map(mapping).fillna(0).astype(np.int64).values
        X_cat_list.append(enc)
        cat_sizes.append(len(mapping))

    if len(X_cat_list) > 0:
        X_cat = np.stack(X_cat_list, axis=1).astype(np.int64)
    else:
        X_cat = np.zeros((len(train_df), 0), dtype=np.int64)

    y = pd.to_numeric(train_df[cfg.TARGET_COL], errors="coerce").fillna(0).astype(int).values

    meta = {
        "num_cols_original": num_cols,
        "num_cols_augmented": list(X_num_aug.columns),
        "cat_cols": cat_cols,
        "medians": medians.to_dict(),
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "cat_maps": cat_maps,
        "cat_sizes": cat_sizes,
        "binary_num_cols": binary_num_cols,
        "log1p_cols": log1p_cols
    }
    return X_num, X_cat, y, meta


def preprocess_apply(df, meta, cfg: Config):
    num_cols = meta["num_cols_original"]
    cat_cols = meta["cat_cols"]

    X_num_df = df[num_cols].copy()
    missing_num_flags = pd.DataFrame(index=df.index)

    binary_num_cols = meta.get("binary_num_cols", [])
    log1p_cols = meta.get("log1p_cols", [])

    for c in num_cols:
        X_num_df[c] = pd.to_numeric(X_num_df[c], errors="coerce")
        X_num_df[c] = X_num_df[c].replace([np.inf, -np.inf], np.nan)

        if c in log1p_cols:
            X_num_df[c] = np.log1p(np.maximum(X_num_df[c], 0))

        missing_num_flags[f"{c}_was_missing"] = X_num_df[c].isna().astype(int)

    X_num_aug = pd.concat([X_num_df, missing_num_flags], axis=1)
    X_num_aug = X_num_aug.reindex(columns=meta["num_cols_augmented"], fill_value=0)

    medians = pd.Series(meta["medians"])
    means = pd.Series(meta["means"])
    stds = pd.Series(meta["stds"]).replace(0, 1.0)

    X_num_aug = X_num_aug.fillna(medians)

    for c in binary_num_cols:
        if c in X_num_aug.columns:
            means[c] = 0.0
            stds[c] = 1.0

    for c in missing_num_flags.columns:
        means[c] = 0.0
        stds[c] = 1.0

    X_num = ((X_num_aug - means) / stds).values.astype(np.float32)

    X_cat_list = []
    for c in cat_cols:
        mapping = meta["cat_maps"][c]
        vals = df[c].fillna("__MISSING__").astype(str)
        enc = vals.map(mapping).fillna(0).astype(np.int64).values
        X_cat_list.append(enc)

    if len(X_cat_list) > 0:
        X_cat = np.stack(X_cat_list, axis=1).astype(np.int64)
    else:
        X_cat = np.zeros((len(df), 0), dtype=np.int64)

    y = pd.to_numeric(df[cfg.TARGET_COL], errors="coerce").fillna(0).astype(int).values
    return X_num, X_cat, y


# =========================================================
# METRICS
# =========================================================
def safe_auc(y_true, probs):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    mask = np.isfinite(y_true) & np.isfinite(probs)
    y_true = y_true[mask]
    probs = probs[mask]
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probs))


def safe_ap(y_true, probs):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    mask = np.isfinite(y_true) & np.isfinite(probs)
    y_true = y_true[mask]
    probs = probs[mask]
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, probs))


def find_best_threshold(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 19)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}

    for th in thresholds:
        preds = (probs >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {
                "threshold": float(th),
                "f1": float(f1),
                "precision": float(p),
                "recall": float(r)
            }
    return best


def compute_metrics(y_true, probs, threshold=0.5):
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    preds = (probs >= threshold).astype(int)

    auc = safe_auc(y_true, probs)
    ap = safe_ap(y_true, probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(auc),
        "pr_auc": float(ap),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =========================================================
# TRAIN / EVAL
# =========================================================
def make_weighted_sampler(y):
    class_counts = np.bincount(y.astype(int), minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y.astype(int)]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )


def train_one_epoch(model, loader, optimizer, criterion, device, cfg: Config, global_model=None, scheduler=None):
    model.train()
    epoch_loss = 0.0
    steps = 0

    global_params = None
    if cfg.ENABLE_FEDPROX and global_model is not None:
        global_params = {k: v.detach().clone().to(device) for k, v in global_model.state_dict().items()}

    for xnum, xcat, y in loader:
        xnum = xnum.to(device)
        xcat = xcat.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xnum, xcat)
        loss = criterion(logits, y)

        if cfg.ENABLE_FEDPROX and global_params is not None:
            prox_term = 0.0
            for name, param in model.named_parameters():
                prox_term = prox_term + torch.sum((param - global_params[name]) ** 2)
            loss = loss + (cfg.FEDPROX_MU / 2.0) * prox_term

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_loss += float(loss.item())
        steps += 1

    return epoch_loss / max(steps, 1)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    ys, ps = [], []

    for xnum, xcat, y in loader:
        xnum = xnum.to(device)
        xcat = xcat.to(device)
        logits = model(xnum, xcat)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        ys.append(y.numpy())
        ps.append(probs)

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return ys, ps


# =========================================================
# STATE / AGG HELPERS
# =========================================================
def tensor_state_clone_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def zeros_like_state_dict(state_dict):
    return {k: torch.zeros_like(v.float()) for k, v in state_dict.items()}


def state_dict_difference(local_state, global_state):
    return {k: local_state[k].float() - global_state[k].float() for k in local_state}


def state_dict_add(base_state, delta_state):
    return {k: base_state[k].float() + delta_state[k].float() for k in base_state}


def compute_state_l2_norm(state_dict):
    total_sq = 0.0
    for v in state_dict.values():
        total_sq += torch.sum(v.float() ** 2).item()
    return math.sqrt(total_sq + 1e-12)


def flatten_state_dict(state_dict):
    vecs = []
    for k in sorted(state_dict.keys()):
        vecs.append(state_dict[k].float().reshape(-1))
    if len(vecs) == 0:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(vecs, dim=0)


def split_secure_public_layers(state_dict):
    secure_dict = {}
    public_dict = {}
    for k, v in state_dict.items():
        if k.startswith("secure_part."):
            secure_dict[k] = v
        else:
            public_dict[k] = v
    return secure_dict, public_dict


def fedavg_state_dict(client_states, client_sizes):
    total = float(sum(client_sizes))
    new_state = {}
    keys = client_states[0].keys()

    for k in keys:
        acc = None
        for st, sz in zip(client_states, client_sizes):
            w = sz / total
            t = st[k].float() * w
            acc = t if acc is None else acc + t
        new_state[k] = acc
    return new_state


def coordinate_median_state_dict(client_states):
    new_state = {}
    keys = client_states[0].keys()
    for k in keys:
        stacked = torch.stack([st[k].float() for st in client_states], dim=0)
        new_state[k] = torch.median(stacked, dim=0).values
    return new_state


def trimmed_mean_state_dict(client_states, trim_ratio=0.1):
    new_state = {}
    keys = client_states[0].keys()
    n = len(client_states)
    trim_k = int(n * trim_ratio)

    for k in keys:
        stacked = torch.stack([st[k].float() for st in client_states], dim=0)
        if trim_k > 0 and 2 * trim_k < n:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_k:n - trim_k]
            new_state[k] = trimmed.mean(dim=0)
        else:
            new_state[k] = stacked.mean(dim=0)
    return new_state


def robust_aggregate(client_states, client_sizes, cfg: Config):
    if cfg.AGGREGATION_METHOD == "fedavg":
        return fedavg_state_dict(client_states, client_sizes)
    elif cfg.AGGREGATION_METHOD == "coordinate_median":
        return coordinate_median_state_dict(client_states)
    elif cfg.AGGREGATION_METHOD == "trimmed_mean":
        return trimmed_mean_state_dict(client_states, trim_ratio=cfg.TRIM_RATIO)
    else:
        raise ValueError(f"Unknown AGGREGATION_METHOD: {cfg.AGGREGATION_METHOD}")


def apply_server_momentum(old_state, aggregated_state, velocity_state, momentum):
    if velocity_state is None:
        velocity_state = zeros_like_state_dict(old_state)

    new_velocity = {}
    new_state = {}
    for k in aggregated_state:
        delta = aggregated_state[k].float() - old_state[k].float()
        v = momentum * velocity_state[k].float() + delta
        new_velocity[k] = v
        new_state[k] = old_state[k].float() + v
    return new_state, new_velocity


# =========================================================
# DP ACCOUNTANT
# =========================================================
class SimpleRDPAccountant:
    def __init__(self, noise_multiplier: float, sample_rate: float, delta: float):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.steps = 0
        self.orders = [1.25, 1.5, 2, 3, 4, 5, 8, 10, 16, 32, 64, 128]

    def step(self):
        self.steps += 1

    def get_epsilon(self):
        if self.noise_multiplier <= 0:
            return float("inf")

        q = max(min(self.sample_rate, 1.0), 1e-12)
        sigma = max(self.noise_multiplier, 1e-12)
        delta = max(self.delta, 1e-12)

        eps_candidates = []
        for alpha in self.orders:
            rdp = self.steps * (q ** 2) * alpha / (2 * sigma ** 2)
            eps = rdp + math.log(1 / delta) / max(alpha - 1, 1e-12)
            eps_candidates.append(float(eps))
        return min(eps_candidates)


# =========================================================
# SECURITY HELPERS
# =========================================================
def serialize_state_dict_npz(state_dict):
    arr_dict = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arr_dict)
    return buffer.getvalue()


def deserialize_state_dict_npz(raw_bytes):
    buffer = io.BytesIO(raw_bytes)
    loaded = np.load(buffer, allow_pickle=False)
    state_dict = {k: torch.tensor(loaded[k]) for k in loaded.files}
    return state_dict


def sign_bytes(raw_bytes, client_id):
    key = CLIENT_KEYS[client_id]
    return hmac.new(key, raw_bytes, hashlib.sha256).hexdigest()


def verify_bytes_signature(raw_bytes, signature, client_id):
    expected = sign_bytes(raw_bytes, client_id)
    return hmac.compare_digest(signature, expected)


def aesgcm_encrypt(raw_bytes, round_id, client_id, cfg: Config):
    key = bytes.fromhex(cfg.TRANSPORT_KEY_HEX)
    aesgcm = AESGCM(key)
    nonce_material = hashlib.sha256(f"{client_id}_{round_id}".encode("utf-8")).digest()
    nonce = nonce_material[:12]
    aad = f"{client_id}|{round_id}".encode("utf-8")
    ciphertext = aesgcm.encrypt(nonce, raw_bytes, aad)
    return {
        "nonce_hex": nonce.hex(),
        "aad_hex": aad.hex(),
        "ciphertext_hex": ciphertext.hex()
    }


def aesgcm_decrypt(enc_package, round_id, client_id, cfg: Config):
    key = bytes.fromhex(cfg.TRANSPORT_KEY_HEX)
    aesgcm = AESGCM(key)
    nonce = bytes.fromhex(enc_package["nonce_hex"])
    aad = bytes.fromhex(enc_package["aad_hex"])
    ciphertext = bytes.fromhex(enc_package["ciphertext_hex"])
    raw_bytes = aesgcm.decrypt(nonce, ciphertext, aad)
    return raw_bytes


def clip_noise_single_group(diff_state, clip_norm, noise_multiplier):
    pre_clip_norm = compute_state_l2_norm(diff_state)
    clip_scale = min(1.0, clip_norm / max(pre_clip_norm, 1e-12))

    noisy_diff = {}
    noise_norm_sq = 0.0
    for k, v in diff_state.items():
        clipped = v.float() * clip_scale
        noise = torch.randn_like(clipped) * (clip_norm * noise_multiplier)
        noisy = clipped + noise
        noisy_diff[k] = noisy
        noise_norm_sq += torch.sum(noise ** 2).item()

    post_clip_norm = compute_state_l2_norm({k: v.float() * clip_scale for k, v in diff_state.items()})
    noisy_norm = compute_state_l2_norm(noisy_diff)
    noise_norm = math.sqrt(noise_norm_sq + 1e-12)

    return noisy_diff, {
        "pre_clip_norm": float(pre_clip_norm),
        "clip_scale": float(clip_scale),
        "post_clip_norm": float(post_clip_norm),
        "noise_multiplier": float(noise_multiplier),
        "noise_norm": float(noise_norm),
        "noisy_update_norm": float(noisy_norm),
    }


def clip_and_add_noise_layerwise(diff_state, cfg: Config):
    secure_state, public_state = split_secure_public_layers(diff_state)

    secure_noisy, secure_info = clip_noise_single_group(
        secure_state, cfg.DP_CLIP_NORM_SECURE, cfg.DP_NOISE_MULTIPLIER_SECURE
    ) if len(secure_state) > 0 else ({}, {})

    public_noisy, public_info = clip_noise_single_group(
        public_state, cfg.DP_CLIP_NORM_PUBLIC, cfg.DP_NOISE_MULTIPLIER_PUBLIC
    ) if len(public_state) > 0 else ({}, {})

    noisy_diff = {}
    noisy_diff.update(secure_noisy)
    noisy_diff.update(public_noisy)

    dp_info = {
        "secure": secure_info,
        "public": public_info,
        "total_noisy_update_norm": float(compute_state_l2_norm(noisy_diff)),
    }
    return noisy_diff, dp_info


def package_client_update(state_dict, client_id, round_id, dp_info, stage_info, cfg: Config):
    raw_bytes = serialize_state_dict_npz(state_dict)
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    payload_size_bytes = len(raw_bytes)

    if cfg.ENABLE_SIGNING:
        signature = sign_bytes(raw_bytes, client_id)
    else:
        signature = "SIGNING_DISABLED"

    if cfg.ENABLE_ENCRYPTION:
        payload = aesgcm_encrypt(raw_bytes, round_id, client_id, cfg)
        encrypted = True
    else:
        payload = {"raw_hex": raw_bytes.hex()}
        encrypted = False

    package = {
        "client_id": client_id,
        "round_id": round_id,
        "payload": payload,
        "payload_sha256": raw_hash,
        "signature": signature,
        "encrypted_transport": encrypted,
        "dp_info": dp_info,
        "stage_info": stage_info,
        "payload_size_bytes": int(payload_size_bytes),
        "package_hash_id": hashlib.sha256(f"{client_id}_{round_id}_{raw_hash}".encode("utf-8")).hexdigest(),
    }
    return package


def unpack_and_verify_client_update(package, seen_package_ids, cfg: Config):
    client_id = package["client_id"]
    round_id = package["round_id"]
    package_hash_id = package["package_hash_id"]

    replay_ok = True
    if cfg.ENABLE_REPLAY_PROTECTION:
        if package_hash_id in seen_package_ids:
            replay_ok = False
        else:
            seen_package_ids.add(package_hash_id)

    if package["encrypted_transport"]:
        raw_bytes = aesgcm_decrypt(package["payload"], round_id, client_id, cfg)
    else:
        raw_bytes = bytes.fromhex(package["payload"]["raw_hex"])

    recv_hash = hashlib.sha256(raw_bytes).hexdigest()
    hash_ok = recv_hash == package["payload_sha256"]

    if cfg.ENABLE_SIGNING:
        sig_ok = verify_bytes_signature(raw_bytes, package["signature"], client_id)
    else:
        sig_ok = True

    trust_ok = bool(hash_ok and sig_ok and replay_ok)

    state_dict = deserialize_state_dict_npz(raw_bytes)

    verify_info = {
        "hash_ok": bool(hash_ok),
        "signature_ok": bool(sig_ok),
        "replay_ok": bool(replay_ok),
        "trusted_update": bool(trust_ok)
    }
    return state_dict, verify_info


# =========================================================
# TEE-STYLE STAGES
# =========================================================
def tee1_preprocess_client_data(train_xn, train_xc, train_y, cidx):
    t0 = time.time()
    c_xn = train_xn[cidx]
    c_xc = train_xc[cidx]
    c_y = train_y[cidx].astype(np.float32)
    stage_time = time.time() - t0

    stage_info = {
        "tee1_samples": int(len(cidx)),
        "tee1_time_sec": float(stage_time)
    }
    return c_xn, c_xc, c_y, stage_info


def tee2_train_client(global_model, c_xn, c_xc, c_y, n_num, cat_sizes, criterion, device, cfg: Config):
    t0 = time.time()

    sampler = make_weighted_sampler(c_y)
    loader = DataLoader(
        FraudDataset(c_xn, c_xc, c_y),
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        shuffle=False
    )

    client_model = HybridFraudTransformer(n_num=n_num, cat_sizes=cat_sizes, cfg=cfg).to(device)
    client_model.load_state_dict({k: v.detach().clone() for k, v in global_model.state_dict().items()})

    optimizer = torch.optim.AdamW(client_model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    total_local_steps = max(len(loader) * cfg.LOCAL_EPOCHS, 1)
    warmup_steps = max(int(total_local_steps * cfg.LR_WARMUP_RATIO), 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_local_steps
    )

    epoch_losses = []
    for _ in range(cfg.LOCAL_EPOCHS):
        loss_value = train_one_epoch(
            client_model,
            loader,
            optimizer,
            criterion,
            device,
            cfg,
            global_model=global_model,
            scheduler=scheduler
        )
        epoch_losses.append(loss_value)

    local_state = tensor_state_clone_cpu(client_model.state_dict())

    secure_layers = [
        "secure_part.num_tokenizers",
        "secure_part.cat_embeddings",
        "secure_part.cls_token",
        "secure_part.transformer_secure"
    ]
    public_layers = [
        "public_part.transformer_public",
        "public_part.head"
    ]

    stage_time = time.time() - t0

    stage_info = {
        "tee2_time_sec": float(stage_time),
        "tee2_mean_loss": float(np.mean(epoch_losses) if len(epoch_losses) > 0 else 0.0),
        "hybrid_secure_layers": secure_layers,
        "hybrid_public_layers": public_layers,
        "tee_partition_note": "TEE-inspired software partition only, not a real hardware enclave deployment.",
        "hybrid_enabled": bool(cfg.ENABLE_TEE_INSPIRED_PARTITION)
    }

    del client_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return local_state, stage_info


def tee3_protect_and_package_update(local_state, global_state, client_id, round_id, cfg: Config):
    t0 = time.time()

    raw_delta = state_dict_difference(local_state, global_state)
    raw_update_norm = compute_state_l2_norm(raw_delta)

    if cfg.ENABLE_DP:
        protected_delta, dp_info = clip_and_add_noise_layerwise(raw_delta, cfg)
    else:
        protected_delta = raw_delta
        dp_info = {
            "secure": {},
            "public": {},
            "total_noisy_update_norm": float(raw_update_norm)
        }

    protected_state = state_dict_add(global_state, protected_delta)
    packaged = package_client_update(
        state_dict=protected_state,
        client_id=client_id,
        round_id=round_id,
        dp_info=dp_info,
        stage_info={},
        cfg=cfg
    )

    stage_time = time.time() - t0
    stage_info = {
        "tee3_time_sec": float(stage_time),
        "raw_update_norm": float(raw_update_norm),
        "protected_update_norm": float(dp_info["total_noisy_update_norm"]),
        "transport_encrypted": bool(cfg.ENABLE_ENCRYPTION),
        "signed_update": bool(cfg.ENABLE_SIGNING),
        "payload_size_bytes": int(packaged["payload_size_bytes"])
    }
    return packaged, stage_info


# =========================================================
# NON-IID PARTITION
# =========================================================
def create_noniid_client_splits(y, num_clients, seed, dominance=0.60, min_pos_per_client=10):
    """
    Create non-IID client splits while ensuring every client receives
    at least `min_pos_per_client` positive samples if possible.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)

    idx_neg = np.where(y == 0)[0]
    idx_pos = np.where(y == 1)[0]

    rng.shuffle(idx_neg)
    rng.shuffle(idx_pos)

    n_total = len(y)
    base_size = n_total // num_clients
    sizes = [base_size] * num_clients
    sizes[-1] += n_total - sum(sizes)

    total_pos = len(idx_pos)
    feasible_min_pos = min(min_pos_per_client, max(total_pos // num_clients, 0))

    splits = []
    neg_ptr = 0
    pos_ptr = 0

    client_pos_counts = [feasible_min_pos] * num_clients
    assigned_pos = feasible_min_pos * num_clients
    remaining_pos = total_pos - assigned_pos

    for cid in range(num_clients):
        size = sizes[cid]
        dominant_class = cid % 2

        current_pos = client_pos_counts[cid]
        remaining_slots = max(size - current_pos, 0)

        if dominant_class == 1:
            extra_pos = int(remaining_slots * dominance)
        else:
            extra_pos = int(remaining_slots * (1.0 - dominance))

        extra_pos = min(extra_pos, remaining_pos)
        client_pos_counts[cid] += extra_pos
        remaining_pos -= extra_pos

    cid_cycle = 0
    while remaining_pos > 0:
        if client_pos_counts[cid_cycle] < sizes[cid_cycle]:
            client_pos_counts[cid_cycle] += 1
            remaining_pos -= 1
        cid_cycle = (cid_cycle + 1) % num_clients

    for cid in range(num_clients):
        size = sizes[cid]
        pos_count = client_pos_counts[cid]
        neg_count = size - pos_count

        chosen_pos = idx_pos[pos_ptr:pos_ptr + pos_count]
        pos_ptr += len(chosen_pos)

        chosen_neg = idx_neg[neg_ptr:neg_ptr + neg_count]
        neg_ptr += len(chosen_neg)

        current = np.concatenate([chosen_pos, chosen_neg])
        rng.shuffle(current)
        splits.append(current)

    leftovers = []
    if pos_ptr < len(idx_pos):
        leftovers.extend(idx_pos[pos_ptr:].tolist())
    if neg_ptr < len(idx_neg):
        leftovers.extend(idx_neg[neg_ptr:].tolist())

    rng.shuffle(leftovers)
    for i, extra_idx in enumerate(leftovers):
        splits[i % num_clients] = np.append(splits[i % num_clients], extra_idx)

    for i in range(num_clients):
        rng.shuffle(splits[i])

    return splits


def sample_clients(num_clients, fraction, rng):
    k = max(1, int(math.ceil(num_clients * fraction)))
    selected = rng.choice(np.arange(num_clients), size=k, replace=False)
    return sorted(selected.tolist())


# =========================================================
# EVAL HELPERS
# =========================================================
def evaluate_model_full(model, loader, device, threshold):
    y_true, probs = predict_probs(model, loader, device)
    metrics = compute_metrics(y_true, probs, threshold)
    return metrics, y_true, probs


def evaluate_per_client(global_model, train_xn, train_xc, train_y, client_splits, threshold, device, cfg: Config):
    client_rows = []
    for cid, cidx in enumerate(client_splits):
        loader = DataLoader(
            FraudDataset(train_xn[cidx], train_xc[cidx], train_y[cidx]),
            batch_size=cfg.BATCH_SIZE,
            shuffle=False
        )
        metrics, _, _ = evaluate_model_full(global_model, loader, device, threshold)
        metrics["client_id"] = f"client_{cid}"
        metrics["samples"] = int(len(cidx))
        metrics["positive_rate"] = float(np.mean(train_y[cidx])) if len(cidx) > 0 else 0.0
        client_rows.append(metrics)
    return client_rows


# =========================================================
# ANOMALY FILTERING
# =========================================================
def robust_mad_zscores(values):
    values = np.asarray(values, dtype=np.float64)
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + 1e-12
    z = 0.6745 * (values - med) / mad
    return np.abs(z)


def cosine_similarity_torch(a, b):
    a = a.float()
    b = b.float()
    denom = (torch.norm(a) * torch.norm(b)).item()
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


# =========================================================
# XAI
# =========================================================
class WrappedModel(nn.Module):
    def __init__(self, base_model, n_num, n_cat):
        super().__init__()
        self.base_model = base_model
        self.n_num = n_num
        self.n_cat = n_cat

    def forward(self, x):
        x_num = x[:, :self.n_num].float()
        if self.n_cat > 0:
            x_cat = x[:, self.n_num:self.n_num + self.n_cat].long()
        else:
            x_cat = torch.zeros((x.shape[0], 0), dtype=torch.long, device=x.device)

        logits = self.base_model(x_num, x_cat)
        probs = torch.sigmoid(logits)
        return probs.unsqueeze(1)


class SecurePartWrapper(nn.Module):
    """
    Explain the secure/TTE-inspired part independently.

    Input:
        combined tabular input = [numeric features | categorical features]

    Output:
        one scalar target derived from secure CLS representation.
    """
    def __init__(self, base_model, n_num, n_cat, mode="cls_norm", dim_idx=0):
        super().__init__()
        self.base_model = base_model
        self.n_num = n_num
        self.n_cat = n_cat
        self.mode = mode
        self.dim_idx = dim_idx

    def forward(self, x):
        x_num = x[:, :self.n_num].float()
        if self.n_cat > 0:
            x_cat = x[:, self.n_num:self.n_num + self.n_cat].long()
        else:
            x_cat = torch.zeros((x.shape[0], 0), dtype=torch.long, device=x.device)

        z = self.base_model.secure_part(x_num, x_cat)   # [B, seq_len, d_model]
        cls_vec = z[:, 0, :]                            # [B, d_model]

        if self.mode == "cls_norm":
            out = torch.norm(cls_vec, dim=1, keepdim=True)
        elif self.mode == "cls_mean":
            out = cls_vec.mean(dim=1, keepdim=True)
        elif self.mode == "cls_dim":
            out = cls_vec[:, self.dim_idx:self.dim_idx + 1]
        else:
            raise ValueError(f"Unknown secure wrapper mode: {self.mode}")

        return out


class PublicPartWrapper(nn.Module):
    """
    Explain the public predictor independently.

    Input:
        secure CLS latent vector only

    Output:
        final fraud probability
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)   # [B, 1, d_model]

        logits = self.base_model.public_part(x)
        probs = torch.sigmoid(logits)
        return probs.unsqueeze(1)


@torch.no_grad()
def extract_secure_cls_embeddings(model, x_num, x_cat, device, batch_size=512):
    model.eval()
    ds = FraudDataset(x_num, x_cat, np.zeros(len(x_num), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_cls = []
    for xb_num, xb_cat, _ in loader:
        xb_num = xb_num.to(device)
        xb_cat = xb_cat.to(device)

        z = model.secure_part(xb_num, xb_cat)
        cls_vec = z[:, 0, :].detach().cpu().numpy()
        all_cls.append(cls_vec)

    return np.concatenate(all_cls, axis=0)


def save_class_specific_shap_outputs(shap_values, ex_y, feature_names, output_dir):
    shap_values = np.asarray(shap_values)
    ex_y = np.asarray(ex_y)

    if len(shap_values) != len(ex_y):
        return

    for class_value, class_name in [(0, "nonfraud"), (1, "fraud")]:
        mask = ex_y == class_value
        if np.sum(mask) == 0:
            continue
        importance = np.abs(shap_values[mask]).mean(axis=0)
        df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": importance
        }).sort_values("mean_abs_shap", ascending=False)
        df.to_csv(os.path.join(output_dir, f"shap_feature_importance_{class_name}.csv"), index=False)


def run_shap_analysis(global_model, meta, cfg: Config):
    if not cfg.ENABLE_SHAP:
        LOGGER.info("SHAP analysis is disabled.")
        return

    try:
        LOGGER.info("Starting improved SHAP analysis...")
        background_df = pd.read_csv(cfg.SHAP_BACKGROUND_PATH).head(cfg.SHAP_BACKGROUND_LIMIT)
        explain_df = pd.read_csv(cfg.SHAP_EXPLAIN_PATH).head(cfg.SHAP_EXPLAIN_LIMIT)

        background_df = add_engineered_features(background_df, cfg)
        explain_df = add_engineered_features(explain_df, cfg)

        bg_xn, bg_xc, bg_y = preprocess_apply(background_df, meta, cfg)
        ex_xn, ex_xc, ex_y = preprocess_apply(explain_df, meta, cfg)

        background_combined = np.concatenate([bg_xn, bg_xc.astype(np.float32)], axis=1)
        explain_combined = np.concatenate([ex_xn, ex_xc.astype(np.float32)], axis=1)

        feature_names = meta["num_cols_augmented"] + meta["cat_cols"]

        wrapped_model = WrappedModel(
            base_model=global_model.to("cpu"),
            n_num=len(meta["num_cols_augmented"]),
            n_cat=len(meta["cat_cols"])
        ).to("cpu")
        wrapped_model.eval()

        background_np = background_combined.astype(np.float32)
        explain_np = explain_combined.astype(np.float32)

        def predict_fn(x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                out = wrapped_model(x_tensor).detach().cpu().numpy()
            return out

        shap_values = None
        expected_value = None

        try:
            bg_tensor = torch.tensor(background_np, dtype=torch.float32)
            ex_tensor = torch.tensor(explain_np, dtype=torch.float32)
            grad_explainer = shap.GradientExplainer(wrapped_model, bg_tensor)
            shap_values = grad_explainer.shap_values(ex_tensor)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 0]
            expected_value = float(np.mean(predict_fn(background_np)))
            LOGGER.info("Used GradientExplainer for SHAP.")
        except Exception as e:
            LOGGER.warning(f"GradientExplainer failed, falling back to KernelExplainer: {str(e)}")
            kernel_explainer = shap.KernelExplainer(predict_fn, background_np)
            shap_values = kernel_explainer.shap_values(explain_np, nsamples=cfg.SHAP_NSAMPLES)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 0]
            try:
                expected_value = kernel_explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = np.array(expected_value).reshape(-1)[0]
                expected_value = float(expected_value)
            except Exception:
                expected_value = float(np.mean(predict_fn(background_np)))

        pred_probs = predict_fn(explain_np).reshape(-1)

        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "shap_values.csv"), index=False)

        importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": importance
        }).sort_values("mean_abs_shap", ascending=False)
        importance_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "shap_feature_importance.csv"), index=False)

        save_class_specific_shap_outputs(shap_values, ex_y, feature_names, cfg.OUTPUT_DIR)

        pred_df = pd.DataFrame({
            "sample_index": np.arange(len(pred_probs)),
            "true_label": ex_y,
            "predicted_probability": pred_probs,
            "predicted_label_0.5": (pred_probs >= 0.5).astype(int)
        })
        pred_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "shap_predicted_probabilities.csv"), index=False)

        top_k = min(cfg.SHAP_TOPK_FEATURES, len(feature_names))

        sample_top_rows = []
        for i in range(len(explain_np)):
            row_vals = shap_values[i]
            top_idx = np.argsort(np.abs(row_vals))[::-1][:top_k]
            sample_top_rows.append({
                "sample_index": int(i),
                "true_label": int(ex_y[i]),
                "pred_prob": float(pred_probs[i]),
                "top_features": [feature_names[j] for j in top_idx],
                "top_shap_values": [float(row_vals[j]) for j in top_idx]
            })

        save_json(os.path.join(cfg.OUTPUT_DIR, "shap_sample_top_features.json"), sample_top_rows)
        save_json(os.path.join(cfg.OUTPUT_DIR, "shap_expected_value.json"), {"expected_value": expected_value})

        explanation = shap.Explanation(
            values=shap_values,
            base_values=np.full(shape=(explain_np.shape[0],), fill_value=expected_value, dtype=np.float32),
            data=explain_np,
            feature_names=feature_names
        )

        plt.figure()
        shap.summary_plot(shap_values, explain_np, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, "shap_summary_beeswarm.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, explain_np, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, "shap_bar.png"), dpi=300, bbox_inches="tight")
        plt.close()

        try:
            plt.figure()
            shap.plots.violin(explanation, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.OUTPUT_DIR, "shap_violin.png"), dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            LOGGER.warning(f"SHAP violin failed: {str(e)}")

        try:
            plt.figure()
            shap.plots.heatmap(explanation, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.OUTPUT_DIR, "shap_heatmap.png"), dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            LOGGER.warning(f"SHAP heatmap failed: {str(e)}")

        try:
            plt.figure()
            shap.decision_plot(
                base_value=expected_value,
                shap_values=shap_values,
                features=explain_np,
                feature_names=feature_names,
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.OUTPUT_DIR, "shap_decision.png"), dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            LOGGER.warning(f"SHAP decision plot failed: {str(e)}")

        try:
            fraud_mask = ex_y == 1
            nonfraud_mask = ex_y == 0

            if np.sum(fraud_mask) > 0:
                fraud_idx_local = int(np.argmax(pred_probs[fraud_mask]))
                fraud_idx = np.where(fraud_mask)[0][fraud_idx_local]
                plt.figure()
                shap.plots.waterfall(explanation[fraud_idx], show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(cfg.OUTPUT_DIR, "shap_waterfall_fraud_case.png"), dpi=300, bbox_inches="tight")
                plt.close()

            if np.sum(nonfraud_mask) > 0:
                nonfraud_idx_local = int(np.argmin(pred_probs[nonfraud_mask]))
                nonfraud_idx = np.where(nonfraud_mask)[0][nonfraud_idx_local]
                plt.figure()
                shap.plots.waterfall(explanation[nonfraud_idx], show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(cfg.OUTPUT_DIR, "shap_waterfall_nonfraud_case.png"), dpi=300, bbox_inches="tight")
                plt.close()
        except Exception as e:
            LOGGER.warning(f"SHAP waterfall failed: {str(e)}")

        LOGGER.info("Improved SHAP analysis completed.")
    except Exception as e:
        LOGGER.warning(f"SHAP analysis failed: {str(e)}")


def run_tee_independent_shap_analysis(global_model, meta, cfg: Config):
    if not cfg.ENABLE_SHAP or not cfg.ENABLE_TEE_INDEPENDENT_SHAP:
        LOGGER.info("TEE-independent SHAP skipped.")
        return

    try:
        LOGGER.info("Starting TEE-independent SHAP analysis...")

        background_df = pd.read_csv(cfg.SHAP_BACKGROUND_PATH).head(cfg.SHAP_BACKGROUND_LIMIT)
        explain_df = pd.read_csv(cfg.SHAP_EXPLAIN_PATH).head(cfg.SHAP_EXPLAIN_LIMIT)

        background_df = add_engineered_features(background_df, cfg)
        explain_df = add_engineered_features(explain_df, cfg)

        bg_xn, bg_xc, bg_y = preprocess_apply(background_df, meta, cfg)
        ex_xn, ex_xc, ex_y = preprocess_apply(explain_df, meta, cfg)

        feature_names = meta["num_cols_augmented"] + meta["cat_cols"]

        background_combined = np.concatenate([bg_xn, bg_xc.astype(np.float32)], axis=1).astype(np.float32)
        explain_combined = np.concatenate([ex_xn, ex_xc.astype(np.float32)], axis=1).astype(np.float32)

        model_cpu = global_model.to("cpu")
        model_cpu.eval()

        # ======================================================
        # PART A: raw input -> secure part
        # ======================================================
        secure_wrapper = SecurePartWrapper(
            base_model=model_cpu,
            n_num=len(meta["num_cols_augmented"]),
            n_cat=len(meta["cat_cols"]),
            mode=cfg.SECURE_SHAP_MODE,
            dim_idx=cfg.SECURE_SHAP_DIM_IDX
        ).to("cpu")
        secure_wrapper.eval()

        def secure_predict_fn(x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                out = secure_wrapper(x_tensor).detach().cpu().numpy()
            return out

        try:
            bg_tensor = torch.tensor(background_combined, dtype=torch.float32)
            ex_tensor = torch.tensor(explain_combined, dtype=torch.float32)
            secure_explainer = shap.GradientExplainer(secure_wrapper, bg_tensor)
            secure_shap_values = secure_explainer.shap_values(ex_tensor)
            if isinstance(secure_shap_values, list):
                secure_shap_values = secure_shap_values[0]
            secure_shap_values = np.array(secure_shap_values)
            if secure_shap_values.ndim == 3:
                secure_shap_values = secure_shap_values[:, :, 0]
            secure_expected_value = float(np.mean(secure_predict_fn(background_combined)))
            LOGGER.info("Used GradientExplainer for secure-part SHAP.")
        except Exception as e:
            LOGGER.warning(f"Secure GradientExplainer failed, fallback to KernelExplainer: {str(e)}")
            secure_explainer = shap.KernelExplainer(secure_predict_fn, background_combined)
            secure_shap_values = secure_explainer.shap_values(explain_combined, nsamples=cfg.SHAP_NSAMPLES)
            if isinstance(secure_shap_values, list):
                secure_shap_values = secure_shap_values[0]
            secure_shap_values = np.array(secure_shap_values)
            if secure_shap_values.ndim == 3:
                secure_shap_values = secure_shap_values[:, :, 0]
            secure_expected_value = float(np.mean(secure_predict_fn(background_combined)))

        secure_importance = np.abs(secure_shap_values).mean(axis=0)
        secure_imp_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": secure_importance
        }).sort_values("mean_abs_shap", ascending=False)
        secure_imp_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "shap_secure_feature_importance.csv"), index=False)

        pd.DataFrame(secure_shap_values, columns=feature_names).to_csv(
            os.path.join(cfg.OUTPUT_DIR, "shap_secure_values.csv"), index=False
        )

        plt.figure()
        shap.summary_plot(secure_shap_values, explain_combined, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, "shap_secure_summary.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(secure_shap_values, explain_combined, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, "shap_secure_bar.png"), dpi=300, bbox_inches="tight")
        plt.close()

        save_json(
            os.path.join(cfg.OUTPUT_DIR, "shap_secure_expected_value.json"),
            {"expected_value": secure_expected_value, "mode": cfg.SECURE_SHAP_MODE, "dim_idx": cfg.SECURE_SHAP_DIM_IDX}
        )

        # ======================================================
        # PART B: secure latent -> public part
        # ======================================================
        bg_cls = extract_secure_cls_embeddings(
            model_cpu, bg_xn, bg_xc,
            device=torch.device("cpu"),
            batch_size=cfg.BATCH_SIZE
        )
        ex_cls = extract_secure_cls_embeddings(
            model_cpu, ex_xn, ex_xc,
            device=torch.device("cpu"),
            batch_size=cfg.BATCH_SIZE
        )

        latent_feature_names = [f"secure_cls_dim_{i}" for i in range(bg_cls.shape[1])]

        public_wrapper = PublicPartWrapper(model_cpu).to("cpu")
        public_wrapper.eval()

        def public_predict_fn(x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                out = public_wrapper(x_tensor).detach().cpu().numpy()
            return out

        try:
            bg_tensor_latent = torch.tensor(bg_cls, dtype=torch.float32)
            ex_tensor_latent = torch.tensor(ex_cls, dtype=torch.float32)
            public_explainer = shap.GradientExplainer(public_wrapper, bg_tensor_latent)
            public_shap_values = public_explainer.shap_values(ex_tensor_latent)
            if isinstance(public_shap_values, list):
                public_shap_values = public_shap_values[0]
            public_shap_values = np.array(public_shap_values)
            if public_shap_values.ndim == 3:
                public_shap_values = public_shap_values[:, :, 0]
            public_expected_value = float(np.mean(public_predict_fn(bg_cls)))
            LOGGER.info("Used GradientExplainer for public-part SHAP.")
        except Exception as e:
            LOGGER.warning(f"Public GradientExplainer failed, fallback to KernelExplainer: {str(e)}")
            public_explainer = shap.KernelExplainer(public_predict_fn, bg_cls)
            public_shap_values = public_explainer.shap_values(ex_cls, nsamples=cfg.SHAP_NSAMPLES)
            if isinstance(public_shap_values, list):
                public_shap_values = public_shap_values[0]
            public_shap_values = np.array(public_shap_values)
            if public_shap_values.ndim == 3:
                public_shap_values = public_shap_values[:, :, 0]
            public_expected_value = float(np.mean(public_predict_fn(bg_cls)))

        public_importance = np.abs(public_shap_values).mean(axis=0)
        public_imp_df = pd.DataFrame({
            "feature": latent_feature_names,
            "mean_abs_shap": public_importance
        }).sort_values("mean_abs_shap", ascending=False)
        public_imp_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "shap_public_latent_importance.csv"), index=False)

        pd.DataFrame(public_shap_values, columns=latent_feature_names).to_csv(
            os.path.join(cfg.OUTPUT_DIR, "shap_public_latent_values.csv"), index=False
        )

        pd.DataFrame(ex_cls, columns=latent_feature_names).to_csv(
            os.path.join(cfg.OUTPUT_DIR, "secure_cls_embeddings_explain.csv"), index=False
        )

        plt.figure()
        shap.summary_plot(public_shap_values, ex_cls, feature_names=latent_feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, "shap_public_summary.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(public_shap_values, ex_cls, feature_names=latent_feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, "shap_public_bar.png"), dpi=300, bbox_inches="tight")
        plt.close()

        save_json(
            os.path.join(cfg.OUTPUT_DIR, "shap_public_expected_value.json"),
            {"expected_value": public_expected_value}
        )

        LOGGER.info("TEE-independent SHAP analysis completed.")

    except Exception as e:
        LOGGER.warning(f"TEE-independent SHAP analysis failed: {str(e)}")


def run_permutation_importance(global_model, meta, test_df, cfg: Config):
    if not cfg.ENABLE_PERMUTATION_IMPORTANCE:
        return

    try:
        LOGGER.info("Starting permutation importance...")
        test_df = add_engineered_features(test_df, cfg)
        x_num, x_cat, y = preprocess_apply(test_df, meta, cfg)
        X = np.concatenate([x_num, x_cat.astype(np.float32)], axis=1)
        feature_names = meta["num_cols_augmented"] + meta["cat_cols"]

        wrapped_model = WrappedModel(
            base_model=global_model.to("cpu"),
            n_num=len(meta["num_cols_augmented"]),
            n_cat=len(meta["cat_cols"])
        ).to("cpu")
        wrapped_model.eval()

        class SkModel(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                return self

            def predict_proba(self, X_):
                with torch.no_grad():
                    t = torch.tensor(X_, dtype=torch.float32)
                    p = wrapped_model(t).detach().cpu().numpy().reshape(-1)
                return np.vstack([1 - p, p]).T

        estimator = SkModel()

        result = permutation_importance(
            estimator,
            X,
            y,
            scoring="average_precision",
            n_repeats=5,
            random_state=cfg.SEED
        )

        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        }).sort_values("importance_mean", ascending=False)

        imp_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "permutation_importance.csv"), index=False)
        LOGGER.info("Permutation importance completed.")
    except Exception as e:
        LOGGER.warning(f"Permutation importance failed: {str(e)}")


# =========================================================
# REPORTING
# =========================================================
def print_hybrid_info_once(cfg: Config):
    LOGGER.info("================ SECURITY ARCHITECTURE ================")
    LOGGER.info("TEE-inspired partition is ENABLED")
    LOGGER.info("loading......")
    LOGGER.info("Secure part  : tokenizers + embeddings + cls token + first transformer block")
    LOGGER.info("Public part  : remaining transformer block(s) + classifier head")
    LOGGER.info("Stages       : TEE1 preprocessing | TEE2 training | TEE3 packaging")
    LOGGER.info(f"Loss         : {cfg.LOSS_TYPE}")
    LOGGER.info(f"DP updates   : {'ENABLED' if cfg.ENABLE_DP else 'DISABLED'}")
    LOGGER.info(f"Signing      : {'ENABLED' if cfg.ENABLE_SIGNING else 'DISABLED'}")
    LOGGER.info(f"Encryption   : {'AES-GCM ENABLED' if cfg.ENABLE_ENCRYPTION else 'DISABLED'}")
    LOGGER.info(f"Trusted only : {'YES' if cfg.ENABLE_TRUSTED_UPDATES_ONLY else 'NO'}")
    LOGGER.info(f"Aggregation  : {cfg.AGGREGATION_METHOD}")
    LOGGER.info(f"FedAvgM      : {'ENABLED' if cfg.SERVER_USE_MOMENTUM else 'DISABLED'}")
    LOGGER.info(f"Selection    : {cfg.MODEL_SELECTION_METRIC}")
    LOGGER.info("======================================================")


# =========================================================
# SINGLE RUN
# =========================================================
def single_run(run_id: int, cfg: Config):
    seed_everything(cfg.SEED + run_id)
    print_hybrid_info_once(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    df = pd.read_csv(cfg.DATA_PATH)
    validate_schema(df, cfg)
    df[cfg.TARGET_COL] = pd.to_numeric(df[cfg.TARGET_COL], errors="coerce").fillna(0).astype(int)

    df = add_engineered_features(df, cfg)

    num_cols, cat_cols = detect_columns(df, cfg)

    train_full_df, test_df = train_test_split(
        df, test_size=0.15, random_state=cfg.SEED + run_id, stratify=df[cfg.TARGET_COL]
    )
    train_df, val_df = train_test_split(
        train_full_df, test_size=0.17647, random_state=cfg.SEED + run_id, stratify=train_full_df[cfg.TARGET_COL]
    )

    train_xn, train_xc, train_y, meta = preprocess_train_fit(train_df, num_cols, cat_cols, cfg)
    val_xn, val_xc, val_y = preprocess_apply(val_df, meta, cfg)
    test_xn, test_xc, test_y = preprocess_apply(test_df, meta, cfg)

    save_json(PREPROCESS_META_JSON, meta)

    val_loader = DataLoader(FraudDataset(val_xn, val_xc, val_y), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(FraudDataset(test_xn, test_xc, test_y), batch_size=cfg.BATCH_SIZE, shuffle=False)

    cat_sizes = meta["cat_sizes"]
    n_num = len(meta["num_cols_augmented"])
    global_model = HybridFraudTransformer(n_num=n_num, cat_sizes=cat_sizes, cfg=cfg).to(device)

    pos = float(np.mean(train_y))
    pos_weight = torch.tensor([(1.0 - pos) / max(pos, 1e-8)], device=device)
    criterion = build_criterion(cfg, pos_weight=pos_weight)

    client_splits = create_noniid_client_splits(
        y=train_y,
        num_clients=cfg.NUM_CLIENTS,
        seed=cfg.SEED + run_id,
        dominance=cfg.NONIID_LABEL_DOMINANCE
    )

    for cid, cidx in enumerate(client_splits):
        LOGGER.info(f"Client {cid} | samples={len(cidx)} | fraud_rate={np.mean(train_y[cidx]):.6f}")

    security_round_rows = []
    round_metric_rows = []
    best_metric_value = -1.0
    best_threshold = 0.5
    best_round = 0
    patience_counter = 0
    total_rejected_updates = 0
    total_accepted_updates = 0
    seen_package_ids = set()
    best_checkpoint_metadata = None

    rng = np.random.default_rng(cfg.SEED + run_id)
    accountant = SimpleRDPAccountant(
        noise_multiplier=max(cfg.DP_NOISE_MULTIPLIER_SECURE, cfg.DP_NOISE_MULTIPLIER_PUBLIC) if cfg.ENABLE_DP else 0.0,
        sample_rate=cfg.CLIENT_FRACTION,
        delta=cfg.DP_DELTA
    )

    if os.path.exists(CLIENT_METRICS_CSV):
        os.remove(CLIENT_METRICS_CSV)

    server_velocity_state = None

    for rnd in range(1, cfg.ROUNDS + 1):
        LOGGER.info(f"================ FL ROUND {rnd}/{cfg.ROUNDS} ================")

        selected_clients = sample_clients(cfg.NUM_CLIENTS, cfg.CLIENT_FRACTION, rng)
        LOGGER.info(f"Selected clients this round: {selected_clients}")

        client_packages = []
        client_sizes = []
        global_state_cpu = tensor_state_clone_cpu(global_model.state_dict())

        for cid in selected_clients:
            client_id = f"client_{cid}"
            LOGGER.info(f"--- {client_id}: start protected pipeline ---")

            cidx = client_splits[cid]

            c_xn, c_xc, c_y, tee1_info = tee1_preprocess_client_data(train_xn, train_xc, train_y, cidx)

            local_state, tee2_info = tee2_train_client(
                global_model=global_model,
                c_xn=c_xn,
                c_xc=c_xc,
                c_y=c_y,
                n_num=n_num,
                cat_sizes=cat_sizes,
                criterion=criterion,
                device=device,
                cfg=cfg
            )

            package, tee3_info = tee3_protect_and_package_update(
                local_state=local_state,
                global_state=global_state_cpu,
                client_id=client_id,
                round_id=rnd,
                cfg=cfg
            )

            client_packages.append(package)
            client_sizes.append(len(cidx))

            secure_dp = package["dp_info"].get("secure", {})
            public_dp = package["dp_info"].get("public", {})

            LOGGER.info(f"{client_id} | TEE1 samples            : {tee1_info['tee1_samples']}")
            LOGGER.info(f"{client_id} | TEE2 loss               : {tee2_info['tee2_mean_loss']:.6f}")
            LOGGER.info(f"{client_id} | Raw update norm         : {tee3_info['raw_update_norm']:.6f}")
            LOGGER.info(f"{client_id} | Secure pre-clip norm    : {secure_dp.get('pre_clip_norm', 0.0):.6f}")
            LOGGER.info(f"{client_id} | Public pre-clip norm    : {public_dp.get('pre_clip_norm', 0.0):.6f}")
            LOGGER.info(f"{client_id} | Protected update norm   : {tee3_info['protected_update_norm']:.6f}")
            LOGGER.info(f"{client_id} | Payload size bytes      : {tee3_info['payload_size_bytes']}")

            security_round_rows.append({
                "round": rnd,
                "client_id": client_id,
                "selected": True,
                "tee1_samples": tee1_info["tee1_samples"],
                "tee1_time_sec": tee1_info["tee1_time_sec"],
                "tee2_time_sec": tee2_info["tee2_time_sec"],
                "tee2_mean_loss": tee2_info["tee2_mean_loss"],
                "hybrid_enabled": tee2_info["hybrid_enabled"],
                "raw_update_norm": tee3_info["raw_update_norm"],
                "secure_pre_clip_norm": secure_dp.get("pre_clip_norm", np.nan),
                "secure_clip_scale": secure_dp.get("clip_scale", np.nan),
                "secure_post_clip_norm": secure_dp.get("post_clip_norm", np.nan),
                "secure_noise_norm": secure_dp.get("noise_norm", np.nan),
                "public_pre_clip_norm": public_dp.get("pre_clip_norm", np.nan),
                "public_clip_scale": public_dp.get("clip_scale", np.nan),
                "public_post_clip_norm": public_dp.get("post_clip_norm", np.nan),
                "public_noise_norm": public_dp.get("noise_norm", np.nan),
                "protected_update_norm": tee3_info["protected_update_norm"],
                "signed_update": tee3_info["signed_update"],
                "transport_encrypted": tee3_info["transport_encrypted"],
                "tee3_time_sec": tee3_info["tee3_time_sec"],
                "payload_size_bytes": tee3_info["payload_size_bytes"],
            })

        LOGGER.info(f"--- SERVER VERIFICATION ROUND {rnd} ---")
        candidate_states = []
        candidate_sizes = []
        candidate_pkgs = []
        candidate_delta_vecs = []
        candidate_norms = []

        for pkg, sz in zip(client_packages, client_sizes):
            restored_state, verify_info = unpack_and_verify_client_update(pkg, seen_package_ids, cfg)

            LOGGER.info(
                f"{pkg['client_id']} | hash_ok={verify_info['hash_ok']} | "
                f"signature_ok={verify_info['signature_ok']} | replay_ok={verify_info['replay_ok']} | "
                f"trusted_update={verify_info['trusted_update']}"
            )

            for row in reversed(security_round_rows):
                if row["round"] == rnd and row["client_id"] == pkg["client_id"]:
                    row["hash_ok"] = verify_info["hash_ok"]
                    row["signature_ok"] = verify_info["signature_ok"]
                    row["replay_ok"] = verify_info["replay_ok"]
                    row["trusted_update"] = verify_info["trusted_update"]
                    break

            if verify_info["trusted_update"] or not cfg.ENABLE_TRUSTED_UPDATES_ONLY:
                delta_state = state_dict_difference(restored_state, global_state_cpu)
                candidate_states.append(restored_state)
                candidate_sizes.append(sz)
                candidate_pkgs.append(pkg)
                candidate_delta_vecs.append(flatten_state_dict(delta_state))
                candidate_norms.append(compute_state_l2_norm(delta_state))
                total_accepted_updates += 1
            else:
                total_rejected_updates += 1

        verified_states = []
        verified_sizes = []

        if len(candidate_states) > 0:
            if cfg.ENABLE_UPDATE_ANOMALY_FILTER:
                robust_z = robust_mad_zscores(candidate_norms)
                centroid = torch.stack(candidate_delta_vecs, dim=0).mean(dim=0)

                for st, sz, pkg, vec, rz in zip(candidate_states, candidate_sizes, candidate_pkgs, candidate_delta_vecs, robust_z):
                    cos_to_centroid = cosine_similarity_torch(vec, centroid)
                    anomaly_ok = True

                    if rz > cfg.MAX_ROBUST_ZSCORE:
                        anomaly_ok = False
                    if cos_to_centroid < cfg.MIN_COSINE_TO_CENTROID:
                        anomaly_ok = False

                    for row in reversed(security_round_rows):
                        if row["round"] == rnd and row["client_id"] == pkg["client_id"]:
                            row["update_norm_robust_zscore"] = float(rz)
                            row["cosine_to_centroid"] = float(cos_to_centroid)
                            row["anomaly_ok"] = bool(anomaly_ok)
                            break

                    if anomaly_ok:
                        verified_states.append(st)
                        verified_sizes.append(sz)
                    else:
                        total_rejected_updates += 1
                        total_accepted_updates -= 1
                        LOGGER.warning(
                            f"Rejected anomalous update from {pkg['client_id']} "
                            f"(robust_z={rz:.4f}, cosine={cos_to_centroid:.4f})"
                        )
            else:
                verified_states = candidate_states
                verified_sizes = candidate_sizes
                for pkg in candidate_pkgs:
                    for row in reversed(security_round_rows):
                        if row["round"] == rnd and row["client_id"] == pkg["client_id"]:
                            row["update_norm_robust_zscore"] = np.nan
                            row["cosine_to_centroid"] = np.nan
                            row["anomaly_ok"] = True
                            break

        if len(verified_states) == 0:
            LOGGER.warning("No verified client updates available. Stopping training.")
            break

        aggregated_state = robust_aggregate(verified_states, verified_sizes, cfg)

        if cfg.SERVER_USE_MOMENTUM:
            new_state, server_velocity_state = apply_server_momentum(
                old_state=global_state_cpu,
                aggregated_state=aggregated_state,
                velocity_state=server_velocity_state,
                momentum=cfg.SERVER_MOMENTUM
            )
        else:
            new_state = aggregated_state

        global_model.load_state_dict(new_state)

        if cfg.ENABLE_DP:
            accountant.step()
        eps_rdp = accountant.get_epsilon()

        val_y_true, val_probs = predict_probs(global_model, val_loader, device)
        best_th_info = find_best_threshold(val_y_true, val_probs)
        best_threshold = best_th_info["threshold"]

        val_metrics = compute_metrics(val_y_true, val_probs, best_threshold)
        test_metrics, _, _ = evaluate_model_full(global_model, test_loader, device, best_threshold)

        client_rows = evaluate_per_client(
            global_model=global_model,
            train_xn=train_xn,
            train_xc=train_xc,
            train_y=train_y,
            client_splits=client_splits,
            threshold=best_threshold,
            device=device,
            cfg=cfg
        )

        client_df = pd.DataFrame(client_rows)
        client_df["round"] = rnd

        worst_client_auc = float(client_df["roc_auc"].min())
        avg_client_auc = float(client_df["roc_auc"].mean())

        round_row = {
            "round": rnd,
            "selected_clients": len(selected_clients),
            "trusted_updates_after_filter": len(verified_states),
            "threshold": best_threshold,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "avg_client_auc": avg_client_auc,
            "worst_client_auc": worst_client_auc,
            "epsilon_rdp": eps_rdp,
            "comm_payload_bytes_total": int(sum(
                row["payload_size_bytes"] for row in security_round_rows if row["round"] == rnd
            ))
        }
        round_metric_rows.append(round_row)

        LOGGER.info(
            f"Round {rnd} | Val AUC={val_metrics['roc_auc']:.4f} | Val PR-AUC={val_metrics['pr_auc']:.4f} | "
            f"Val F1={val_metrics['f1']:.4f} | Test AUC={test_metrics['roc_auc']:.4f} | "
            f"Threshold={best_threshold:.2f} | eps_rdp≈{eps_rdp:.4f}"
        )

        if os.path.exists(CLIENT_METRICS_CSV):
            client_df.to_csv(CLIENT_METRICS_CSV, mode="a", header=False, index=False)
        else:
            client_df.to_csv(CLIENT_METRICS_CSV, index=False)

        selection_value = val_metrics["pr_auc"] if cfg.MODEL_SELECTION_METRIC == "pr_auc" else val_metrics["roc_auc"]

        if selection_value > best_metric_value:
            best_metric_value = selection_value
            best_round = rnd
            patience_counter = 0

            torch.save(global_model.state_dict(), BEST_MODEL_PATH)

            best_checkpoint_metadata = {
                "best_round": rnd,
                "best_threshold": float(best_threshold),
                "selection_metric_name": cfg.MODEL_SELECTION_METRIC,
                "selection_metric_value": float(selection_value),
                "val_metrics": val_metrics,
                "test_metrics_at_save_time": test_metrics,
                "model_path": BEST_MODEL_PATH,
            }

            checkpoint_bundle = {
                "round": rnd,
                "threshold": float(best_threshold),
                "selection_metric_name": cfg.MODEL_SELECTION_METRIC,
                "selection_metric_value": float(selection_value),
                "model_state_dict": global_model.state_dict(),
                "metadata": make_json_safe(best_checkpoint_metadata)
            }
            torch.save(checkpoint_bundle, BEST_CHECKPOINT_PATH)
            LOGGER.info(f"Saved new best checkpoint to {BEST_CHECKPOINT_PATH}")
        else:
            patience_counter += 1

        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            LOGGER.info("Early stopping triggered.")
            break

    torch.save(global_model.state_dict(), FINAL_MODEL_PATH)

    pd.DataFrame(security_round_rows).to_csv(SECURITY_RESULTS_CSV, index=False)
    pd.DataFrame(round_metric_rows).to_csv(ROUND_METRICS_CSV, index=False)

    final_threshold = best_threshold

    if os.path.exists(BEST_CHECKPOINT_PATH):
        best_checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device, weights_only=False)
        global_model.load_state_dict(best_checkpoint["model_state_dict"])
        final_threshold = float(best_checkpoint["threshold"])
    elif os.path.exists(BEST_MODEL_PATH):
        global_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    final_test_metrics, _, _ = evaluate_model_full(global_model, test_loader, device, final_threshold)

    final_summary = {
        "config": make_json_safe(asdict(cfg)),
        "tee_inspired_partition_enabled": cfg.ENABLE_TEE_INSPIRED_PARTITION,
        "tee_note": "Software TEE-inspired partition only. No real hardware enclave is implemented in this code.",
        "dp_enabled": cfg.ENABLE_DP,
        "signing_enabled": cfg.ENABLE_SIGNING,
        "transport_encryption_enabled": cfg.ENABLE_ENCRYPTION,
        "trusted_updates_only": cfg.ENABLE_TRUSTED_UPDATES_ONLY,
        "aggregation_method": cfg.AGGREGATION_METHOD,
        "server_use_momentum": cfg.SERVER_USE_MOMENTUM,
        "server_momentum": cfg.SERVER_MOMENTUM,
        "rounds_completed": len(round_metric_rows),
        "best_selection_metric_name": cfg.MODEL_SELECTION_METRIC,
        "best_selection_metric_value": float(best_metric_value) if len(round_metric_rows) > 0 else None,
        "best_round": int(best_round),
        "best_threshold": float(final_threshold),
        "final_test_metrics": make_json_safe(final_test_metrics),
        "total_accepted_updates": int(total_accepted_updates),
        "total_rejected_updates": int(total_rejected_updates),
        "dp_clip_norm_secure": cfg.DP_CLIP_NORM_SECURE,
        "dp_clip_norm_public": cfg.DP_CLIP_NORM_PUBLIC,
        "dp_noise_multiplier_secure": cfg.DP_NOISE_MULTIPLIER_SECURE,
        "dp_noise_multiplier_public": cfg.DP_NOISE_MULTIPLIER_PUBLIC,
        "delta": cfg.DP_DELTA,
        "best_model_path": BEST_MODEL_PATH,
        "final_model_path": FINAL_MODEL_PATH,
        "best_checkpoint_path": BEST_CHECKPOINT_PATH,
        "best_checkpoint_metadata": best_checkpoint_metadata
    }

    save_json(SECURITY_SUMMARY_JSON, final_summary)

    LOGGER.info("================ FINAL SUMMARY ================")
    LOGGER.info(json.dumps(make_json_safe(final_summary), indent=2))

    run_shap_analysis(global_model, meta, cfg)
    run_tee_independent_shap_analysis(global_model, meta, cfg)
    run_permutation_importance(global_model, meta, test_df, cfg)

    return {
        "run_id": run_id,
        "best_metric_name": cfg.MODEL_SELECTION_METRIC,
        "best_metric_value": float(best_metric_value) if len(round_metric_rows) > 0 else None,
        "final_test_roc_auc": final_test_metrics["roc_auc"],
        "final_test_pr_auc": final_test_metrics["pr_auc"],
        "final_test_f1": final_test_metrics["f1"],
        "rounds_completed": len(round_metric_rows),
        "aggregation_method": cfg.AGGREGATION_METHOD,
        "dp_enabled": cfg.ENABLE_DP,
        "encryption_enabled": cfg.ENABLE_ENCRYPTION,
        "anomaly_filter_enabled": cfg.ENABLE_UPDATE_ANOMALY_FILTER
    }


# =========================================================
# ABLATIONS
# =========================================================
def run_ablation_study(base_cfg: Config):
    LOGGER.info("Starting ablation study...")
    experiments = [
        ("full_system", base_cfg),
        ("no_dp", base_cfg.clone_with(ENABLE_DP=False)),
        ("no_encryption", base_cfg.clone_with(ENABLE_ENCRYPTION=False)),
        ("no_signing", base_cfg.clone_with(ENABLE_SIGNING=False)),
        ("no_anomaly_filter", base_cfg.clone_with(ENABLE_UPDATE_ANOMALY_FILTER=False)),
        ("fedavg", base_cfg.clone_with(AGGREGATION_METHOD="fedavg")),
        ("coord_median", base_cfg.clone_with(AGGREGATION_METHOD="coordinate_median")),
        ("trimmed_mean", base_cfg.clone_with(AGGREGATION_METHOD="trimmed_mean", TRIM_RATIO=0.1)),
    ]

    results = []
    for exp_name, exp_cfg in experiments:
        LOGGER.info(f"================ ABLATION: {exp_name} ================")
        exp_dir = os.path.join(base_cfg.OUTPUT_DIR, f"ablation_{exp_name}")
        os.makedirs(exp_dir, exist_ok=True)
        exp_cfg = exp_cfg.clone_with(OUTPUT_DIR=exp_dir)
        result = single_run(run_id=0, cfg=exp_cfg)
        result["experiment_name"] = exp_name
        results.append(result)

    save_json(ABLATION_RESULTS_JSON, results)
    LOGGER.info("Ablation study completed.")


# =========================================================
# MAIN
# =========================================================
def main():
    if CFG.ENABLE_ABLATION_STUDY:
        run_ablation_study(CFG)
        return

    all_runs = []
    for run_id in range(CFG.NUM_RUNS):
        LOGGER.info(f"################ RUN {run_id + 1}/{CFG.NUM_RUNS} ################")
        run_result = single_run(run_id, CFG)
        all_runs.append(run_result)

    save_json(RUNS_SUMMARY_JSON, all_runs)

    if len(all_runs) > 1:
        df = pd.DataFrame(all_runs)
        summary = {
            "best_metric_name": CFG.MODEL_SELECTION_METRIC,
            "best_metric_value_mean": float(df["best_metric_value"].mean()),
            "best_metric_value_std": float(df["best_metric_value"].std(ddof=0)),
            "final_test_roc_auc_mean": float(df["final_test_roc_auc"].mean()),
            "final_test_roc_auc_std": float(df["final_test_roc_auc"].std(ddof=0)),
            "final_test_pr_auc_mean": float(df["final_test_pr_auc"].mean()),
            "final_test_pr_auc_std": float(df["final_test_pr_auc"].std(ddof=0)),
            "final_test_f1_mean": float(df["final_test_f1"].mean()),
            "final_test_f1_std": float(df["final_test_f1"].std(ddof=0)),
        }
        save_json(os.path.join(CFG.OUTPUT_DIR, "multi_run_statistics.json"), summary)
        LOGGER.info(json.dumps(make_json_safe(summary), indent=2))




# =========================================================
# PYSIDE6 GUI
# =========================================================
import sys
import re
import traceback
import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSplitter,
)

GUI_LOG_SIGNAL = None

class QtLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if GUI_LOG_SIGNAL is not None:
                GUI_LOG_SIGNAL.emit(msg)
        except Exception:
            pass

def ensure_gui_logger_handler():
    global LOGGER
    for h in LOGGER.handlers:
        if isinstance(h, QtLogHandler):
            return
    handler = QtLogHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)

def patch_output_paths(cfg: Config):
    global RUNS_SUMMARY_JSON, PREPROCESS_META_JSON, SECURITY_RESULTS_CSV, SECURITY_SUMMARY_JSON
    global ROUND_METRICS_CSV, CLIENT_METRICS_CSV, BEST_MODEL_PATH, FINAL_MODEL_PATH
    global BEST_CHECKPOINT_PATH, LOG_PATH, ABLATION_RESULTS_JSON, CLIENT_KEYS

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    RUNS_SUMMARY_JSON = os.path.join(cfg.OUTPUT_DIR, "runs_summary.json")
    PREPROCESS_META_JSON = os.path.join(cfg.OUTPUT_DIR, "preprocess_meta.json")
    SECURITY_RESULTS_CSV = os.path.join(cfg.OUTPUT_DIR, "security_round_results.csv")
    SECURITY_SUMMARY_JSON = os.path.join(cfg.OUTPUT_DIR, "security_summary.json")
    ROUND_METRICS_CSV = os.path.join(cfg.OUTPUT_DIR, "round_metrics.csv")
    CLIENT_METRICS_CSV = os.path.join(cfg.OUTPUT_DIR, "client_metrics.csv")
    BEST_MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, "best_global_model.pt")
    FINAL_MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, "final_global_model.pt")
    BEST_CHECKPOINT_PATH = os.path.join(cfg.OUTPUT_DIR, cfg.CHECKPOINT_NAME)
    LOG_PATH = os.path.join(cfg.OUTPUT_DIR, "training.log")
    ABLATION_RESULTS_JSON = os.path.join(cfg.OUTPUT_DIR, "ablation_results.json")

    CLIENT_KEYS = {
        f"client_{i}": f"client_key_{i}_secure_demo".encode("utf-8")
        for i in range(cfg.NUM_CLIENTS)
    }

    file_handlers = [h for h in LOGGER.handlers if isinstance(h, logging.FileHandler)]
    for fh in file_handlers:
        try:
            LOGGER.removeHandler(fh)
            fh.close()
        except Exception:
            pass

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    new_fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    new_fh.setLevel(logging.INFO)
    new_fh.setFormatter(fmt)
    LOGGER.addHandler(new_fh)

class TrainingWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    @Slot()
    def run(self):
        global CFG, GUI_LOG_SIGNAL
        try:
            GUI_LOG_SIGNAL = self.log
            ensure_gui_logger_handler()
            CFG = self.cfg
            patch_output_paths(CFG)
            self.progress.emit(0, int(self.cfg.ROUNDS), "Starting...")
            self.log.emit("GUI worker started.")
            main()
            self.finished_ok.emit(self.collect_summary())
        except Exception:
            self.failed.emit(traceback.format_exc())

    def collect_summary(self):
        summary = {
            "best_round": None,
            "final_test_roc_auc": None,
            "final_test_pr_auc": None,
            "final_test_f1": None,
            "output_dir": self.cfg.OUTPUT_DIR,
        }
        try:
            if os.path.exists(SECURITY_SUMMARY_JSON):
                data = json.load(open(SECURITY_SUMMARY_JSON, "r", encoding="utf-8"))
                summary["best_round"] = data.get("best_round")
                summary["final_test_roc_auc"] = data.get("final_test_roc_auc")
                summary["final_test_pr_auc"] = data.get("final_test_pr_auc")
                summary["final_test_f1"] = data.get("final_test_f1")
            elif os.path.exists(RUNS_SUMMARY_JSON):
                data = json.load(open(RUNS_SUMMARY_JSON, "r", encoding="utf-8"))
                if isinstance(data, list) and data:
                    last = data[-1]
                    summary["best_round"] = last.get("best_round")
                    summary["final_test_roc_auc"] = last.get("final_test_roc_auc")
                    summary["final_test_pr_auc"] = last.get("final_test_pr_auc")
                    summary["final_test_f1"] = last.get("final_test_f1")
        except Exception:
            pass
        return summary

class MainWindow(QMainWindow):
    ROUND_RE = re.compile(r"Round\s+(\d+)\s+\|")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Secure Federated Fraud Detection GUI - PySide6")
        self.resize(1500, 900)
        self.thread = None
        self.worker = None
        self.last_round = 0
        self._build_ui()
        self.load_cfg_to_form(CFG)
        self._build_timer()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        toolbar = self.addToolBar("Main")
        act_open_out = QAction("Open Output Folder", self)
        act_open_out.triggered.connect(self.open_output_folder)
        toolbar.addAction(act_open_out)

        act_reload = QAction("Reload Results", self)
        act_reload.triggered.connect(self.refresh_outputs)
        toolbar.addAction(act_reload)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_config_tab(), "Configuration")
        self.tabs.addTab(self._build_run_tab(), "Run")
        self.tabs.addTab(self._build_results_tab(), "Results")
        left_layout.addWidget(self.tabs)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Live Logs"))
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        right_layout.addWidget(self.log_box)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter)

    def _build_config_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        paths_group = QGroupBox("Paths")
        paths_form = QFormLayout(paths_group)

        self.data_path = QLineEdit()
        btn_data = QPushButton("Browse")
        btn_data.clicked.connect(lambda: self.browse_file(self.data_path, "CSV Files (*.csv);;All Files (*)"))
        paths_form.addRow("Dataset path", self._line_with_button(self.data_path, btn_data))

        self.shap_bg_path = QLineEdit()
        btn_bg = QPushButton("Browse")
        btn_bg.clicked.connect(lambda: self.browse_file(self.shap_bg_path, "CSV Files (*.csv);;All Files (*)"))
        paths_form.addRow("SHAP background", self._line_with_button(self.shap_bg_path, btn_bg))

        self.shap_explain_path = QLineEdit()
        btn_explain = QPushButton("Browse")
        btn_explain.clicked.connect(lambda: self.browse_file(self.shap_explain_path, "CSV Files (*.csv);;All Files (*)"))
        paths_form.addRow("SHAP explain", self._line_with_button(self.shap_explain_path, btn_explain))

        self.output_dir = QLineEdit()
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(lambda: self.browse_dir(self.output_dir))
        paths_form.addRow("Output folder", self._line_with_button(self.output_dir, btn_out))

        train_group = QGroupBox("Training / FL")
        train_form = QFormLayout(train_group)
        self.batch_size = self._spin(1, 100000, 512)
        self.lr = self._dspin(1e-7, 1.0, 0.0002, 7)
        self.weight_decay = self._dspin(0.0, 1.0, 0.0001, 7)
        self.num_runs = self._spin(1, 100, 1)
        self.seed = self._spin(0, 999999, 42)
        self.rounds = self._spin(1, 1000, 50)
        self.local_epochs = self._spin(1, 100, 3)
        self.num_clients = self._spin(1, 1000, 10)
        self.client_fraction = self._dspin(0.01, 1.0, 1.0, 2)
        self.early_stopping = self._spin(1, 1000, 10)
        self.noniid_dom = self._dspin(0.0, 1.0, 0.60, 2)
        self.d_model = self._spin(8, 4096, 128)
        self.n_heads = self._spin(1, 64, 8)
        self.n_layers = self._spin(1, 64, 3)
        self.dropout = self._dspin(0.0, 1.0, 0.10, 2)
        self.mlp_hidden = self._spin(1, 8192, 256)

        self.loss_type = QComboBox(); self.loss_type.addItems(["focal", "bce"])
        self.agg_method = QComboBox(); self.agg_method.addItems(["fedavg", "trimmed_mean", "coordinate_median"])
        self.selection_metric = QComboBox(); self.selection_metric.addItems(["pr_auc", "roc_auc"])

        train_form.addRow("Batch size", self.batch_size)
        train_form.addRow("Learning rate", self.lr)
        train_form.addRow("Weight decay", self.weight_decay)
        train_form.addRow("Runs", self.num_runs)
        train_form.addRow("Seed", self.seed)
        train_form.addRow("Rounds", self.rounds)
        train_form.addRow("Local epochs", self.local_epochs)
        train_form.addRow("Clients", self.num_clients)
        train_form.addRow("Client fraction", self.client_fraction)
        train_form.addRow("Early stopping", self.early_stopping)
        train_form.addRow("Non-IID dominance", self.noniid_dom)
        train_form.addRow("D_MODEL", self.d_model)
        train_form.addRow("N_HEADS", self.n_heads)
        train_form.addRow("N_LAYERS", self.n_layers)
        train_form.addRow("Dropout", self.dropout)
        train_form.addRow("MLP hidden", self.mlp_hidden)
        train_form.addRow("Loss type", self.loss_type)
        train_form.addRow("Aggregation", self.agg_method)
        train_form.addRow("Selection metric", self.selection_metric)

        sec_group = QGroupBox("Security / XAI")
        sec_grid = QGridLayout(sec_group)
        self.cb_dp = QCheckBox("Enable DP")
        self.cb_sign = QCheckBox("Enable Signing")
        self.cb_enc = QCheckBox("Enable Encryption")
        self.cb_trusted = QCheckBox("Trusted Updates Only")
        self.cb_replay = QCheckBox("Replay Protection")
        self.cb_anomaly = QCheckBox("Update Anomaly Filter")
        self.cb_fedprox = QCheckBox("Enable FedProx")
        self.cb_feat = QCheckBox("Feature Engineering")
        self.cb_log1p = QCheckBox("Log1p Skewed Features")
        self.cb_bin_no_std = QCheckBox("Binary No Standardize")
        self.cb_shap = QCheckBox("Enable SHAP")
        self.cb_tee_shap = QCheckBox("TEE Independent SHAP")
        self.cb_perm = QCheckBox("Permutation Importance")
        self.cb_ablation = QCheckBox("Ablation Study")
        boxes = [
            self.cb_dp, self.cb_sign, self.cb_enc, self.cb_trusted, self.cb_replay,
            self.cb_anomaly, self.cb_fedprox, self.cb_feat, self.cb_log1p,
            self.cb_bin_no_std, self.cb_shap, self.cb_tee_shap, self.cb_perm, self.cb_ablation
        ]
        for i, box in enumerate(boxes):
            sec_grid.addWidget(box, i // 2, i % 2)

        layout.addWidget(paths_group)
        layout.addWidget(train_group)
        layout.addWidget(sec_group)
        layout.addStretch(1)
        return w

    def _build_run_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        top = QHBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_start.clicked.connect(self.start_training)
        self.btn_stop = QPushButton("Stop Request")
        self.btn_stop.clicked.connect(self.stop_requested)
        self.btn_stop.setEnabled(False)
        self.btn_open = QPushButton("Open Output Folder")
        self.btn_open.clicked.connect(self.open_output_folder)
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)
        top.addWidget(self.btn_open)
        top.addStretch(1)

        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)

        summary_group = QGroupBox("Quick Summary")
        summary_form = QFormLayout(summary_group)
        self.lbl_best_round = QLabel("-")
        self.lbl_auc = QLabel("-")
        self.lbl_pr = QLabel("-")
        self.lbl_f1 = QLabel("-")
        self.lbl_output = QLabel("-")
        summary_form.addRow("Best round", self.lbl_best_round)
        summary_form.addRow("Final ROC-AUC", self.lbl_auc)
        summary_form.addRow("Final PR-AUC", self.lbl_pr)
        summary_form.addRow("Final F1", self.lbl_f1)
        summary_form.addRow("Output folder", self.lbl_output)

        layout.addLayout(top)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addWidget(summary_group)
        layout.addStretch(1)
        return w

    def _build_results_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Round Metrics"))
        self.round_table = QTableWidget()
        self.round_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.round_table)
        return w

    def _build_timer(self):
        self.timer = QTimer(self)
        self.timer.setInterval(2500)
        self.timer.timeout.connect(self.refresh_outputs)

    def _spin(self, mn, mx, val):
        s = QSpinBox(); s.setRange(mn, mx); s.setValue(val); return s

    def _dspin(self, mn, mx, val, decimals=4):
        s = QDoubleSpinBox(); s.setDecimals(decimals); s.setRange(mn, mx); s.setValue(val); return s

    def _line_with_button(self, line, button):
        box = QWidget(); lay = QHBoxLayout(box); lay.setContentsMargins(0,0,0,0); lay.addWidget(line); lay.addWidget(button); return box

    def browse_file(self, line_edit, filt):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", line_edit.text() or "", filt)
        if path: line_edit.setText(path)

    def browse_dir(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text() or "")
        if path: line_edit.setText(path)

    def load_cfg_to_form(self, cfg: Config):
        self.data_path.setText(cfg.DATA_PATH)
        self.shap_bg_path.setText(cfg.SHAP_BACKGROUND_PATH)
        self.shap_explain_path.setText(cfg.SHAP_EXPLAIN_PATH)
        self.output_dir.setText(cfg.OUTPUT_DIR)
        self.batch_size.setValue(cfg.BATCH_SIZE)
        self.lr.setValue(cfg.LR)
        self.weight_decay.setValue(cfg.WEIGHT_DECAY)
        self.num_runs.setValue(cfg.NUM_RUNS)
        self.seed.setValue(cfg.SEED)
        self.rounds.setValue(cfg.ROUNDS)
        self.local_epochs.setValue(cfg.LOCAL_EPOCHS)
        self.num_clients.setValue(cfg.NUM_CLIENTS)
        self.client_fraction.setValue(cfg.CLIENT_FRACTION)
        self.early_stopping.setValue(cfg.EARLY_STOPPING_PATIENCE)
        self.noniid_dom.setValue(cfg.NONIID_LABEL_DOMINANCE)
        self.d_model.setValue(cfg.D_MODEL)
        self.n_heads.setValue(cfg.N_HEADS)
        self.n_layers.setValue(cfg.N_LAYERS)
        self.dropout.setValue(cfg.DROPOUT)
        self.mlp_hidden.setValue(cfg.MLP_HIDDEN)
        self.loss_type.setCurrentText(cfg.LOSS_TYPE)
        self.agg_method.setCurrentText(cfg.AGGREGATION_METHOD)
        self.selection_metric.setCurrentText(cfg.MODEL_SELECTION_METRIC)
        self.cb_dp.setChecked(cfg.ENABLE_DP)
        self.cb_sign.setChecked(cfg.ENABLE_SIGNING)
        self.cb_enc.setChecked(cfg.ENABLE_ENCRYPTION)
        self.cb_trusted.setChecked(cfg.ENABLE_TRUSTED_UPDATES_ONLY)
        self.cb_replay.setChecked(cfg.ENABLE_REPLAY_PROTECTION)
        self.cb_anomaly.setChecked(cfg.ENABLE_UPDATE_ANOMALY_FILTER)
        self.cb_fedprox.setChecked(cfg.ENABLE_FEDPROX)
        self.cb_feat.setChecked(cfg.ENABLE_FEATURE_ENGINEERING)
        self.cb_log1p.setChecked(cfg.ENABLE_LOG1P_SKEWED_FEATURES)
        self.cb_bin_no_std.setChecked(cfg.ENABLE_BINARY_NO_STANDARDIZE)
        self.cb_shap.setChecked(cfg.ENABLE_SHAP)
        self.cb_tee_shap.setChecked(cfg.ENABLE_TEE_INDEPENDENT_SHAP)
        self.cb_perm.setChecked(cfg.ENABLE_PERMUTATION_IMPORTANCE)
        self.cb_ablation.setChecked(cfg.ENABLE_ABLATION_STUDY)

    def build_cfg_from_form(self):
        return Config(
            DATA_PATH=self.data_path.text().strip(),
            TARGET_COL="Class",
            SHAP_BACKGROUND_PATH=self.shap_bg_path.text().strip(),
            SHAP_EXPLAIN_PATH=self.shap_explain_path.text().strip(),
            OUTPUT_DIR=self.output_dir.text().strip(),
            BATCH_SIZE=self.batch_size.value(),
            LR=self.lr.value(),
            WEIGHT_DECAY=self.weight_decay.value(),
            SEED=self.seed.value(),
            CLIP_NORM=1.0,
            NUM_RUNS=self.num_runs.value(),
            LR_WARMUP_RATIO=0.10,
            LOSS_TYPE=self.loss_type.currentText(),
            FOCAL_ALPHA=0.75,
            FOCAL_GAMMA=2.0,
            LABEL_SMOOTHING=0.0,
            NUM_CLIENTS=self.num_clients.value(),
            ROUNDS=self.rounds.value(),
            LOCAL_EPOCHS=self.local_epochs.value(),
            CLIENT_FRACTION=self.client_fraction.value(),
            EARLY_STOPPING_PATIENCE=self.early_stopping.value(),
            NONIID_LABEL_DOMINANCE=self.noniid_dom.value(),
            SERVER_USE_MOMENTUM=True,
            SERVER_MOMENTUM=0.9,
            D_MODEL=self.d_model.value(),
            N_HEADS=self.n_heads.value(),
            N_LAYERS=self.n_layers.value(),
            DROPOUT=self.dropout.value(),
            MLP_HIDDEN=self.mlp_hidden.value(),
            ENABLE_TEE_INSPIRED_PARTITION=True,
            ENABLE_DP=self.cb_dp.isChecked(),
            ENABLE_SIGNING=self.cb_sign.isChecked(),
            ENABLE_ENCRYPTION=self.cb_enc.isChecked(),
            ENABLE_TRUSTED_UPDATES_ONLY=self.cb_trusted.isChecked(),
            ENABLE_REPLAY_PROTECTION=self.cb_replay.isChecked(),
            ENABLE_UPDATE_ANOMALY_FILTER=self.cb_anomaly.isChecked(),
            AGGREGATION_METHOD=self.agg_method.currentText(),
            TRIM_RATIO=0.1,
            MAX_ROBUST_ZSCORE=5.0,
            MIN_COSINE_TO_CENTROID=-0.20,
            DP_CLIP_NORM_SECURE=1.2,
            DP_CLIP_NORM_PUBLIC=1.8,
            DP_NOISE_MULTIPLIER_SECURE=0.01,
            DP_NOISE_MULTIPLIER_PUBLIC=0.005,
            DP_DELTA=1e-5,
            ENABLE_FEDPROX=self.cb_fedprox.isChecked(),
            FEDPROX_MU=1e-3,
            TRANSPORT_KEY_HEX="7f4c7b2d0b5f8c13344f54c7d2e1a911be2890d2f60a2f30d5a8ee7dfe1ab8c1",
            ENABLE_FEATURE_ENGINEERING=self.cb_feat.isChecked(),
            ENABLE_LOG1P_SKEWED_FEATURES=self.cb_log1p.isChecked(),
            ENABLE_BINARY_NO_STANDARDIZE=self.cb_bin_no_std.isChecked(),
            ENABLE_SHAP=self.cb_shap.isChecked(),
            ENABLE_TEE_INDEPENDENT_SHAP=self.cb_tee_shap.isChecked(),
            ENABLE_PERMUTATION_IMPORTANCE=self.cb_perm.isChecked(),
            SHAP_NSAMPLES=100,
            SHAP_TOPK_FEATURES=5,
            SHAP_BACKGROUND_LIMIT=128,
            SHAP_EXPLAIN_LIMIT=100,
            SECURE_SHAP_MODE="cls_norm",
            SECURE_SHAP_DIM_IDX=0,
            MODEL_SELECTION_METRIC=self.selection_metric.currentText(),
            CHECKPOINT_NAME="best_checkpoint.pt",
            ENABLE_ABLATION_STUDY=self.cb_ablation.isChecked(),
        )

    def validate_before_start(self, cfg: Config):
        if not cfg.DATA_PATH or not os.path.exists(cfg.DATA_PATH):
            raise FileNotFoundError("Dataset path is missing or invalid.")
        if cfg.ENABLE_SHAP:
            if not cfg.SHAP_BACKGROUND_PATH or not os.path.exists(cfg.SHAP_BACKGROUND_PATH):
                raise FileNotFoundError("SHAP background path is missing or invalid.")
            if not cfg.SHAP_EXPLAIN_PATH or not os.path.exists(cfg.SHAP_EXPLAIN_PATH):
                raise FileNotFoundError("SHAP explain path is missing or invalid.")
        if not cfg.OUTPUT_DIR:
            raise ValueError("Output folder is required.")
        if cfg.D_MODEL % cfg.N_HEADS != 0:
            raise ValueError("D_MODEL must be divisible by N_HEADS.")

    @Slot()
    def start_training(self):
        if self.thread is not None:
            QMessageBox.warning(self, "Busy", "Training is already running.")
            return
        try:
            cfg = self.build_cfg_from_form()
            self.validate_before_start(cfg)
        except Exception as e:
            QMessageBox.critical(self, "Invalid configuration", str(e))
            return

        self.log_box.clear()
        self.last_round = 0
        self.progress.setMaximum(max(1, cfg.ROUNDS))
        self.progress.setValue(0)
        self.status_label.setText("Starting worker...")
        self.lbl_output.setText(cfg.OUTPUT_DIR)

        self.thread = QThread(self)
        self.worker = TrainingWorker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.on_log)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_finished_ok)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished_ok.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_thread)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.timer.start()
        self.thread.start()

    @Slot()
    def stop_requested(self):
        QMessageBox.information(
            self,
            "Stop request",
            "This GUI version does not force-stop PyTorch safely during a round.\nWait for the run to finish."
        )

    @Slot(str)
    def on_log(self, msg):
        self.log_box.appendPlainText(msg)
        m = self.ROUND_RE.search(msg)
        if m:
            current_round = int(m.group(1))
            self.last_round = max(self.last_round, current_round)
            self.progress.setValue(self.last_round)
            self.status_label.setText(f"Running round {self.last_round}/{self.progress.maximum()}")

    @Slot(int, int, str)
    def on_progress(self, current, total, text):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(current)
        self.status_label.setText(text)

    @Slot(dict)
    def on_finished_ok(self, summary):
        self.timer.stop()
        self.progress.setValue(self.progress.maximum())
        self.status_label.setText("Finished")
        self.lbl_best_round.setText(str(summary.get("best_round", "-")))
        self.lbl_auc.setText(str(summary.get("final_test_roc_auc", "-")))
        self.lbl_pr.setText(str(summary.get("final_test_pr_auc", "-")))
        self.lbl_f1.setText(str(summary.get("final_test_f1", "-")))
        self.lbl_output.setText(str(summary.get("output_dir", "-")))
        self.refresh_outputs()
        QMessageBox.information(self, "Done", "Training completed successfully.")

    @Slot(str)
    def on_failed(self, err):
        self.timer.stop()
        self.status_label.setText("Failed")
        self.log_box.appendPlainText("\n========== ERROR ==========\n" + err)
        QMessageBox.critical(self, "Training failed", err)

    @Slot()
    def cleanup_thread(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.worker is not None: self.worker.deleteLater()
        if self.thread is not None: self.thread.deleteLater()
        self.worker = None
        self.thread = None

    @Slot()
    def refresh_outputs(self):
        out_dir = self.output_dir.text().strip()
        if not out_dir:
            return
        round_csv = os.path.join(out_dir, "round_metrics.csv")
        if os.path.exists(round_csv):
            try:
                df = pd.read_csv(round_csv)
                self.populate_table(df)
                if len(df) > 0 and "round" in df.columns:
                    current_round = int(df["round"].max())
                    self.last_round = max(self.last_round, current_round)
                    self.progress.setValue(min(self.last_round, self.progress.maximum()))
            except Exception as e:
                self.log_box.appendPlainText(f"Failed reading round_metrics.csv: {e}")

        sec_json = os.path.join(out_dir, "security_summary.json")
        if os.path.exists(sec_json):
            try:
                data = json.load(open(sec_json, "r", encoding="utf-8"))
                self.lbl_best_round.setText(str(data.get("best_round", "-")))
                self.lbl_auc.setText(str(data.get("final_test_roc_auc", "-")))
                self.lbl_pr.setText(str(data.get("final_test_pr_auc", "-")))
                self.lbl_f1.setText(str(data.get("final_test_f1", "-")))
            except Exception as e:
                self.log_box.appendPlainText(f"Failed reading security_summary.json: {e}")

    def populate_table(self, df: pd.DataFrame):
        if df is None or df.empty:
            self.round_table.setRowCount(0); self.round_table.setColumnCount(0); return
        show = df.copy()
        for col in show.columns:
            if pd.api.types.is_float_dtype(show[col]):
                show[col] = show[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
        self.round_table.setColumnCount(len(show.columns))
        self.round_table.setRowCount(len(show))
        self.round_table.setHorizontalHeaderLabels([str(c) for c in show.columns])
        for r in range(len(show)):
            for c, col in enumerate(show.columns):
                item = QTableWidgetItem(str(show.iloc[r, c]))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.round_table.setItem(r, c, item)

    @Slot()
    def open_output_folder(self):
        out_dir = self.output_dir.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No folder", "Output folder is empty.")
            return
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(out_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", out_dir])
            else:
                subprocess.Popen(["xdg-open", out_dir])
        except Exception as e:
            QMessageBox.warning(self, "Open folder failed", str(e))

def launch_gui():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()

if __name__ == "__main__":
    if "--cli" in sys.argv:
        main()
    else:
        sys.exit(launch_gui())
