# =========================
# FILE 1: core_training.py
# =========================

import pandas as pd
import time


def run_training(config):
    # Load dataset
    df = pd.read_csv(config["DATA_PATH"])

    results = []

    for r in range(1, config["ROUNDS"] + 1):
        time.sleep(0.5)  # simulate training time

        # fake metrics (replace later with real model)
        pr_auc = 0.60 + (r * 0.005)
        roc_auc = 0.90 + (r * 0.002)
        f1 = 0.65 + (r * 0.004)

        results.append({
            "round": r,
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "f1": round(f1, 4)
        })

    summary = {
        "status": "training_completed",
        "rounds": config["ROUNDS"],
        "num_clients": config["NUM_CLIENTS"],
        "final_metrics": results[-1]
    }

    return results, summary

