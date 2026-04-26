"""
Experiment logging utilities.
"""

from pathlib import Path
from datetime import datetime

import pandas as pd


def log_experiment(
    model_name: str,
    target_column: str,
    model_path: str,
    train_metrics: dict,
    test_metrics: dict,
    train_rows: int,
    test_rows: int,
    feature_count: int,
    dataset_name: str = "model_dataset.parquet",
    log_path: str = "artifacts/experiments/experiment_log.csv",
) -> pd.DataFrame:
    """
    Appends experiment results to a CSV log.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "target_column": target_column,
        "model_path": model_path,
        "dataset_name": dataset_name,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "feature_count": feature_count,
    }

    for key, value in train_metrics.items():
        row[f"train_{key}"] = value

    for key, value in test_metrics.items():
        row[f"test_{key}"] = value

    new_log_df = pd.DataFrame([row])

    if Path(log_path).exists():
        existing_log_df = pd.read_csv(log_path)
        log_df = pd.concat([existing_log_df, new_log_df], ignore_index=True)
    else:
        log_df = new_log_df

    log_df.to_csv(log_path, index=False)

    return log_df