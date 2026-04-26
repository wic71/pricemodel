"""
Model persistence utilities.
"""

from pathlib import Path
from datetime import datetime

import joblib


def save_model(
    model,
    model_name: str,
    output_dir: str = "artifacts/models",
) -> str:
    """
    Saves a trained model to disk and returns the path.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path(output_dir) / f"{timestamp}_{model_name}.joblib"

    joblib.dump(model, model_path)

    return str(model_path)


def load_model(model_path: str):
    """
    Loads a saved model.
    """
    return joblib.load(model_path)