"""
Model evaluation utilities.
"""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_classifier(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
) -> dict:
    """
    Evaluates a classifier and returns metrics.
    """
    y_pred = model.predict(X)

    metrics = {
        "label": label,
        "rows": len(X),
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
    }

    return metrics


def print_evaluation_report(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
) -> None:
    """
    Prints detailed evaluation report.
    """
    y_pred = model.predict(X)

    print(f"\nEvaluation: {label}")
    print("-" * 80)

    print("Accuracy:", accuracy_score(y, y_pred))
    print("Balanced accuracy:", balanced_accuracy_score(y, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification report:")
    print(classification_report(y, y_pred, zero_division=0))


def build_prediction_frame(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    metadata_df: pd.DataFrame | None = None,
    target_column: str = "target",
) -> pd.DataFrame:
    """
    Builds a DataFrame with predictions, probabilities and optional metadata.
    """
    prediction_df = pd.DataFrame()

    if metadata_df is not None:
        prediction_df = metadata_df.reset_index(drop=True).copy()

    prediction_df[target_column] = y.reset_index(drop=True)
    prediction_df["prediction"] = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = model.classes_

        for index, class_value in enumerate(classes):
            prediction_df[f"probability_{class_value}"] = probabilities[:, index]

    return prediction_df