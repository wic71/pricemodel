"""
Feature importance utilities.

Supports:
- built-in feature_importances_ for tree-based models
- coefficient importance for linear models
- permutation importance on train/test data
"""

from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance


def get_builtin_feature_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extracts built-in feature importance from a fitted model.

    Works for models with:
    - feature_importances_
    - coef_
    """
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance_values,
            }
        )

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

        return importance_df

    if hasattr(model, "coef_"):
        coef = model.coef_

        # Binary classification often has shape (1, n_features).
        # Multiclass often has shape (n_classes, n_features).
        if coef.ndim == 2:
            importance_values = abs(coef).mean(axis=0)
        else:
            importance_values = abs(coef)

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance_values,
            }
        )

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

        return importance_df

    return pd.DataFrame(
        columns=[
            "feature",
            "importance",
        ]
    )


def get_permutation_feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "balanced_accuracy",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculates permutation importance.

    This measures how much model score worsens when each feature is shuffled.

    Use on test data to estimate out-of-sample importance.
    """
    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    return importance_df


def save_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    importance_type: str,
    output_dir: str = "artifacts/feature_importance",
) -> str:
    """
    Saves feature importance DataFrame to CSV.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_path = Path(output_dir) / f"{model_name}_{importance_type}.csv"

    importance_df.to_csv(output_path, index=False)

    return str(output_path)


def print_top_features(
    importance_df: pd.DataFrame,
    value_column: str,
    top_n: int = 20,
) -> None:
    """
    Prints top N feature importance rows.
    """
    if importance_df.empty:
        print("No feature importance available.")
        return

    print(importance_df.head(top_n)[["feature", value_column]])