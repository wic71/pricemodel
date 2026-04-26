"""
Dataset balancing utilities.

These functions balance model datasets after features and targets have been built.
"""

import pandas as pd


def print_target_distribution(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Prints target class distribution.
    """
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    print("\nTarget distribution:")
    print(df[target_column].value_counts(dropna=False).sort_index())

    print("\nTarget distribution, normalized:")
    print(df[target_column].value_counts(normalize=True, dropna=False).sort_index())


def balance_by_undersampling(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Balances a classification dataset by undersampling all classes
    to the size of the smallest class.

    This keeps the same number of rows for each target class.
    """
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    df = df.copy()

    class_counts = df[target_column].value_counts()
    min_class_count = class_counts.min()

    balanced_parts = []

    for target_value, group_df in df.groupby(target_column):
        sampled_group = group_df.sample(
            n=min_class_count,
            random_state=random_state,
        )
        balanced_parts.append(sampled_group)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)

    return balanced_df


def split_features_target_metadata(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    metadata_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None]:
    """
    Splits a DataFrame into X, y and optional metadata.
    """
    if metadata_columns is None:
        metadata_columns = []

    required_columns = set(feature_columns + [target_column] + metadata_columns)
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    metadata_df = None
    if metadata_columns:
        metadata_df = df[metadata_columns].copy()

    return X, y, metadata_df