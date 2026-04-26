"""
Dataset splitting utilities.

For financial data, the preferred default is time-based splitting.
"""

import pandas as pd


def split_by_time_fraction(
    model_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    train_fraction: float = 2 / 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits model_df and metadata_df into train/test sets by date.

    The first train_fraction of dates becomes training data.
    The remaining dates become test data.

    This assumes model_df and metadata_df have matching row order.
    """
    if len(model_df) != len(metadata_df):
        raise ValueError("model_df and metadata_df must have the same number of rows.")

    required_metadata_columns = {"date", "ticker"}
    missing_columns = required_metadata_columns - set(metadata_df.columns)

    if missing_columns:
        raise ValueError(f"Missing metadata columns: {missing_columns}")

    combined_df = pd.concat(
        [
            metadata_df.reset_index(drop=True),
            model_df.reset_index(drop=True),
        ],
        axis=1,
    )

    combined_df["date"] = pd.to_datetime(combined_df["date"])
    combined_df = combined_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    unique_dates = combined_df["date"].sort_values().unique()

    if len(unique_dates) < 2:
        raise ValueError("Not enough unique dates to split dataset.")

    split_index = int(len(unique_dates) * train_fraction)

    if split_index <= 0 or split_index >= len(unique_dates):
        raise ValueError("Invalid split. Adjust train_fraction.")

    split_date = unique_dates[split_index]

    train_df = combined_df[combined_df["date"] < split_date].copy()
    test_df = combined_df[combined_df["date"] >= split_date].copy()

    metadata_columns = list(metadata_df.columns)

    train_metadata_df = train_df[metadata_columns].copy()
    test_metadata_df = test_df[metadata_columns].copy()

    train_model_df = train_df.drop(columns=metadata_columns)
    test_model_df = test_df.drop(columns=metadata_columns)

    return train_model_df, test_model_df, train_metadata_df, test_metadata_df


def split_features_and_target(
    model_df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Splits a model DataFrame into X and y.
    """
    if target_column not in model_df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    X = model_df.drop(columns=[target_column]).copy()
    y = model_df[target_column].copy()

    return X, y