"""
Feature builder.

Builds selected model features from a DataFrame that already contains
price data and indicator columns.

This module should:
- create requested derived features
- optionally create targets
- select only model feature columns
- optionally keep metadata for debugging/backtesting
"""

import numpy as np
import pandas as pd

from features.normalization import (
    DEFAULT_CLIP_CONFIG,
    clean_model_features,
    validate_required_columns,
)
from features.targets import (
    add_binary_direction_target,
    add_three_class_direction_target,
)


METADATA_COLUMNS = [
    "date",
    "ticker",
]

DEBUG_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]

def build_model_dataset_from_existing_target(
    price_df: pd.DataFrame,
    features: list[str],
    target_column: str,
    include_metadata: bool = False,
    include_debug_columns: bool = False,
    clip_config: dict[str, tuple[float, float]] | None = DEFAULT_CLIP_CONFIG,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Builds final model dataset when target already exists in price_df.
    """
    validate_required_columns(price_df, {"date", "ticker", target_column})

    feature_dfs = []

    for _, group_df in price_df.groupby("ticker", sort=False):
        feature_df = build_features_for_ticker(
            price_df=group_df,
            features=features,
            include_metadata=True,
            include_debug_columns=include_debug_columns,
            dropna=False,
        )

        target_df = group_df[["date", "ticker", target_column]].copy()

        feature_df = feature_df.merge(
            target_df,
            on=["date", "ticker"],
            how="left",
        )

        feature_dfs.append(feature_df)

    model_df = pd.concat(feature_dfs, ignore_index=True)

    selected_columns = []

    if include_metadata:
        selected_columns += ["date", "ticker"]

    if include_debug_columns:
        selected_columns += [
            column for column in DEBUG_COLUMNS
            if column in model_df.columns
        ]

    selected_columns += features + [target_column]

    validate_required_columns(model_df, selected_columns)

    model_df = model_df[selected_columns]

    model_df = clean_model_features(
        model_df=model_df,
        feature_columns=features,
        target_column=target_column,
        clip_config=clip_config,
        dropna=dropna,
    )

    return model_df

def build_features(
    price_df: pd.DataFrame,
    features: list[str],
    include_metadata: bool = True,
    include_debug_columns: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Builds selected model features for all tickers.

    This function does not add targets. It only creates and selects features.
    """
    validate_required_columns(price_df, {"date", "ticker", "close"})

    feature_dfs = []

    for _, group_df in price_df.groupby("ticker", sort=False):
        feature_df = build_features_for_ticker(
            price_df=group_df,
            features=features,
            include_metadata=include_metadata,
            include_debug_columns=include_debug_columns,
            dropna=False,
        )
        feature_dfs.append(feature_df)

    all_features_df = pd.concat(feature_dfs, ignore_index=True)

    if dropna:
        all_features_df = all_features_df.dropna(subset=features)

    return all_features_df


def build_features_for_ticker(
    price_df: pd.DataFrame,
    features: list[str],
    include_metadata: bool = True,
    include_debug_columns: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Builds selected model features for one ticker.

    Assumes indicator columns already exist in price_df.
    """
    validate_required_columns(price_df, {"date", "close"})

    df = price_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df = add_requested_base_return_features(df, features)
    df = add_requested_scaled_features(df, features)
    df = add_requested_change_features(df, features)
    df = add_requested_shifted_features(df, features)

    selected_columns = []

    if include_metadata:
        selected_columns += [column for column in METADATA_COLUMNS if column in df.columns]

    if include_debug_columns:
        selected_columns += [column for column in DEBUG_COLUMNS if column in df.columns]

    selected_columns += features

    validate_required_columns(df, selected_columns)

    df = df[selected_columns]

    if dropna:
        df = df.dropna(subset=features)

    return df


def build_model_dataset(
    price_df: pd.DataFrame,
    features: list[str],
    target_horizon: int = 5,
    target_type: str = "three_class",
    threshold: float = 0.005,
    target_column: str | None = None,
    include_metadata: bool = False,
    include_debug_columns: bool = False,
    clip_config: dict[str, tuple[float, float]] | None = DEFAULT_CLIP_CONFIG,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Builds final model dataset with selected features and one target.

    By default, the returned DataFrame contains only:
    - selected feature columns
    - target column

    Set include_metadata=True to keep date and ticker for debugging/backtesting.
    """
    validate_required_columns(price_df, {"date", "ticker", "close"})

    dataset_dfs = []

    for _, group_df in price_df.groupby("ticker", sort=False):
        ticker_df = group_df.copy()
        ticker_df = ticker_df.sort_values("date").reset_index(drop=True)

        ticker_df = build_features_for_ticker(
            price_df=ticker_df,
            features=features,
            include_metadata=True,
            include_debug_columns=True,
            dropna=False,
        )

        if target_type == "binary":
            ticker_df = add_binary_direction_target(
                df=ticker_df,
                horizon=target_horizon,
                threshold=threshold,
                price_column="close",
                target_column=target_column,
            )

            if target_column is None:
                final_target_column = f"target_{target_horizon}d_binary"
            else:
                final_target_column = target_column

        elif target_type == "three_class":
            ticker_df = add_three_class_direction_target(
                df=ticker_df,
                horizon=target_horizon,
                threshold=threshold,
                price_column="close",
                target_column=target_column,
            )

            if target_column is None:
                final_target_column = f"target_{target_horizon}d_3class"
            else:
                final_target_column = target_column

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        dataset_dfs.append(ticker_df)

    model_df = pd.concat(dataset_dfs, ignore_index=True)

    selected_columns = []

    if include_metadata:
        selected_columns += [column for column in METADATA_COLUMNS if column in model_df.columns]

    if include_debug_columns:
        selected_columns += [column for column in DEBUG_COLUMNS if column in model_df.columns]

    selected_columns += features + [final_target_column]

    validate_required_columns(model_df, selected_columns)

    model_df = model_df[selected_columns]

    model_df = clean_model_features(
        model_df=model_df,
        feature_columns=features,
        target_column=final_target_column,
        clip_config=clip_config,
        dropna=dropna,
    )

    return model_df


def add_requested_base_return_features(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Adds requested return-based features.
    """
    validate_required_columns(df, {"close"})

    df = df.copy()

    if "return_1d" in features:
        df["return_1d"] = df["close"].pct_change(1)

    if "return_5d" in features:
        df["return_5d"] = df["close"].pct_change(5)

    if "return_20d" in features:
        df["return_20d"] = df["close"].pct_change(20)

    if "log_return_1d" in features:
        previous_close = df["close"].shift(1).replace(0, np.nan)
        df["log_return_1d"] = np.log(df["close"] / previous_close)

    return df


def add_requested_scaled_features(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Adds requested scaled features.

    Examples:
    - rsi_14_scaled from rsi_14
    - mfi_14_scaled from mfi_14
    - adx_14_scaled from adx_14
    """
    df = df.copy()

    scaled_feature_map = {
        "rsi_14_scaled": ("rsi_14", 100),
        "mfi_14_scaled": ("mfi_14", 100),
        "adx_14_scaled": ("adx_14", 100),
        "stochastic_k_14_scaled": ("stochastic_k_14", 100),
        "stochastic_d_14_3_scaled": ("stochastic_d_14_3", 100),
    }

    for target_column, (source_column, divisor) in scaled_feature_map.items():
        if target_column in features:
            validate_required_columns(df, {source_column})
            df[target_column] = df[source_column] / divisor

    return df


def add_requested_change_features(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Adds requested change/delta features.
    """
    df = df.copy()

    change_feature_map = {
        "macd_histogram_change_3d": ("macd_histogram_pct", 3),
        "rsi_14_change_3d": ("rsi_14", 3),
        "volume_ratio_20_change_3d": ("volume_ratio_20", 3),
    }

    for target_column, (source_column, periods) in change_feature_map.items():
        if target_column in features:
            validate_required_columns(df, {source_column})
            df[target_column] = df[source_column] - df[source_column].shift(periods)

    return df


def add_requested_shifted_features(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Adds requested shifted features.

    Naming convention:
    <source_feature>_shifted_<periods>

    Example:
    rsi_14_scaled_shifted_1
    macd_histogram_pct_shifted_1
    """
    df = df.copy()

    shifted_suffix = "_shifted_"

    shifted_features = [
        feature for feature in features
        if shifted_suffix in feature
    ]

    for target_column in shifted_features:
        source_column, periods_text = target_column.rsplit(shifted_suffix, 1)

        try:
            periods = int(periods_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid shifted feature name: {target_column}"
            ) from exc

        # If source itself is derived, create it first.
        df = add_requested_base_return_features(df, [source_column])
        df = add_requested_scaled_features(df, [source_column])
        df = add_requested_change_features(df, [source_column])

        validate_required_columns(df, {source_column})

        df[target_column] = df[source_column].shift(periods)

    return df