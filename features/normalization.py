"""
Feature normalization utilities.

This module contains reusable helpers for preparing indicator columns
for machine learning models.

It should not contain model-specific feature lists.
Those belong in feature_sets.py.
"""

import numpy as np
import pandas as pd


DEFAULT_CLIP_CONFIG = {
    # Returns
    "return_1d": (-0.30, 0.30),
    "return_5d": (-0.50, 0.50),
    "return_20d": (-1.00, 1.00),
    "log_return_1d": (-0.30, 0.30),

    # Trend distances
    "close_vs_sma_20": (-0.50, 0.50),
    "close_vs_sma_50": (-0.75, 0.75),
    "close_vs_sma_200": (-1.00, 1.00),
    "close_vs_ema_20": (-0.50, 0.50),
    "close_vs_ema_50": (-0.75, 0.75),
    "close_vs_ema_200": (-1.00, 1.00),

    # Momentum / bounded indicators
    "rsi_14_scaled": (0.0, 1.0),
    "mfi_14_scaled": (0.0, 1.0),
    "adx_14_scaled": (0.0, 1.0),
    "stochastic_k_14": (0.0, 100.0),
    "stochastic_d_14_3": (0.0, 100.0),
    "williams_r_14": (-100.0, 0.0),

    # Volatility
    "atr_pct_14": (0.0, 0.50),
    "atr_pct_20": (0.0, 0.50),
    "bollinger_width_20": (0.0, 1.0),
    "rolling_volatility_20": (0.0, 0.50),

    # Volume
    "volume_ratio_20": (0.0, 10.0),
    "cmf_20": (-1.0, 1.0),

    # Candlestick
    "body_pct_of_range": (0.0, 1.0),
    "upper_wick_pct_of_range": (0.0, 1.0),
    "lower_wick_pct_of_range": (0.0, 1.0),
}


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str] | set[str],
) -> None:
    """
    Validates that required columns exist in the DataFrame.
    """
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def replace_infinite_values(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces positive and negative infinite values with NaN.
    """
    feature_df = feature_df.copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    return feature_df


def clip_feature(
    feature_df: pd.DataFrame,
    column: str,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    """
    Clips a single feature column to a fixed lower and upper bound.
    """
    feature_df = feature_df.copy()

    validate_required_columns(feature_df, {column})

    feature_df[column] = feature_df[column].clip(lower=lower, upper=upper)

    return feature_df


def clip_features(
    feature_df: pd.DataFrame,
    clip_config: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """
    Clips multiple feature columns based on a configuration dictionary.

    Missing columns are ignored on purpose, because different feature sets
    may contain different subsets of features.
    """
    feature_df = feature_df.copy()

    for column, (lower, upper) in clip_config.items():
        if column in feature_df.columns:
            feature_df[column] = feature_df[column].clip(lower=lower, upper=upper)

    return feature_df


def scale_0_100_to_0_1(
    feature_df: pd.DataFrame,
    source_column: str,
    target_column: str | None = None,
) -> pd.DataFrame:
    """
    Scales an indicator from 0-100 to 0-1.

    Useful for:
    - RSI
    - MFI
    - Stochastic %K/%D
    - ADX, if desired
    """
    feature_df = feature_df.copy()

    validate_required_columns(feature_df, {source_column})

    if target_column is None:
        target_column = f"{source_column}_scaled"

    feature_df[target_column] = feature_df[source_column] / 100

    return feature_df


def create_relative_to_close(
    feature_df: pd.DataFrame,
    source_column: str,
    target_column: str | None = None,
) -> pd.DataFrame:
    """
    Converts a price-level column into a relative close-based feature.

    Example:
    sma_20 -> close_vs_sma_20 = close / sma_20 - 1
    """
    feature_df = feature_df.copy()

    validate_required_columns(feature_df, {"close", source_column})

    if target_column is None:
        target_column = f"close_vs_{source_column}"

    denominator = feature_df[source_column].replace(0, np.nan)

    feature_df[target_column] = feature_df["close"] / denominator - 1

    return feature_df


def create_relative_to_open(
    feature_df: pd.DataFrame,
    source_column: str,
    target_column: str | None = None,
) -> pd.DataFrame:
    """
    Converts a price-level column into a relative open-based feature.

    Useful for after-open models.

    Example:
    pivot_point -> open_vs_pivot_point = open / pivot_point - 1
    """
    feature_df = feature_df.copy()

    validate_required_columns(feature_df, {"open", source_column})

    if target_column is None:
        target_column = f"open_vs_{source_column}"

    denominator = feature_df[source_column].replace(0, np.nan)

    feature_df[target_column] = feature_df["open"] / denominator - 1

    return feature_df


def create_shifted_features(
    feature_df: pd.DataFrame,
    columns: list[str],
    periods: int = 1,
    suffix: str | None = None,
) -> pd.DataFrame:
    """
    Creates shifted versions of selected columns.

    Useful when a model runs after today's open and should only use
    indicators known at the previous close.

    Example:
    rsi_14 -> rsi_14_shifted_1
    """
    feature_df = feature_df.copy()

    if periods < 1:
        raise ValueError("periods must be >= 1")

    if suffix is None:
        suffix = f"shifted_{periods}"

    validate_required_columns(feature_df, set(columns))

    for column in columns:
        feature_df[f"{column}_{suffix}"] = feature_df[column].shift(periods)

    return feature_df


def zscore_column(
    feature_df: pd.DataFrame,
    column: str,
    rolling_window: int | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Creates a z-score version of a column.

    Prefer rolling_window for time series to avoid using future information.

    If rolling_window is None, full-column mean/std are used.
    Use full-column z-score only after a proper train/test split or inside
    a fitted preprocessing pipeline.
    """
    feature_df = feature_df.copy()

    validate_required_columns(feature_df, {column})

    if rolling_window is None:
        mean = feature_df[column].mean()
        std = feature_df[column].std()

        if std == 0 or pd.isna(std):
            feature_df[f"{column}_zscore"] = np.nan
        else:
            feature_df[f"{column}_zscore"] = (feature_df[column] - mean) / std

        return feature_df

    if rolling_window < 2:
        raise ValueError("rolling_window must be >= 2")

    if min_periods is None:
        min_periods = rolling_window

    mean = feature_df[column].rolling(
        window=rolling_window,
        min_periods=min_periods,
    ).mean()

    std = feature_df[column].rolling(
        window=rolling_window,
        min_periods=min_periods,
    ).std()

    feature_df[f"{column}_zscore_{rolling_window}"] = (
        (feature_df[column] - mean) / std.replace(0, np.nan)
    )

    return feature_df


def rolling_rank_percentile(
    feature_df: pd.DataFrame,
    column: str,
    window: int,
) -> pd.DataFrame:
    """
    Creates a rolling percentile-rank feature.

    Example:
    volume_ratio_20_rolling_rank_252 tells where today's value ranks
    compared to the previous rolling window.

    This can be useful for regime-normalizing noisy indicators.
    """
    feature_df = feature_df.copy()

    validate_required_columns(feature_df, {column})

    if window < 2:
        raise ValueError("window must be >= 2")

    target_column = f"{column}_rolling_rank_{window}"

    def _rank_last_value(values: np.ndarray) -> float:
        last_value = values[-1]
        return np.sum(values <= last_value) / len(values)

    feature_df[target_column] = feature_df[column].rolling(
        window=window,
        min_periods=window,
    ).apply(_rank_last_value, raw=True)

    return feature_df


def drop_columns_if_exists(
    feature_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Drops columns if they exist.
    """
    feature_df = feature_df.copy()
    existing_columns = [column for column in columns if column in feature_df.columns]
    return feature_df.drop(columns=existing_columns)


def select_feature_columns(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    keep_metadata_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keeps only selected feature columns and optional metadata columns.

    Metadata columns could be:
    - ticker
    - date
    - target
    - future_return
    """
    feature_df = feature_df.copy()

    if keep_metadata_columns is None:
        keep_metadata_columns = []

    requested_columns = keep_metadata_columns + feature_columns

    validate_required_columns(feature_df, requested_columns)

    return feature_df[requested_columns]


def clean_model_features(
    model_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str | None = None,
    clip_config: dict[str, tuple[float, float]] | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Cleans final model DataFrame before training.

    Steps:
    - validates selected feature/target columns
    - replaces inf/-inf with NaN
    - optionally clips selected features
    - optionally drops rows with missing feature/target values
    """
    model_df = model_df.copy()

    validate_required_columns(model_df, feature_columns)

    if target_column is not None:
        validate_required_columns(model_df, {target_column})

    model_df = replace_infinite_values(model_df)

    if clip_config is not None:
        model_df = clip_features(model_df, clip_config)

    if dropna:
        subset = feature_columns.copy()

        if target_column is not None:
            subset.append(target_column)

        model_df = model_df.dropna(subset=subset)

    return model_df