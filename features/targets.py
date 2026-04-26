"""
Target builders.

Creates future-return targets for model training.

Important:
Targets must be calculated per ticker, never across mixed ticker data.
"""

import numpy as np
import pandas as pd


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """
    Validates that required columns exist in the DataFrame.
    """
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def add_future_return_target(
    df: pd.DataFrame,
    horizon: int = 1,
    price_column: str = "close",
) -> pd.DataFrame:
    """
    Adds future return target for one ticker.

    Example:
    horizon=1 means next row's close compared to today's close.

    future_return_1d = close[t+1] / close[t] - 1
    """
    _validate_required_columns(df, {"date", price_column})

    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    future_return_column = f"future_return_{horizon}d"

    df[future_return_column] = (
        df[price_column].shift(-horizon) / df[price_column].replace(0, np.nan) - 1
    )

    return df


def add_binary_direction_target(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.0,
    price_column: str = "close",
    target_column: str | None = None,
) -> pd.DataFrame:
    """
    Adds binary direction target for one ticker.

    1 = future return is greater than threshold
    0 = future return is less than or equal to threshold
    NaN = future return is unknown
    """
    df = add_future_return_target(
        df=df,
        horizon=horizon,
        price_column=price_column,
    )

    future_return_column = f"future_return_{horizon}d"

    if target_column is None:
        target_column = f"target_{horizon}d_binary"

    df[target_column] = np.where(
        df[future_return_column].isna(),
        np.nan,
        (df[future_return_column] > threshold).astype(int),
    )

    return df


def add_three_class_direction_target(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.005,
    price_column: str = "close",
    target_column: str | None = None,
) -> pd.DataFrame:
    """
    Adds three-class direction target for one ticker.

    0 = negative
    1 = neutral
    2 = positive
    NaN = future return is unknown
    """
    df = add_future_return_target(
        df=df,
        horizon=horizon,
        price_column=price_column,
    )

    future_return_column = f"future_return_{horizon}d"

    if target_column is None:
        target_column = f"target_{horizon}d_3class"

    df[target_column] = np.nan

    df.loc[df[future_return_column] < -threshold, target_column] = 0
    df.loc[
        df[future_return_column].between(-threshold, threshold, inclusive="both"),
        target_column,
    ] = 1
    df.loc[df[future_return_column] > threshold, target_column] = 2

    return df


def add_multiple_targets(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    target_type: str = "three_class",
    threshold: float = 0.005,
    price_column: str = "close",
) -> pd.DataFrame:
    """
    Adds targets for multiple prediction horizons for one ticker.
    """
    if horizons is None:
        horizons = [1, 5, 10]

    df = df.copy()

    for horizon in horizons:
        if target_type == "binary":
            df = add_binary_direction_target(
                df=df,
                horizon=horizon,
                threshold=threshold,
                price_column=price_column,
            )
        elif target_type == "three_class":
            df = add_three_class_direction_target(
                df=df,
                horizon=horizon,
                threshold=threshold,
                price_column=price_column,
            )
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    return df

def add_relative_binary_target_for_all_tickers(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    horizon: int = 10,
    threshold: float = 0.005,
    price_column: str = "close",
    benchmark_price_column: str = "benchmark_close",
    target_column: str | None = None,
) -> pd.DataFrame:
    """
    Adds a binary target based on future return relative to benchmark.

    1 = stock future return beats benchmark future return by more than threshold
    0 = otherwise

    The calculation is done per ticker for stock returns.
    Benchmark future return is calculated once by date.
    """
    required_columns = {"date", "ticker", price_column}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    required_benchmark_columns = {"date", benchmark_price_column}
    missing_benchmark_columns = required_benchmark_columns - set(benchmark_df.columns)

    if missing_benchmark_columns:
        raise ValueError(f"Missing benchmark columns: {missing_benchmark_columns}")

    if target_column is None:
        target_column = f"target_{horizon}d_relative_binary"

    df = df.copy()
    benchmark_df = benchmark_df.copy()

    df["date"] = pd.to_datetime(df["date"])
    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])

    benchmark_df = benchmark_df.sort_values("date").reset_index(drop=True)

    benchmark_future_return_column = f"benchmark_future_return_{horizon}d"

    benchmark_df[benchmark_future_return_column] = (
        benchmark_df[benchmark_price_column].shift(-horizon)
        / benchmark_df[benchmark_price_column].replace(0, pd.NA)
        - 1
    )

    output_dfs = []

    for _, group_df in df.groupby("ticker", sort=False):
        group_df = group_df.sort_values("date").reset_index(drop=True)

        stock_future_return_column = f"future_return_{horizon}d"

        group_df[stock_future_return_column] = (
            group_df[price_column].shift(-horizon)
            / group_df[price_column].replace(0, pd.NA)
            - 1
        )

        output_dfs.append(group_df)

    df = pd.concat(output_dfs, ignore_index=True)

    df = df.merge(
        benchmark_df[["date", benchmark_future_return_column]],
        on="date",
        how="left",
    )

    relative_return_column = f"relative_future_return_{horizon}d"

    df[relative_return_column] = (
        df[stock_future_return_column] - df[benchmark_future_return_column]
    )

    df[target_column] = pd.NA

    valid_mask = df[relative_return_column].notna()

    df.loc[valid_mask, target_column] = (
        df.loc[valid_mask, relative_return_column] > threshold
    ).astype(int)

    return df