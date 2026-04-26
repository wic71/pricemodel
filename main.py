from datetime import date

import pandas as pd

from data_providers import PriceDataRequest, get_market_data_provider

from indicators.indicator_builder import build_indicators_for_ticker

from config.experiment_config import DATASET_CONFIG, TARGET_CONFIG

from features.feature_builder import build_model_dataset
from features.feature_sets import DEFAULT_FEATURES, INDICATOR_FEATURES

from features.targets import add_relative_binary_target_for_all_tickers
from features.feature_builder import build_model_dataset_from_existing_target

from features.balancing import (
    balance_by_undersampling,
    print_target_distribution,
    split_features_target_metadata,
)

from pathlib import Path

def print_missing_features(
    df: pd.DataFrame,
    features: list[str],
    label: str,
) -> None:
    """
    Prints missing features from a DataFrame.
    """
    missing_features = [
        feature for feature in features
        if feature not in df.columns
    ]

    if not missing_features:
        print(f"\nAll requested {label} features exist.")
        return

    print(f"\nMissing {label} features:")
    for feature in missing_features:
        print(f"- {feature}")


def load_price_data_for_tickers(
    tickers: list[str],
    start_date: date,
    end_date: date,
    interval: str = "1d",
    provider_name: str = "yfinance",
) -> pd.DataFrame:
    """
    Loads price data for multiple tickers and returns one combined DataFrame.
    """
    provider = get_market_data_provider(provider_name)

    price_dfs = []

    for ticker in tickers:
        request = PriceDataRequest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

        ticker_df = provider.get_price_history(request)

        if ticker_df is None or ticker_df.empty:
            print(f"No data returned for {ticker}")
            continue

        ticker_df = ticker_df.copy()

        if "ticker" not in ticker_df.columns:
            ticker_df["ticker"] = ticker

        price_dfs.append(ticker_df)

    if not price_dfs:
        raise ValueError("No price data was loaded for any ticker.")

    price_df = pd.concat(price_dfs, ignore_index=True)

    if "date" not in price_df.columns:
        raise ValueError("Loaded price data must contain a 'date' column.")

    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return price_df


def build_indicators_for_all_tickers(
    price_df: pd.DataFrame,
    indicator_set: str = "default",
) -> pd.DataFrame:
    """
    Builds indicators for all tickers.

    Indicators must be calculated per ticker so rolling windows do not leak
    across ticker boundaries.
    """
    required_columns = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing_columns = required_columns - set(price_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    indicator_dfs = []

    for ticker, group_df in price_df.groupby("ticker", sort=False):
        group_df = group_df.sort_values("date").reset_index(drop=True)

        indicator_df = build_indicators_for_ticker(
            group_df,
            indicator_set=indicator_set,
        )

        indicator_dfs.append(indicator_df)

    indicators_df = pd.concat(indicator_dfs, ignore_index=True)
    indicators_df = indicators_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return indicators_df


def build_training_frames(
    indicators_df: pd.DataFrame,
    features: list[str],
    target_horizon: int = 5,
    target_type: str = "three_class",
    threshold: float = 0.005,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Builds final training frames.

    Returns:
    - X: only model-ready features
    - y: target column
    - metadata_df: date/ticker/target for debugging/backtesting
    - model_df: features + target, no metadata
    """
    if target_type == "binary":
        target_column = f"target_{target_horizon}d_binary"
    elif target_type == "three_class":
        target_column = f"target_{target_horizon}d_3class"
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # Build with metadata first so we can split it out.
    model_with_metadata_df = build_model_dataset(
        price_df=indicators_df,
        features=features,
        target_horizon=target_horizon,
        target_type=target_type,
        threshold=threshold,
        target_column=target_column,
        include_metadata=True,
        include_debug_columns=False,
        dropna=True,
    )

    # Metadata is not used by the model, but kept for debugging/backtesting.
    metadata_df = model_with_metadata_df[["date", "ticker", target_column]].copy()

    # X and y are what you use for training.
    X = model_with_metadata_df[features].copy()
    y = model_with_metadata_df[target_column].copy()

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    metadata_df = metadata_df.reset_index(drop=True)

    # Final model_df: only features + target.
    model_df = pd.concat(
        [
            X,
            y.rename(target_column),
        ],
        axis=1,
    )

    return X, y, metadata_df, model_df

def load_benchmark_data(
    benchmark_ticker: str,
    start_date: date,
    end_date: date,
    interval: str = "1d",
    provider_name: str = "yfinance",
) -> pd.DataFrame:
    """
    Loads benchmark price data.

    Returns:
    - date
    - benchmark_close
    """
    provider = get_market_data_provider(provider_name)

    request = PriceDataRequest(
        ticker=benchmark_ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )

    benchmark_df = provider.get_price_history(request)

    if benchmark_df is None or benchmark_df.empty:
        raise ValueError(f"No benchmark data returned for {benchmark_ticker}")

    benchmark_df = benchmark_df.copy()
    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])

    benchmark_df = benchmark_df.sort_values("date").reset_index(drop=True)

    benchmark_df = benchmark_df.rename(
        columns={
            "close": "benchmark_close",
        }
    )

    return benchmark_df[["date", "benchmark_close"]]


def main() -> None:
    tickers = DATASET_CONFIG.tickers

    target_horizon = TARGET_CONFIG.target_horizon
    threshold = TARGET_CONFIG.threshold
    target_type = TARGET_CONFIG.target_type
    target_column = TARGET_CONFIG.target_column

    price_df = load_price_data_for_tickers(
        tickers=tickers,
        start_date=DATASET_CONFIG.start_date,
        end_date=DATASET_CONFIG.end_date,
        interval=DATASET_CONFIG.interval,
        provider_name=DATASET_CONFIG.provider_name,
    )

    if TARGET_CONFIG.relative_to_benchmark:
        if DATASET_CONFIG.benchmark_ticker is None:
            raise ValueError("benchmark_ticker must be set for relative target.")

        benchmark_df = load_benchmark_data(
            benchmark_ticker=DATASET_CONFIG.benchmark_ticker,
            start_date=DATASET_CONFIG.start_date,
            end_date=DATASET_CONFIG.end_date,
            interval=DATASET_CONFIG.interval,
            provider_name=DATASET_CONFIG.provider_name,
        )
    else:
        benchmark_df = None

    print("Loaded price data:")
    print(price_df[["date", "ticker", "open", "high", "low", "close", "volume"]].head())
    print(f"Rows: {len(price_df)}")

    indicators_df = build_indicators_for_all_tickers(
        price_df=price_df,
        indicator_set="default",
    )

    print("\nBuilt indicators:")
    print(indicators_df.head())
    print(f"Rows: {len(indicators_df)}")
    print(f"Columns: {len(indicators_df.columns)}")

    print_missing_features(
        indicators_df,
        INDICATOR_FEATURES,
        label="indicator",
    )

    if TARGET_CONFIG.relative_to_benchmark:
        indicators_df = add_relative_binary_target_for_all_tickers(
            df=indicators_df,
            benchmark_df=benchmark_df,
            horizon=target_horizon,
            threshold=threshold,
            price_column="close",
            benchmark_price_column="benchmark_close",
            target_column=target_column,
        )

        model_with_metadata_df = build_model_dataset_from_existing_target(
            price_df=indicators_df,
            features=DEFAULT_FEATURES,
            target_column=target_column,
            include_metadata=True,
            include_debug_columns=False,
            dropna=True,
        )

        metadata_df = model_with_metadata_df[["date", "ticker", target_column]].copy()
        X = model_with_metadata_df[DEFAULT_FEATURES].copy()
        y = model_with_metadata_df[target_column].copy()

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        metadata_df = metadata_df.reset_index(drop=True)

        model_df = pd.concat(
            [
                X,
                y.rename(target_column),
            ],
            axis=1,
        )

    else:
        X, y, metadata_df, model_df = build_training_frames(
            indicators_df=indicators_df,
            features=DEFAULT_FEATURES,
            target_horizon=target_horizon,
            target_type=target_type,
            threshold=threshold,
        )

    print_missing_features(
        X,
        DEFAULT_FEATURES,
        label="final model",
    )

    print("\nTarget column:")
    print(target_column)

    print("\nBefore balancing:")
    print_target_distribution(model_df, target_column)

    training_analysis_df = pd.concat(
        [
            metadata_df[["date", "ticker"]].reset_index(drop=True),
            X.reset_index(drop=True),
            y.rename(target_column).reset_index(drop=True),
        ],
        axis=1,
    )

    balanced_training_analysis_df = balance_by_undersampling(
        df=training_analysis_df,
        target_column=target_column,
        random_state=42,
    )

    print("\nAfter balancing:")
    print_target_distribution(balanced_training_analysis_df, target_column)

    X_balanced, y_balanced, metadata_balanced_df = split_features_target_metadata(
        df=balanced_training_analysis_df,
        feature_columns=DEFAULT_FEATURES,
        target_column=target_column,
        metadata_columns=["date", "ticker", target_column],
    )

    balanced_model_df = pd.concat(
        [
            X_balanced.reset_index(drop=True),
            y_balanced.rename(target_column).reset_index(drop=True),
        ],
        axis=1,
    )

    print("\nFinal training data:")
    print(X.head())
    print(y.head())

    print("\nBalanced training data:")
    print(X_balanced.head())
    print(y_balanced.head())

    print("\nMetadata:")
    print(metadata_df.head())

    print("\nBalanced metadata:")
    print(metadata_balanced_df.head())

    print("\nShapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"metadata_df: {metadata_df.shape}")
    print(f"model_df: {model_df.shape}")
    print(f"X_balanced: {X_balanced.shape}")
    print(f"y_balanced: {y_balanced.shape}")
    print(f"metadata_balanced_df: {metadata_balanced_df.shape}")
    print(f"balanced_model_df: {balanced_model_df.shape}")

    dataset_dir = Path("datasets")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    model_df.to_parquet(dataset_dir / "model_dataset.parquet", index=False)
    metadata_df.to_parquet(dataset_dir / "model_metadata.parquet", index=False)

    balanced_model_df.to_parquet(dataset_dir / "model_dataset_balanced.parquet", index=False)
    metadata_balanced_df.to_parquet(dataset_dir / "model_metadata_balanced.parquet", index=False)

    print("\nSaved:")
    print(dataset_dir / "model_dataset.parquet")
    print(dataset_dir / "model_metadata.parquet")
    print(dataset_dir / "model_dataset_balanced.parquet")
    print(dataset_dir / "model_metadata_balanced.parquet")



if __name__ == "__main__":
    main()