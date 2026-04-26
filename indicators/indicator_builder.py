"""
Indicator builder.

Builds complete indicator sets for one ticker DataFrame.
"""

import pandas as pd

from indicators.trend import calculate_all_indicators as calculate_trend_indicators
from indicators.momentum import calculate_all_indicators as calculate_momentum_indicators
from indicators.volatility import calculate_all_indicators as calculate_volatility_indicators
from indicators.volume import calculate_all_indicators as calculate_volume_indicators
from indicators.price_levels import calculate_all_indicators as calculate_price_level_indicators
from indicators.candlestick import calculate_all_indicators as calculate_candlestick_indicators


def _validate_price_df(price_df: pd.DataFrame) -> None:
    """
    Validates that the required OHLCV columns exist.
    """
    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns - set(price_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def build_indicators_for_ticker(
    price_df: pd.DataFrame,
    indicator_set: str = "default",
) -> pd.DataFrame:
    """
    Builds indicators for one ticker.

    Supported indicator sets:
    - minimal
    - default
    - full
    """
    _validate_price_df(price_df)

    price_df = price_df.copy()

    if indicator_set == "minimal":
        return build_minimal_indicators(price_df)

    if indicator_set == "default":
        return build_default_indicators(price_df)

    if indicator_set == "full":
        return build_full_indicators(price_df)

    raise ValueError(f"Unknown indicator_set: {indicator_set}")


def build_minimal_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a compact indicator set for fast experiments.
    """
    price_df = price_df.copy()

    from indicators.trend import (
        calculate_sma,
        calculate_ema,
        calculate_macd,
        calculate_price_vs_sma,
        calculate_price_vs_ema,
    )
    from indicators.momentum import (
        calculate_rsi,
        calculate_roc,
    )
    from indicators.volatility import (
        calculate_true_range,
        calculate_atr,
        calculate_bollinger_bands,
    )
    from indicators.volume import (
        calculate_volume_sma,
        calculate_volume_ratio,
    )
    from indicators.price_levels import (
        calculate_rolling_high,
        calculate_rolling_low,
        calculate_distance_to_rolling_high,
        calculate_distance_to_rolling_low,
        calculate_position_in_rolling_range,
    )
    from indicators.candlestick import calculate_candle_parts

    # Trend
    for period in [20, 50]:
        price_df = calculate_sma(price_df, period=period)
        price_df = calculate_ema(price_df, period=period)
        price_df = calculate_price_vs_sma(price_df, period=period)
        price_df = calculate_price_vs_ema(price_df, period=period)

    price_df = calculate_macd(price_df)

    # Momentum
    price_df = calculate_rsi(price_df, period=14)
    price_df = calculate_roc(price_df, period=14)

    # Volatility
    price_df = calculate_true_range(price_df)
    price_df = calculate_atr(price_df, period=14)
    price_df = calculate_bollinger_bands(price_df, period=20)

    # Volume
    price_df = calculate_volume_sma(price_df, period=20)
    price_df = calculate_volume_ratio(price_df, period=20)

    # Price levels
    price_df = calculate_rolling_high(price_df, period=20)
    price_df = calculate_rolling_low(price_df, period=20)
    price_df = calculate_distance_to_rolling_high(price_df, period=20)
    price_df = calculate_distance_to_rolling_low(price_df, period=20)
    price_df = calculate_position_in_rolling_range(price_df, period=20)

    # Candlestick shape only
    price_df = calculate_candle_parts(price_df)

    return price_df


def build_default_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a balanced default indicator set.

    Recommended starting point for model experiments.
    """
    price_df = price_df.copy()

    price_df = calculate_trend_indicators(price_df)
    price_df = calculate_momentum_indicators(price_df)
    price_df = calculate_volatility_indicators(price_df)
    price_df = calculate_volume_indicators(price_df)
    price_df = calculate_price_level_indicators(price_df)
    price_df = calculate_candlestick_indicators(price_df)

    return price_df


def build_full_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a broad exploratory indicator set.

    For now this runs all indicator families.
    Later this can be expanded with more periods and experimental indicators.
    """
    price_df = price_df.copy()

    price_df = calculate_trend_indicators(price_df)
    price_df = calculate_momentum_indicators(price_df)
    price_df = calculate_volatility_indicators(price_df)
    price_df = calculate_volume_indicators(price_df)
    price_df = calculate_price_level_indicators(price_df)
    price_df = calculate_candlestick_indicators(price_df)

    return price_df