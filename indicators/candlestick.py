"""
Candlestick indicators.

Includes:
- Candle body
- Upper/lower wick
- Doji
- Hammer
- Shooting star
- Engulfing bullish/bearish
- Morning star
- Evening star
- Inside bar
- Outside bar
- Gap up/down
- Long body
- Long wick
"""

import numpy as np
import pandas as pd


def _validate_required_columns(price_df: pd.DataFrame, required_columns: set[str]) -> None:
    """
    Validates that required columns exist in the DataFrame.
    """
    missing_columns = required_columns - set(price_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def calculate_candle_parts(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates basic candle-shape features.

    These features are often more useful for ML than named candlestick patterns.
    """
    _validate_required_columns(price_df, {"open", "high", "low", "close"})

    price_df = price_df.copy()

    candle_range = price_df["high"] - price_df["low"]
    candle_range_safe = candle_range.replace(0, np.nan)

    body = price_df["close"] - price_df["open"]
    body_abs = body.abs()

    upper_wick = price_df["high"] - price_df[["open", "close"]].max(axis=1)
    lower_wick = price_df[["open", "close"]].min(axis=1) - price_df["low"]

    close_safe = price_df["close"].replace(0, np.nan)
    open_safe = price_df["open"].replace(0, np.nan)

    price_df["candle_range_pct"] = candle_range / close_safe
    price_df["candle_body_pct"] = body / open_safe
    price_df["candle_body_abs_pct"] = body_abs / open_safe

    price_df["body_pct_of_range"] = body_abs / candle_range_safe
    price_df["upper_wick_pct_of_range"] = upper_wick / candle_range_safe
    price_df["lower_wick_pct_of_range"] = lower_wick / candle_range_safe

    price_df["upper_wick_pct"] = upper_wick / close_safe
    price_df["lower_wick_pct"] = lower_wick / close_safe

    price_df["candle_direction"] = np.sign(body)

    return price_df


def calculate_doji(
    price_df: pd.DataFrame,
    body_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Flags doji candles.

    A doji is approximated as a candle where the body is very small
    relative to the full high-low range.
    """
    price_df = price_df.copy()

    if "body_pct_of_range" not in price_df.columns:
        price_df = calculate_candle_parts(price_df)

    price_df["doji"] = (
        price_df["body_pct_of_range"] <= body_threshold
    ).astype(int)

    return price_df


def calculate_hammer(
    price_df: pd.DataFrame,
    max_body_pct_of_range: float = 0.35,
    min_lower_wick_pct_of_range: float = 0.55,
    max_upper_wick_pct_of_range: float = 0.20,
) -> pd.DataFrame:
    """
    Flags hammer-like candles.

    Approximation:
    - relatively small body
    - long lower wick
    - small upper wick
    """
    price_df = price_df.copy()

    if "body_pct_of_range" not in price_df.columns:
        price_df = calculate_candle_parts(price_df)

    price_df["hammer"] = (
        (price_df["body_pct_of_range"] <= max_body_pct_of_range)
        & (price_df["lower_wick_pct_of_range"] >= min_lower_wick_pct_of_range)
        & (price_df["upper_wick_pct_of_range"] <= max_upper_wick_pct_of_range)
    ).astype(int)

    return price_df


def calculate_shooting_star(
    price_df: pd.DataFrame,
    max_body_pct_of_range: float = 0.35,
    min_upper_wick_pct_of_range: float = 0.55,
    max_lower_wick_pct_of_range: float = 0.20,
) -> pd.DataFrame:
    """
    Flags shooting-star-like candles.

    Approximation:
    - relatively small body
    - long upper wick
    - small lower wick
    """
    price_df = price_df.copy()

    if "body_pct_of_range" not in price_df.columns:
        price_df = calculate_candle_parts(price_df)

    price_df["shooting_star"] = (
        (price_df["body_pct_of_range"] <= max_body_pct_of_range)
        & (price_df["upper_wick_pct_of_range"] >= min_upper_wick_pct_of_range)
        & (price_df["lower_wick_pct_of_range"] <= max_lower_wick_pct_of_range)
    ).astype(int)

    return price_df


def calculate_engulfing_patterns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags bullish and bearish engulfing patterns.

    Uses candle bodies rather than full high-low ranges.
    """
    _validate_required_columns(price_df, {"open", "close"})

    price_df = price_df.copy()

    previous_open = price_df["open"].shift(1)
    previous_close = price_df["close"].shift(1)

    previous_bearish = previous_close < previous_open
    previous_bullish = previous_close > previous_open

    current_bullish = price_df["close"] > price_df["open"]
    current_bearish = price_df["close"] < price_df["open"]

    current_body_low = price_df[["open", "close"]].min(axis=1)
    current_body_high = price_df[["open", "close"]].max(axis=1)

    previous_body_low = pd.concat(
        [previous_open, previous_close],
        axis=1,
    ).min(axis=1)

    previous_body_high = pd.concat(
        [previous_open, previous_close],
        axis=1,
    ).max(axis=1)

    price_df["engulfing_bullish"] = (
        previous_bearish
        & current_bullish
        & (current_body_low <= previous_body_low)
        & (current_body_high >= previous_body_high)
    ).astype(int)

    price_df["engulfing_bearish"] = (
        previous_bullish
        & current_bearish
        & (current_body_low <= previous_body_low)
        & (current_body_high >= previous_body_high)
    ).astype(int)

    return price_df


def calculate_inside_outside_bar(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags inside and outside bars.

    Inside bar:
    - current high is lower than previous high
    - current low is higher than previous low

    Outside bar:
    - current high is higher than previous high
    - current low is lower than previous low
    """
    _validate_required_columns(price_df, {"high", "low"})

    price_df = price_df.copy()

    previous_high = price_df["high"].shift(1)
    previous_low = price_df["low"].shift(1)

    price_df["inside_bar"] = (
        (price_df["high"] < previous_high)
        & (price_df["low"] > previous_low)
    ).astype(int)

    price_df["outside_bar"] = (
        (price_df["high"] > previous_high)
        & (price_df["low"] < previous_low)
    ).astype(int)

    return price_df


def calculate_gap_patterns(
    price_df: pd.DataFrame,
    gap_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Flags gap up/down based on today's open vs previous close.

    gap_threshold is a decimal.
    Example:
    0.005 = 0.5% minimum gap.
    """
    _validate_required_columns(price_df, {"open", "close"})

    price_df = price_df.copy()

    previous_close = price_df["close"].shift(1).replace(0, np.nan)

    price_df["gap_pct"] = price_df["open"] / previous_close - 1

    price_df["gap_up"] = (
        price_df["gap_pct"] > gap_threshold
    ).astype(int)

    price_df["gap_down"] = (
        price_df["gap_pct"] < -gap_threshold
    ).astype(int)

    return price_df


def calculate_long_body(
    price_df: pd.DataFrame,
    period: int = 20,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Flags candles with unusually large bodies compared to recent average body size.
    """
    price_df = price_df.copy()

    if "candle_body_abs_pct" not in price_df.columns:
        price_df = calculate_candle_parts(price_df)

    average_body = price_df["candle_body_abs_pct"].rolling(
        window=period,
        min_periods=period,
    ).mean()

    price_df[f"long_body_{period}"] = (
        price_df["candle_body_abs_pct"] > multiplier * average_body
    ).astype(int)

    return price_df


def calculate_long_wicks(
    price_df: pd.DataFrame,
    wick_threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Flags candles with long upper/lower wicks relative to candle range.
    """
    price_df = price_df.copy()

    if "upper_wick_pct_of_range" not in price_df.columns:
        price_df = calculate_candle_parts(price_df)

    price_df["long_upper_wick"] = (
        price_df["upper_wick_pct_of_range"] >= wick_threshold
    ).astype(int)

    price_df["long_lower_wick"] = (
        price_df["lower_wick_pct_of_range"] >= wick_threshold
    ).astype(int)

    price_df["long_wick"] = (
        (price_df["long_upper_wick"] == 1)
        | (price_df["long_lower_wick"] == 1)
    ).astype(int)

    return price_df


def calculate_morning_evening_star(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags simplified morning star and evening star patterns.

    This is an approximation using three candles.

    Morning star approximation:
    - candle 1 bearish and relatively large
    - candle 2 small body
    - candle 3 bullish and closes above midpoint of candle 1 body

    Evening star approximation:
    - candle 1 bullish and relatively large
    - candle 2 small body
    - candle 3 bearish and closes below midpoint of candle 1 body
    """
    _validate_required_columns(price_df, {"open", "close"})

    price_df = price_df.copy()

    if "body_pct_of_range" not in price_df.columns:
        price_df = calculate_candle_parts(price_df)

    open_2 = price_df["open"].shift(2)
    close_2 = price_df["close"].shift(2)

    open_1 = price_df["open"].shift(1)
    close_1 = price_df["close"].shift(1)

    first_body_abs = (close_2 - open_2).abs()
    first_midpoint = (open_2 + close_2) / 2

    first_bearish = close_2 < open_2
    first_bullish = close_2 > open_2

    second_small_body = price_df["body_pct_of_range"].shift(1) <= 0.30

    current_bullish = price_df["close"] > price_df["open"]
    current_bearish = price_df["close"] < price_df["open"]

    # Require first candle body to be non-trivial relative to price.
    first_large_body = first_body_abs / open_2.replace(0, np.nan) > 0.005

    price_df["morning_star"] = (
        first_bearish
        & first_large_body
        & second_small_body
        & current_bullish
        & (price_df["close"] > first_midpoint)
    ).astype(int)

    price_df["evening_star"] = (
        first_bullish
        & first_large_body
        & second_small_body
        & current_bearish
        & (price_df["close"] < first_midpoint)
    ).astype(int)

    return price_df


def calculate_candlestick_patterns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all candlestick features and pattern flags.
    """
    price_df = price_df.copy()

    price_df = calculate_candle_parts(price_df)
    price_df = calculate_doji(price_df)
    price_df = calculate_hammer(price_df)
    price_df = calculate_shooting_star(price_df)
    price_df = calculate_engulfing_patterns(price_df)
    price_df = calculate_morning_evening_star(price_df)
    price_df = calculate_inside_outside_bar(price_df)
    price_df = calculate_gap_patterns(price_df, gap_threshold=0.005)
    price_df = calculate_long_body(price_df, period=20)
    price_df = calculate_long_wicks(price_df)

    return price_df


def calculate_all_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all candlestick indicators.
    """
    price_df = calculate_candlestick_patterns(price_df)
    return price_df