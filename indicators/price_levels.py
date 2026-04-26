"""
Rolling high
Rolling low
Distance to rolling high
Distance to rolling low
Breakout above N-day high
Breakdown below N-day low
Support/resistance approximation
Pivot points
"""

import pandas as pd
import numpy as np  

def calculate_rolling_high(price_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates the highest high over the previous N completed periods.

    Uses shift(1) to avoid look-ahead leakage.
    """
    price_df = price_df.copy()

    rolling_high_column = f"rolling_high_{period}"

    price_df[rolling_high_column] = (
        price_df["high"]
        .shift(1)
        .rolling(window=period, min_periods=period)
        .max()
    )

    return price_df


def calculate_rolling_low(price_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates the lowest low over the previous N completed periods.

    Uses shift(1) to avoid look-ahead leakage.
    """
    price_df = price_df.copy()

    rolling_low_column = f"rolling_low_{period}"

    price_df[rolling_low_column] = (
        price_df["low"]
        .shift(1)
        .rolling(window=period, min_periods=period)
        .min()
    )

    return price_df

def calculate_distance_to_rolling_high(price_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates close distance to previous N-period rolling high as a percentage.
    """
    price_df = price_df.copy()

    rolling_high_column = f"rolling_high_{period}"
    distance_to_high_column = f"close_vs_rolling_high_{period}"

    rolling_high_safe = price_df[rolling_high_column].replace(0, np.nan)

    price_df[distance_to_high_column] = price_df["close"] / rolling_high_safe - 1

    return price_df


def calculate_distance_to_rolling_low(price_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates close distance to previous N-period rolling low as a percentage.
    """
    price_df = price_df.copy()

    rolling_low_column = f"rolling_low_{period}"
    distance_to_low_column = f"close_vs_rolling_low_{period}"

    rolling_low_safe = price_df[rolling_low_column].replace(0, np.nan)

    price_df[distance_to_low_column] = price_df["close"] / rolling_low_safe - 1

    return price_df

def calculate_breakout_above_rolling_high(price_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Flags if close breaks above the previous N-period rolling high.
    """
    price_df = price_df.copy()

    rolling_high_column = f"rolling_high_{period}"
    breakout_column = f"close_breakout_above_rolling_high_{period}"

    price_df[breakout_column] = (
        price_df["close"] > price_df[rolling_high_column]
    ).astype(int)

    return price_df


def calculate_breakdown_below_rolling_low(price_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Flags if close breaks below the previous N-period rolling low.
    """
    price_df = price_df.copy()

    rolling_low_column = f"rolling_low_{period}"
    breakdown_column = f"close_breakdown_below_rolling_low_{period}"

    price_df[breakdown_column] = (
        price_df["close"] < price_df[rolling_low_column]
    ).astype(int)

    return price_df

def calculate_position_in_rolling_range(price_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates where close is positioned inside the previous N-period high/low range.

    0 = close is at previous rolling low
    1 = close is at previous rolling high
    > 1 = close has broken above previous rolling high
    < 0 = close has broken below previous rolling low
    """
    price_df = price_df.copy()

    rolling_high_column = f"rolling_high_{period}"
    rolling_low_column = f"rolling_low_{period}"
    position_column = f"close_position_in_rolling_range_{period}"
    range_width_column = f"rolling_range_width_pct_{period}"

    rolling_high = price_df[rolling_high_column]
    rolling_low = price_df[rolling_low_column]

    rolling_range = rolling_high - rolling_low
    rolling_range_safe = rolling_range.replace(0, np.nan)

    price_df[position_column] = (
        price_df["close"] - rolling_low
    ) / rolling_range_safe

    price_df[range_width_column] = (
        rolling_range / price_df["close"].replace(0, np.nan)
    )

    return price_df


def calculate_support_resistance(
    price_df: pd.DataFrame,
    period: int = 20,
) -> pd.DataFrame:
    """
    Calculates rolling support and resistance levels.

    Support = lowest low over previous N completed periods.
    Resistance = highest high over previous N completed periods.

    Uses shift(1) to avoid look-ahead leakage.

    Required columns:
    - high
    - low
    - close

    Optional columns:
    - open
    """
    required_columns = {"high", "low", "close"}
    missing_columns = required_columns - set(price_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    price_df = price_df.copy()

    support = (
        price_df["low"]
        .shift(1)
        .rolling(window=period, min_periods=period)
        .min()
    )

    resistance = (
        price_df["high"]
        .shift(1)
        .rolling(window=period, min_periods=period)
        .max()
    )

    sr_range = resistance - support
    sr_range_safe = sr_range.replace(0, np.nan)

    support_safe = support.replace(0, np.nan)
    resistance_safe = resistance.replace(0, np.nan)
    close_safe = price_df["close"].replace(0, np.nan)

    price_df[f"support_{period}"] = support
    price_df[f"resistance_{period}"] = resistance
    price_df[f"sr_range_width_pct_{period}"] = sr_range / close_safe

    price_df[f"close_vs_support_{period}"] = price_df["close"] / support_safe - 1
    price_df[f"close_vs_resistance_{period}"] = price_df["close"] / resistance_safe - 1

    price_df[f"close_position_in_sr_range_{period}"] = (
        price_df["close"] - support
    ) / sr_range_safe

    price_df[f"close_above_resistance_{period}"] = (
        price_df["close"] > resistance
    ).astype(int)

    price_df[f"close_below_support_{period}"] = (
        price_df["close"] < support
    ).astype(int)

    if "open" in price_df.columns:
        open_safe = price_df["open"].replace(0, np.nan)

        price_df[f"open_vs_support_{period}"] = price_df["open"] / support_safe - 1
        price_df[f"open_vs_resistance_{period}"] = price_df["open"] / resistance_safe - 1

        price_df[f"open_position_in_sr_range_{period}"] = (
            price_df["open"] - support
        ) / sr_range_safe

        price_df[f"open_above_resistance_{period}"] = (
            price_df["open"] > resistance
        ).astype(int)

        price_df[f"open_below_support_{period}"] = (
            price_df["open"] < support
        ).astype(int)

    return price_df

import numpy as np
import pandas as pd


def calculate_pivot_points(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates classic pivot points using previous completed session values.

    Adds both raw pivot levels and relative distance features.
    """
    required_columns = {"high", "low", "close"}
    missing_columns = required_columns - set(price_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    price_df = price_df.copy()

    previous_high = price_df["high"].shift(1)
    previous_low = price_df["low"].shift(1)
    previous_close = price_df["close"].shift(1)

    pivot_point = (previous_high + previous_low + previous_close) / 3
    previous_range = previous_high - previous_low

    support_1 = (2 * pivot_point) - previous_high
    resistance_1 = (2 * pivot_point) - previous_low
    support_2 = pivot_point - previous_range
    resistance_2 = pivot_point + previous_range

    price_df["pivot_point"] = pivot_point
    price_df["pivot_support_1"] = support_1
    price_df["pivot_resistance_1"] = resistance_1
    price_df["pivot_support_2"] = support_2
    price_df["pivot_resistance_2"] = resistance_2
    price_df["pivot_range_pct"] = (
        resistance_2 - support_2
    ) / price_df["close"].replace(0, np.nan)

    close_safe = price_df["close"].replace(0, np.nan)

    price_df["close_vs_pivot_point"] = price_df["close"] / pivot_point.replace(0, np.nan) - 1
    price_df["close_vs_pivot_support_1"] = price_df["close"] / support_1.replace(0, np.nan) - 1
    price_df["close_vs_pivot_resistance_1"] = price_df["close"] / resistance_1.replace(0, np.nan) - 1
    price_df["close_vs_pivot_support_2"] = price_df["close"] / support_2.replace(0, np.nan) - 1
    price_df["close_vs_pivot_resistance_2"] = price_df["close"] / resistance_2.replace(0, np.nan) - 1

    price_df["close_above_pivot_point"] = (price_df["close"] > pivot_point).astype(int)
    price_df["close_above_pivot_resistance_1"] = (price_df["close"] > resistance_1).astype(int)
    price_df["close_above_pivot_resistance_2"] = (price_df["close"] > resistance_2).astype(int)
    price_df["close_below_pivot_support_1"] = (price_df["close"] < support_1).astype(int)
    price_df["close_below_pivot_support_2"] = (price_df["close"] < support_2).astype(int)

    if "open" in price_df.columns:
        price_df["open_vs_pivot_point"] = price_df["open"] / pivot_point.replace(0, np.nan) - 1
        price_df["open_vs_pivot_support_1"] = price_df["open"] / support_1.replace(0, np.nan) - 1
        price_df["open_vs_pivot_resistance_1"] = price_df["open"] / resistance_1.replace(0, np.nan) - 1
        price_df["open_vs_pivot_support_2"] = price_df["open"] / support_2.replace(0, np.nan) - 1
        price_df["open_vs_pivot_resistance_2"] = price_df["open"] / resistance_2.replace(0, np.nan) - 1

        price_df["open_above_pivot_point"] = (price_df["open"] > pivot_point).astype(int)
        price_df["open_above_pivot_resistance_1"] = (price_df["open"] > resistance_1).astype(int)
        price_df["open_above_pivot_resistance_2"] = (price_df["open"] > resistance_2).astype(int)
        price_df["open_below_pivot_support_1"] = (price_df["open"] < support_1).astype(int)
        price_df["open_below_pivot_support_2"] = (price_df["open"] < support_2).astype(int)

    return price_df

def calculate_all_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all price level indicators.
    """
    price_df = price_df.copy()

    for period in [20, 50, 100, 200]:
        price_df = calculate_rolling_high(price_df, period)
        price_df = calculate_rolling_low(price_df, period)
        price_df = calculate_distance_to_rolling_high(price_df, period)
        price_df = calculate_distance_to_rolling_low(price_df, period)
        price_df = calculate_position_in_rolling_range(price_df, period)
        price_df = calculate_breakout_above_rolling_high(price_df, period)
        price_df = calculate_breakdown_below_rolling_low(price_df, period)

    price_df = calculate_pivot_points(price_df)

    return price_df    


