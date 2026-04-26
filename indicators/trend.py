"""
Trend indicators.

Includes:
- SMA
- EMA
- MACD
- ADX
- DMI / +DI / -DI
- Moving average slope
- Price vs moving average
- Moving average crossovers
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


def calculate_sma(price_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates simple moving average based on close.
    """
    _validate_required_columns(price_df, {"close"})

    price_df = price_df.copy()

    price_df[f"sma_{period}"] = (
        price_df["close"]
        .rolling(window=period, min_periods=period)
        .mean()
    )

    return price_df


def calculate_ema(price_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates exponential moving average based on close.
    """
    _validate_required_columns(price_df, {"close"})

    price_df = price_df.copy()

    price_df[f"ema_{period}"] = (
        price_df["close"]
        .ewm(span=period, adjust=False, min_periods=period)
        .mean()
    )

    return price_df


def calculate_macd(
    price_df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Calculates MACD line, signal line, histogram and normalized MACD features.
    """
    _validate_required_columns(price_df, {"close"})

    price_df = price_df.copy()

    fast_ema_column = f"ema_{fast_period}"
    slow_ema_column = f"ema_{slow_period}"

    if fast_ema_column not in price_df.columns:
        price_df = calculate_ema(price_df, fast_period)

    if slow_ema_column not in price_df.columns:
        price_df = calculate_ema(price_df, slow_period)

    macd_line = price_df[fast_ema_column] - price_df[slow_ema_column]
    signal_line = macd_line.ewm(
        span=signal_period,
        adjust=False,
        min_periods=signal_period,
    ).mean()
    histogram = macd_line - signal_line

    close_safe = price_df["close"].replace(0, np.nan)

    price_df["macd_line"] = macd_line
    price_df["macd_signal"] = signal_line
    price_df["macd_histogram"] = histogram

    # More ML-friendly normalized versions.
    price_df["macd_line_pct"] = macd_line / close_safe
    price_df["macd_signal_pct"] = signal_line / close_safe
    price_df["macd_histogram_pct"] = histogram / close_safe

    price_df["macd_above_signal"] = (macd_line > signal_line).astype(int)

    price_df["macd_crossed_above_signal"] = (
        (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    ).astype(int)

    price_df["macd_crossed_below_signal"] = (
        (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    ).astype(int)

    return price_df


def calculate_adx(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates ADX, +DI and -DI using Wilder-style smoothing.

    ADX measures trend strength.
    +DI and -DI describe directional movement.
    """
    _validate_required_columns(price_df, {"high", "low", "close"})

    price_df = price_df.copy()

    high = price_df["high"]
    low = price_df["low"]
    close = price_df["close"]

    previous_high = high.shift(1)
    previous_low = low.shift(1)
    previous_close = close.shift(1)

    up_move = high - previous_high
    down_move = previous_low - low

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0,
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0.0,
    )

    true_range_1 = high - low
    true_range_2 = (high - previous_close).abs()
    true_range_3 = (low - previous_close).abs()

    true_range = pd.concat(
        [true_range_1, true_range_2, true_range_3],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_dm_smoothed = pd.Series(
        plus_dm,
        index=price_df.index,
    ).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    minus_dm_smoothed = pd.Series(
        minus_dm,
        index=price_df.index,
    ).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    atr_safe = atr.replace(0, np.nan)

    plus_di = 100 * plus_dm_smoothed / atr_safe
    minus_di = 100 * minus_dm_smoothed / atr_safe

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    price_df[f"plus_di_{period}"] = plus_di
    price_df[f"minus_di_{period}"] = minus_di
    price_df[f"adx_{period}"] = adx

    price_df[f"di_spread_{period}"] = plus_di - minus_di
    price_df[f"di_spread_pct_{period}"] = (plus_di - minus_di) / di_sum
    price_df[f"plus_di_above_minus_di_{period}"] = (plus_di > minus_di).astype(int)

    return price_df


def calculate_price_vs_sma(price_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates close position relative to SMA as percentage.
    """
    _validate_required_columns(price_df, {"close"})

    price_df = price_df.copy()

    sma_column = f"sma_{period}"

    if sma_column not in price_df.columns:
        price_df = calculate_sma(price_df, period)

    sma_safe = price_df[sma_column].replace(0, np.nan)

    price_df[f"close_vs_sma_{period}"] = price_df["close"] / sma_safe - 1
    price_df[f"close_above_sma_{period}"] = (price_df["close"] > price_df[sma_column]).astype(int)

    return price_df


def calculate_price_vs_ema(price_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates close position relative to EMA as percentage.
    """
    _validate_required_columns(price_df, {"close"})

    price_df = price_df.copy()

    ema_column = f"ema_{period}"

    if ema_column not in price_df.columns:
        price_df = calculate_ema(price_df, period)

    ema_safe = price_df[ema_column].replace(0, np.nan)

    price_df[f"close_vs_ema_{period}"] = price_df["close"] / ema_safe - 1
    price_df[f"close_above_ema_{period}"] = (price_df["close"] > price_df[ema_column]).astype(int)

    return price_df


def calculate_sma_slope(
    price_df: pd.DataFrame,
    period: int,
    slope_period: int = 5,
) -> pd.DataFrame:
    """
    Calculates SMA slope as percentage change over slope_period.
    """
    price_df = price_df.copy()

    sma_column = f"sma_{period}"

    if sma_column not in price_df.columns:
        price_df = calculate_sma(price_df, period)

    previous_sma = price_df[sma_column].shift(slope_period).replace(0, np.nan)

    price_df[f"sma_{period}_slope_{slope_period}"] = (
        price_df[sma_column] / previous_sma - 1
    )

    return price_df


def calculate_ema_slope(
    price_df: pd.DataFrame,
    period: int,
    slope_period: int = 5,
) -> pd.DataFrame:
    """
    Calculates EMA slope as percentage change over slope_period.
    """
    price_df = price_df.copy()

    ema_column = f"ema_{period}"

    if ema_column not in price_df.columns:
        price_df = calculate_ema(price_df, period)

    previous_ema = price_df[ema_column].shift(slope_period).replace(0, np.nan)

    price_df[f"ema_{period}_slope_{slope_period}"] = (
        price_df[ema_column] / previous_ema - 1
    )

    return price_df


def calculate_sma_crossovers(
    price_df: pd.DataFrame,
    short_period: int,
    long_period: int,
) -> pd.DataFrame:
    """
    Calculates SMA crossover state and crossover events.
    """
    price_df = price_df.copy()

    short_column = f"sma_{short_period}"
    long_column = f"sma_{long_period}"

    if short_column not in price_df.columns:
        price_df = calculate_sma(price_df, short_period)

    if long_column not in price_df.columns:
        price_df = calculate_sma(price_df, long_period)

    short_sma = price_df[short_column]
    long_sma = price_df[long_column]

    price_df[f"sma_{short_period}_above_sma_{long_period}"] = (
        short_sma > long_sma
    ).astype(int)

    price_df[f"sma_{short_period}_vs_sma_{long_period}"] = (
        short_sma / long_sma.replace(0, np.nan) - 1
    )

    price_df[f"sma_{short_period}_crossed_above_sma_{long_period}"] = (
        (short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1))
    ).astype(int)

    price_df[f"sma_{short_period}_crossed_below_sma_{long_period}"] = (
        (short_sma < long_sma) & (short_sma.shift(1) >= long_sma.shift(1))
    ).astype(int)

    return price_df


def calculate_ema_crossovers(
    price_df: pd.DataFrame,
    short_period: int,
    long_period: int,
) -> pd.DataFrame:
    """
    Calculates EMA crossover state and crossover events.
    """
    price_df = price_df.copy()

    short_column = f"ema_{short_period}"
    long_column = f"ema_{long_period}"

    if short_column not in price_df.columns:
        price_df = calculate_ema(price_df, short_period)

    if long_column not in price_df.columns:
        price_df = calculate_ema(price_df, long_period)

    short_ema = price_df[short_column]
    long_ema = price_df[long_column]

    price_df[f"ema_{short_period}_above_ema_{long_period}"] = (
        short_ema > long_ema
    ).astype(int)

    price_df[f"ema_{short_period}_vs_ema_{long_period}"] = (
        short_ema / long_ema.replace(0, np.nan) - 1
    )

    price_df[f"ema_{short_period}_crossed_above_ema_{long_period}"] = (
        (short_ema > long_ema) & (short_ema.shift(1) <= long_ema.shift(1))
    ).astype(int)

    price_df[f"ema_{short_period}_crossed_below_ema_{long_period}"] = (
        (short_ema < long_ema) & (short_ema.shift(1) >= long_ema.shift(1))
    ).astype(int)

    return price_df


def calculate_all_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all trend indicators.
    """
    price_df = price_df.copy()

    for period in [10, 20, 50, 200]:
        price_df = calculate_sma(price_df, period)
        price_df = calculate_price_vs_sma(price_df, period)
        price_df = calculate_sma_slope(price_df, period, slope_period=5)

        price_df = calculate_ema(price_df, period)
        price_df = calculate_price_vs_ema(price_df, period)
        price_df = calculate_ema_slope(price_df, period, slope_period=5)

    # MACD uses EMA 12 and EMA 26 internally if they do not already exist.
    price_df = calculate_macd(price_df)

    price_df = calculate_adx(price_df, period=14)

    price_df = calculate_sma_crossovers(price_df, 20, 50)
    price_df = calculate_sma_crossovers(price_df, 50, 200)
    price_df = calculate_ema_crossovers(price_df, 12, 26)

    return price_df