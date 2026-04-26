"""
Volatility indicators.

Includes:
- True range
- ATR
- Bollinger Bands
- Bollinger width
- Bollinger %B
- Rolling volatility
- Donchian channel
- Keltner channel
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


def calculate_true_range(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates true range.

    True range is the maximum of:
    - high - low
    - abs(high - previous close)
    - abs(low - previous close)
    """
    _validate_required_columns(price_df, {"high", "low", "close"})

    price_df = price_df.copy()

    previous_close = price_df["close"].shift(1)

    high_low = price_df["high"] - price_df["low"]
    high_close = (price_df["high"] - previous_close).abs()
    low_close = (price_df["low"] - previous_close).abs()

    price_df["true_range"] = pd.concat(
        [high_low, high_close, low_close],
        axis=1,
    ).max(axis=1)

    price_df["true_range_pct"] = (
        price_df["true_range"] / price_df["close"].replace(0, np.nan)
    )

    return price_df


def calculate_atr(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates Average True Range using Wilder-style smoothing.

    Adds:
    - atr_{period}
    - atr_pct_{period}
    """
    _validate_required_columns(price_df, {"high", "low", "close"})

    price_df = price_df.copy()

    if "true_range" not in price_df.columns:
        price_df = calculate_true_range(price_df)

    price_df[f"atr_{period}"] = price_df["true_range"].ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    price_df[f"atr_pct_{period}"] = (
        price_df[f"atr_{period}"] / price_df["close"].replace(0, np.nan)
    )

    return price_df


def calculate_bollinger_bands(
    price_df: pd.DataFrame,
    period: int = 20,
    std_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Calculates Bollinger Bands and related features.

    Adds:
    - bollinger_middle_{period}
    - bollinger_upper_{period}
    - bollinger_lower_{period}
    - bollinger_width_{period}
    - bollinger_percent_b_{period}
    - close_above_bollinger_upper_{period}
    - close_below_bollinger_lower_{period}
    """
    _validate_required_columns(price_df, {"close"})

    price_df = price_df.copy()

    middle = price_df["close"].rolling(
        window=period,
        min_periods=period,
    ).mean()

    std_dev = price_df["close"].rolling(
        window=period,
        min_periods=period,
    ).std()

    upper = middle + std_multiplier * std_dev
    lower = middle - std_multiplier * std_dev

    band_width = upper - lower
    band_width_safe = band_width.replace(0, np.nan)

    middle_safe = middle.replace(0, np.nan)

    price_df[f"bollinger_middle_{period}"] = middle
    price_df[f"bollinger_upper_{period}"] = upper
    price_df[f"bollinger_lower_{period}"] = lower

    # ML-friendly relative width.
    price_df[f"bollinger_width_{period}"] = band_width / middle_safe

    # 0 = lower band, 1 = upper band.
    # >1 = above upper band, <0 = below lower band.
    price_df[f"bollinger_percent_b_{period}"] = (
        price_df["close"] - lower
    ) / band_width_safe

    price_df[f"close_vs_bollinger_middle_{period}"] = (
        price_df["close"] / middle_safe - 1
    )

    price_df[f"close_above_bollinger_upper_{period}"] = (
        price_df["close"] > upper
    ).astype(int)

    price_df[f"close_below_bollinger_lower_{period}"] = (
        price_df["close"] < lower
    ).astype(int)

    return price_df


def calculate_rolling_volatility(price_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculates rolling volatility based on close-to-close percentage returns.

    This is not annualized by default.
    """
    _validate_required_columns(price_df, {"close"})

    price_df = price_df.copy()

    returns = price_df["close"].pct_change()

    price_df[f"rolling_volatility_{period}"] = returns.rolling(
        window=period,
        min_periods=period,
    ).std()

    return price_df


def calculate_donchian_channel(price_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculates Donchian channel using previous completed periods.

    Uses shift(1) to avoid look-ahead leakage.

    Adds:
    - donchian_upper_{period}
    - donchian_lower_{period}
    - donchian_width_pct_{period}
    - close_position_in_donchian_{period}
    - close_above_donchian_upper_{period}
    - close_below_donchian_lower_{period}
    """
    _validate_required_columns(price_df, {"high", "low", "close"})

    price_df = price_df.copy()

    upper = price_df["high"].shift(1).rolling(
        window=period,
        min_periods=period,
    ).max()

    lower = price_df["low"].shift(1).rolling(
        window=period,
        min_periods=period,
    ).min()

    channel_width = upper - lower
    channel_width_safe = channel_width.replace(0, np.nan)

    price_df[f"donchian_upper_{period}"] = upper
    price_df[f"donchian_lower_{period}"] = lower

    price_df[f"donchian_width_pct_{period}"] = (
        channel_width / price_df["close"].replace(0, np.nan)
    )

    price_df[f"close_position_in_donchian_{period}"] = (
        price_df["close"] - lower
    ) / channel_width_safe

    price_df[f"close_above_donchian_upper_{period}"] = (
        price_df["close"] > upper
    ).astype(int)

    price_df[f"close_below_donchian_lower_{period}"] = (
        price_df["close"] < lower
    ).astype(int)

    return price_df


def calculate_keltner_channel(
    price_df: pd.DataFrame,
    period: int = 20,
    atr_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Calculates Keltner Channel using EMA and ATR.

    Adds:
    - keltner_middle_{period}
    - keltner_upper_{period}
    - keltner_lower_{period}
    - keltner_width_pct_{period}
    - close_position_in_keltner_{period}
    """
    _validate_required_columns(price_df, {"high", "low", "close"})

    price_df = price_df.copy()

    atr_column = f"atr_{period}"

    if atr_column not in price_df.columns:
        price_df = calculate_atr(price_df, period=period)

    middle = price_df["close"].ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()

    atr = price_df[atr_column]

    upper = middle + atr_multiplier * atr
    lower = middle - atr_multiplier * atr

    channel_width = upper - lower
    channel_width_safe = channel_width.replace(0, np.nan)

    price_df[f"keltner_middle_{period}"] = middle
    price_df[f"keltner_upper_{period}"] = upper
    price_df[f"keltner_lower_{period}"] = lower

    price_df[f"keltner_width_pct_{period}"] = (
        channel_width / middle.replace(0, np.nan)
    )

    price_df[f"close_position_in_keltner_{period}"] = (
        price_df["close"] - lower
    ) / channel_width_safe

    price_df[f"close_above_keltner_upper_{period}"] = (
        price_df["close"] > upper
    ).astype(int)

    price_df[f"close_below_keltner_lower_{period}"] = (
        price_df["close"] < lower
    ).astype(int)

    return price_df


def calculate_all_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all volatility indicators.
    """
    price_df = price_df.copy()

    price_df = calculate_true_range(price_df)

    for period in [14, 20]:
        price_df = calculate_atr(price_df, period=period)
        price_df = calculate_bollinger_bands(price_df, period=period)
        price_df = calculate_rolling_volatility(price_df, period=period)
        price_df = calculate_donchian_channel(price_df, period=period)
        price_df = calculate_keltner_channel(price_df, period=period)

    return price_df