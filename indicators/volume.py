"""
Volume indicators.

Includes:
- Volume SMA
- Volume ratio
- OBV
- VWAP approximation
- Money Flow Index
- Accumulation/Distribution Line
- Chaikin Money Flow
- Volume Price Trend
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


def calculate_volume_sma(price_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculates rolling average volume.
    """
    _validate_required_columns(price_df, {"volume"})

    price_df = price_df.copy()

    price_df[f"volume_sma_{period}"] = (
        price_df["volume"]
        .rolling(window=period, min_periods=period)
        .mean()
    )

    return price_df


def calculate_volume_ratio(price_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculates current volume relative to rolling average volume.

    Example:
    volume_ratio_20 = 2.0 means today's volume is twice the 20-period average.
    """
    _validate_required_columns(price_df, {"volume"})

    price_df = price_df.copy()

    volume_sma_column = f"volume_sma_{period}"

    if volume_sma_column not in price_df.columns:
        price_df = calculate_volume_sma(price_df, period)

    volume_sma_safe = price_df[volume_sma_column].replace(0, np.nan)

    price_df[f"volume_ratio_{period}"] = price_df["volume"] / volume_sma_safe

    return price_df


def calculate_obv(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates On-Balance Volume.

    OBV adds volume on up days and subtracts volume on down days.
    """
    _validate_required_columns(price_df, {"close", "volume"})

    price_df = price_df.copy()

    close_change = price_df["close"].diff()

    direction = np.select(
        [
            close_change > 0,
            close_change < 0,
        ],
        [
            1,
            -1,
        ],
        default=0,
    )

    price_df["obv"] = (direction * price_df["volume"]).cumsum()

    return price_df


def calculate_obv_features(price_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Adds ML-friendly OBV features.

    Raw OBV is cumulative and scale-dependent, so slope/change is often more useful.
    """
    _validate_required_columns(price_df, {"obv", "volume"})

    price_df = price_df.copy()

    average_volume = price_df["volume"].rolling(
        window=period,
        min_periods=period,
    ).mean()

    average_volume_safe = average_volume.replace(0, np.nan)

    price_df[f"obv_change_{period}"] = price_df["obv"] - price_df["obv"].shift(period)

    price_df[f"obv_change_vs_avg_volume_{period}"] = (
        price_df[f"obv_change_{period}"] / average_volume_safe
    )

    return price_df


def calculate_vwap(price_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculates rolling VWAP approximation using daily candles.

    Note:
    This is not true intraday VWAP. True VWAP requires intraday trades or bars.
    With daily OHLCV data, this is an approximation using typical price.
    """
    _validate_required_columns(price_df, {"high", "low", "close", "volume"})

    price_df = price_df.copy()

    typical_price = (price_df["high"] + price_df["low"] + price_df["close"]) / 3
    price_volume = typical_price * price_df["volume"]

    rolling_volume = price_df["volume"].rolling(
        window=period,
        min_periods=period,
    ).sum()

    rolling_volume_safe = rolling_volume.replace(0, np.nan)

    price_df[f"vwap_{period}"] = (
        price_volume.rolling(window=period, min_periods=period).sum()
        / rolling_volume_safe
    )

    price_df[f"close_vs_vwap_{period}"] = (
        price_df["close"] / price_df[f"vwap_{period}"].replace(0, np.nan) - 1
    )

    return price_df


def calculate_money_flow_index(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates Money Flow Index.

    MFI is similar to RSI but volume-weighted.
    Values usually range between 0 and 100.
    """
    _validate_required_columns(price_df, {"high", "low", "close", "volume"})

    price_df = price_df.copy()

    typical_price = (price_df["high"] + price_df["low"] + price_df["close"]) / 3
    raw_money_flow = typical_price * price_df["volume"]

    typical_price_change = typical_price.diff()

    positive_money_flow = raw_money_flow.where(typical_price_change > 0, 0)
    negative_money_flow = raw_money_flow.where(typical_price_change < 0, 0)

    positive_sum = positive_money_flow.rolling(
        window=period,
        min_periods=period,
    ).sum()

    negative_sum = negative_money_flow.rolling(
        window=period,
        min_periods=period,
    ).sum()

    negative_sum_safe = negative_sum.replace(0, np.nan)

    money_flow_ratio = positive_sum / negative_sum_safe

    price_df[f"mfi_{period}"] = 100 - (100 / (1 + money_flow_ratio))

    price_df.loc[
        (negative_sum == 0) & (positive_sum > 0),
        f"mfi_{period}",
    ] = 100

    price_df.loc[
        (positive_sum == 0) & (negative_sum > 0),
        f"mfi_{period}",
    ] = 0

    return price_df


def calculate_accumulation_distribution_line(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Accumulation/Distribution Line.

    ADL attempts to measure whether volume is flowing into or out of the asset.
    """
    _validate_required_columns(price_df, {"high", "low", "close", "volume"})

    price_df = price_df.copy()

    high_low_range = (price_df["high"] - price_df["low"]).replace(0, np.nan)

    money_flow_multiplier = (
        ((price_df["close"] - price_df["low"]) - (price_df["high"] - price_df["close"]))
        / high_low_range
    )

    money_flow_volume = money_flow_multiplier.fillna(0) * price_df["volume"]

    price_df["adl"] = money_flow_volume.cumsum()

    return price_df


def calculate_adl_features(price_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Adds ML-friendly ADL features.
    """
    _validate_required_columns(price_df, {"adl", "volume"})

    price_df = price_df.copy()

    average_volume = price_df["volume"].rolling(
        window=period,
        min_periods=period,
    ).mean()

    average_volume_safe = average_volume.replace(0, np.nan)

    price_df[f"adl_change_{period}"] = price_df["adl"] - price_df["adl"].shift(period)

    price_df[f"adl_change_vs_avg_volume_{period}"] = (
        price_df[f"adl_change_{period}"] / average_volume_safe
    )

    return price_df


def calculate_chaikin_money_flow(price_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculates Chaikin Money Flow.

    CMF usually ranges between -1 and 1.
    """
    _validate_required_columns(price_df, {"high", "low", "close", "volume"})

    price_df = price_df.copy()

    high_low_range = (price_df["high"] - price_df["low"]).replace(0, np.nan)

    money_flow_multiplier = (
        ((price_df["close"] - price_df["low"]) - (price_df["high"] - price_df["close"]))
        / high_low_range
    )

    money_flow_volume = money_flow_multiplier.fillna(0) * price_df["volume"]

    volume_sum = price_df["volume"].rolling(
        window=period,
        min_periods=period,
    ).sum()

    volume_sum_safe = volume_sum.replace(0, np.nan)

    price_df[f"cmf_{period}"] = (
        money_flow_volume.rolling(window=period, min_periods=period).sum()
        / volume_sum_safe
    )

    return price_df


def calculate_volume_price_trend(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Volume Price Trend.

    VPT adds volume weighted by percentage price change.
    """
    _validate_required_columns(price_df, {"close", "volume"})

    price_df = price_df.copy()

    previous_close = price_df["close"].shift(1).replace(0, np.nan)
    close_return = price_df["close"] / previous_close - 1

    price_df["vpt"] = (price_df["volume"] * close_return.fillna(0)).cumsum()

    return price_df


def calculate_vpt_features(price_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Adds ML-friendly VPT features.
    """
    _validate_required_columns(price_df, {"vpt", "volume"})

    price_df = price_df.copy()

    average_volume = price_df["volume"].rolling(
        window=period,
        min_periods=period,
    ).mean()

    average_volume_safe = average_volume.replace(0, np.nan)

    price_df[f"vpt_change_{period}"] = price_df["vpt"] - price_df["vpt"].shift(period)

    price_df[f"vpt_change_vs_avg_volume_{period}"] = (
        price_df[f"vpt_change_{period}"] / average_volume_safe
    )

    return price_df


def calculate_all_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all volume indicators.
    """
    price_df = price_df.copy()

    for period in [14, 20]:
        price_df = calculate_volume_sma(price_df, period)
        price_df = calculate_volume_ratio(price_df, period)
        price_df = calculate_money_flow_index(price_df, period)
        price_df = calculate_chaikin_money_flow(price_df, period)
        price_df = calculate_vwap(price_df, period)

    price_df = calculate_obv(price_df)
    price_df = calculate_accumulation_distribution_line(price_df)
    price_df = calculate_volume_price_trend(price_df)

    for period in [14, 20]:
        price_df = calculate_obv_features(price_df, period)
        price_df = calculate_adl_features(price_df, period)
        price_df = calculate_vpt_features(price_df, period)

    return price_df