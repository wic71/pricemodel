"""
Momentum indicators.

Includes:
- RSI
- ROC
- Momentum
- Stochastic oscillator
- Williams %R
- CCI
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


def calculate_rsi(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates RSI using Wilder's smoothing method.

    RSI is scaled between 0 and 100.
    """
    _validate_required_columns(price_df, {"close"})

    price_df = price_df.copy()

    delta = price_df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    avg_loss_safe = avg_loss.replace(0, np.nan)

    rs = avg_gain / avg_loss_safe

    price_df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # If there were gains and no losses, RSI should be 100.
    price_df.loc[(avg_loss == 0) & (avg_gain > 0), f"rsi_{period}"] = 100

    # If there were losses and no gains, RSI should be 0.
    price_df.loc[(avg_gain == 0) & (avg_loss > 0), f"rsi_{period}"] = 0

    return price_df


def calculate_roc(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates rate of change over N periods.
    """
    _validate_required_columns(price_df, {"close"})

    price_df = price_df.copy()

    previous_close = price_df["close"].shift(period).replace(0, np.nan)

    price_df[f"roc_{period}"] = price_df["close"] / previous_close - 1

    return price_df


def calculate_momentum(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates absolute and percentage momentum over N periods.

    Absolute momentum can be useful for charting/debugging.
    Percentage momentum is usually better for ML features.
    """
    _validate_required_columns(price_df, {"close"})

    price_df = price_df.copy()

    previous_close = price_df["close"].shift(period).replace(0, np.nan)

    price_df[f"momentum_abs_{period}"] = price_df["close"] - price_df["close"].shift(period)
    price_df[f"momentum_pct_{period}"] = price_df["close"] / previous_close - 1

    return price_df


def calculate_stochastic_oscillator(
    price_df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Calculates stochastic oscillator %K and %D.

    %K = position of close within the recent high/low range.
    %D = moving average of %K.
    """
    _validate_required_columns(price_df, {"high", "low", "close"})

    price_df = price_df.copy()

    lowest_low = price_df["low"].rolling(window=k_period, min_periods=k_period).min()
    highest_high = price_df["high"].rolling(window=k_period, min_periods=k_period).max()

    range_width = (highest_high - lowest_low).replace(0, np.nan)

    k_column = f"stochastic_k_{k_period}"
    d_column = f"stochastic_d_{k_period}_{d_period}"

    price_df[k_column] = 100 * (price_df["close"] - lowest_low) / range_width
    price_df[d_column] = price_df[k_column].rolling(
        window=d_period,
        min_periods=d_period,
    ).mean()

    return price_df


def calculate_williams_r(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates Williams %R.

    Values typically range from -100 to 0.
    """
    _validate_required_columns(price_df, {"high", "low", "close"})

    price_df = price_df.copy()

    highest_high = price_df["high"].rolling(window=period, min_periods=period).max()
    lowest_low = price_df["low"].rolling(window=period, min_periods=period).min()

    range_width = (highest_high - lowest_low).replace(0, np.nan)

    price_df[f"williams_r_{period}"] = (
        -100 * (highest_high - price_df["close"]) / range_width
    )

    return price_df


def calculate_cci(price_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculates Commodity Channel Index.

    CCI measures how far the typical price is from its moving average,
    normalized by mean absolute deviation.
    """
    _validate_required_columns(price_df, {"high", "low", "close"})

    price_df = price_df.copy()

    typical_price = (price_df["high"] + price_df["low"] + price_df["close"]) / 3
    sma_typical_price = typical_price.rolling(
        window=period,
        min_periods=period,
    ).mean()

    mean_deviation = typical_price.rolling(
        window=period,
        min_periods=period,
    ).apply(
        lambda values: np.mean(np.abs(values - np.mean(values))),
        raw=True,
    )

    mean_deviation_safe = mean_deviation.replace(0, np.nan)

    price_df[f"cci_{period}"] = (
        (typical_price - sma_typical_price) / (0.015 * mean_deviation_safe)
    )

    return price_df


def calculate_all_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all momentum indicators.
    """
    price_df = price_df.copy()

    for period in [14]:
        price_df = calculate_rsi(price_df, period=period)
        price_df = calculate_roc(price_df, period=period)
        price_df = calculate_momentum(price_df, period=period)
        price_df = calculate_stochastic_oscillator(
            price_df,
            k_period=period,
            d_period=3,
        )
        price_df = calculate_williams_r(price_df, period=period)
        price_df = calculate_cci(price_df, period=period)

    return price_df