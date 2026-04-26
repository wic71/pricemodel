import numpy as np
import pandas as pd

def calculate_previous_session_eod_context(price_df):
    """
    Builds features known at the end of the previous session.
    Useful for next-day prediction before today's open exists.
    """
    previous_open = price_df["open"].shift(1)
    previous_high = price_df["high"].shift(1)
    previous_low = price_df["low"].shift(1)
    previous_close = price_df["close"].shift(1)

    previous_range = previous_high - previous_low
    previous_range_safe = previous_range.replace(0, np.nan)

    price_df["prev_body_pct"] = (previous_close - previous_open) / previous_open
    price_df["prev_range_pct"] = previous_range / previous_close
    price_df["prev_close_position_in_prev_range"] = (
        previous_close - previous_low
    ) / previous_range_safe
    price_df["prev_upper_wick_pct_of_range"] = (
        previous_high - np.maximum(previous_open, previous_close)
    ) / previous_range_safe
    price_df["prev_lower_wick_pct_of_range"] = (
        np.minimum(previous_open, previous_close) - previous_low
    ) / previous_range_safe
    price_df["prev_day_direction"] = np.sign(previous_close - previous_open)

    return price_df

def calculate_previous_session_context(price_df: pd.DataFrame, keep_raw_columns: bool = False) -> pd.DataFrame:
    """
    Adds features describing the previous trading session in relation to today's open.

    Required columns:
    - open
    - high
    - low
    - close

    The function only uses previous-day values and today's open.
    It should therefore be safe from look-ahead leakage when used for predictions
    made at or near the market open.
    """
    required_columns = {"open", "high", "low", "close"}
    missing_columns = required_columns - set(price_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    price_df = price_df.copy()

    previous_open = price_df["open"].shift(1)
    previous_high = price_df["high"].shift(1)
    previous_low = price_df["low"].shift(1)
    previous_close = price_df["close"].shift(1)

    previous_range = previous_high - previous_low
    previous_body = previous_close - previous_open

    # Avoid division by zero.
    previous_range_safe = previous_range.replace(0, np.nan)
    previous_close_safe = previous_close.replace(0, np.nan)

    # Basic previous-day structure.
    if keep_raw_columns:
        price_df["prev_open"] = previous_open
        price_df["prev_high"] = previous_high
        price_df["prev_low"] = previous_low
        price_df["prev_close"] = previous_close

    price_df["prev_range_pct"] = previous_range / previous_close_safe
    price_df["prev_body_pct"] = previous_body / previous_open.replace(0, np.nan)

    # Today's open relative to previous close: classic overnight gap.
    price_df["open_vs_prev_close"] = price_df["open"] / previous_close_safe - 1

    # Today's open relative to previous high/low.
    price_df["open_vs_prev_high"] = price_df["open"] / previous_high.replace(0, np.nan) - 1
    price_df["open_vs_prev_low"] = price_df["open"] / previous_low.replace(0, np.nan) - 1
    price_df["open_vs_prev_open"] = price_df["open"] / previous_open.replace(0, np.nan) - 1

    # Where today's open sits inside yesterday's range.
    # 0 = at previous low, 1 = at previous high.
    price_df["open_position_in_prev_range"] = (
        price_df["open"] - previous_low
    ) / previous_range_safe

    # Gap outside previous range.
    price_df["gap_above_prev_high"] = (price_df["open"] > previous_high).astype(int)
    price_df["gap_below_prev_low"] = (price_df["open"] < previous_low).astype(int)
    price_df["open_inside_prev_range"] = (
        (price_df["open"] >= previous_low) & (price_df["open"] <= previous_high)
    ).astype(int)

    # Size of gap outside previous range.
    price_df["gap_above_prev_high_pct"] = np.where(
        price_df["open"] > previous_high,
        price_df["open"] / previous_high.replace(0, np.nan) - 1,
        0,
    )

    price_df["gap_below_prev_low_pct"] = np.where(
        price_df["open"] < previous_low,
        price_df["open"] / previous_low.replace(0, np.nan) - 1,
        0,
    )

    # Previous candle shape.
    price_df["prev_upper_wick_pct_of_range"] = (
        previous_high - np.maximum(previous_open, previous_close)
    ) / previous_range_safe

    price_df["prev_lower_wick_pct_of_range"] = (
        np.minimum(previous_open, previous_close) - previous_low
    ) / previous_range_safe

    price_df["prev_body_pct_of_range"] = previous_body.abs() / previous_range_safe

    # Direction of previous day.
    price_df["prev_day_direction"] = np.sign(previous_close - previous_open)

    return price_df

import numpy as np
import pandas as pd


def build_previous_intraday_profile(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds one previous-session intraday profile row per ticker and date.

    The output is intended to be shifted/merged onto the following daily session.

    Required columns:
    - ticker
    - datetime
    - open
    - high
    - low
    - close
    - volume
    """
    required_columns = {"ticker", "datetime", "open", "high", "low", "close", "volume"}
    missing_columns = required_columns - set(intraday_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    intraday_df = intraday_df.copy()
    intraday_df["datetime"] = pd.to_datetime(intraday_df["datetime"])
    intraday_df["date"] = intraday_df["datetime"].dt.date

    rows = []

    for (ticker, session_date), group in intraday_df.groupby(["ticker", "date"]):
        group = group.sort_values("datetime").copy()

        session_open = group["open"].iloc[0]
        session_close = group["close"].iloc[-1]
        session_high = group["high"].max()
        session_low = group["low"].min()
        session_volume = group["volume"].sum()

        high_row = group.loc[group["high"].idxmax()]
        low_row = group.loc[group["low"].idxmin()]

        high_time = high_row["datetime"]
        low_time = low_row["datetime"]

        session_start = group["datetime"].iloc[0]
        session_end = group["datetime"].iloc[-1]
        session_seconds = max((session_end - session_start).total_seconds(), 1)

        high_time_fraction = (high_time - session_start).total_seconds() / session_seconds
        low_time_fraction = (low_time - session_start).total_seconds() / session_seconds

        # Circular time encoding.
        high_angle = 2 * np.pi * high_time_fraction
        low_angle = 2 * np.pi * low_time_fraction

        average_bar_volume = group["volume"].mean()

        high_volume_ratio = high_row["volume"] / average_bar_volume if average_bar_volume else np.nan
        low_volume_ratio = low_row["volume"] / average_bar_volume if average_bar_volume else np.nan

        typical_price = (group["high"] + group["low"] + group["close"]) / 3
        vwap = (typical_price * group["volume"]).sum() / session_volume if session_volume else np.nan

        rows.append(
            {
                "ticker": ticker,
                "date": session_date,

                "prev_intraday_range_pct": (
                    (session_high - session_low) / session_open
                    if session_open else np.nan
                ),
                "prev_intraday_body_pct": (
                    (session_close - session_open) / session_open
                    if session_open else np.nan
                ),

                "prev_high_time_sin": np.sin(high_angle),
                "prev_high_time_cos": np.cos(high_angle),
                "prev_low_time_sin": np.sin(low_angle),
                "prev_low_time_cos": np.cos(low_angle),

                "prev_high_time_fraction": high_time_fraction,
                "prev_low_time_fraction": low_time_fraction,
                "prev_high_before_low": int(high_time < low_time),
                "prev_minutes_between_high_low": abs(
                    (high_time - low_time).total_seconds()
                ) / 60,

                "prev_high_volume_ratio": high_volume_ratio,
                "prev_low_volume_ratio": low_volume_ratio,

                "prev_high_was_low_volume_spike": int(high_volume_ratio < 0.75),
                "prev_low_was_low_volume_spike": int(low_volume_ratio < 0.75),

                "prev_close_vs_vwap": (
                    session_close / vwap - 1
                    if vwap and not np.isnan(vwap) else np.nan
                ),

                "prev_total_volume": session_volume,
            }
        )

    profile_df = pd.DataFrame(rows)

    return profile_df