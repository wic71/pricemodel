"""
Feature sets for model training.

This module defines which already-created indicator columns should be used
as model features.

The indicator modules may create many raw/debug/chart columns.
Only columns listed here should be sent into the model.
"""

BASE_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "log_return_1d",
]


TREND_FEATURES = [
    "close_vs_sma_20",
    "close_vs_sma_50",
    "close_vs_sma_200",

    "close_vs_ema_20",
    "close_vs_ema_50",
    "close_vs_ema_200",

    "sma_20_vs_sma_50",
    "sma_50_vs_sma_200",

    "ema_12_vs_ema_26",

    "sma_20_slope_5",
    "sma_50_slope_5",
    "ema_20_slope_5",
    "ema_50_slope_5",

    "macd_line_pct",
    "macd_signal_pct",
    "macd_histogram_pct",
    "macd_above_signal",

    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "di_spread_14",
    "di_spread_pct_14",
    "plus_di_above_minus_di_14",
]


MOMENTUM_FEATURES = [
    "rsi_14_scaled",

    "roc_14",
    "momentum_pct_14",

    "stochastic_k_14",
    "stochastic_d_14_3",

    "williams_r_14",
    "cci_14",
]


VOLATILITY_FEATURES = [
    "true_range_pct",

    "atr_pct_14",
    "atr_pct_20",

    "rolling_volatility_14",
    "rolling_volatility_20",

    "bollinger_width_20",
    "bollinger_percent_b_20",
    "close_vs_bollinger_middle_20",
    "close_above_bollinger_upper_20",
    "close_below_bollinger_lower_20",

    "keltner_width_pct_20",
    "close_position_in_keltner_20",

    "donchian_width_pct_20",
    "close_position_in_donchian_20",
]


VOLUME_FEATURES = [
    "volume_ratio_20",

    "mfi_14_scaled",
    "cmf_20",

    "close_vs_vwap_20",

    "obv_change_vs_avg_volume_20",
    "adl_change_vs_avg_volume_20",
    "vpt_change_vs_avg_volume_20",
]


PRICE_LEVEL_FEATURES = [
    "close_vs_rolling_high_20",
    "close_vs_rolling_low_20",
    "close_position_in_rolling_range_20",
    "rolling_range_width_pct_20",
    "close_breakout_above_rolling_high_20",
    "close_breakdown_below_rolling_low_20",

    "close_vs_rolling_high_50",
    "close_vs_rolling_low_50",
    "close_position_in_rolling_range_50",
    "rolling_range_width_pct_50",

    "close_vs_pivot_point",
    "close_vs_pivot_support_1",
    "close_vs_pivot_resistance_1",
    "close_vs_pivot_support_2",
    "close_vs_pivot_resistance_2",
    "pivot_range_pct",

    "close_above_pivot_point",
    "close_above_pivot_resistance_1",
    "close_below_pivot_support_1",
]


CANDLESTICK_FEATURES = [
    "candle_range_pct",
    "candle_body_pct",
    "candle_body_abs_pct",

    "body_pct_of_range",
    "upper_wick_pct_of_range",
    "lower_wick_pct_of_range",

    "upper_wick_pct",
    "lower_wick_pct",

    "candle_direction",

    "doji",
    "hammer",
    "shooting_star",
    "engulfing_bullish",
    "engulfing_bearish",
    "inside_bar",
    "outside_bar",

    "gap_pct",
    "gap_up",
    "gap_down",

    "long_body_20",
    "long_upper_wick",
    "long_lower_wick",
]


DEFAULT_FEATURES = (
    BASE_FEATURES
    + TREND_FEATURES
    + MOMENTUM_FEATURES
    + VOLATILITY_FEATURES
    + VOLUME_FEATURES
    + PRICE_LEVEL_FEATURES
    + CANDLESTICK_FEATURES
)

DERIVED_FEATURES = (
    BASE_FEATURES
    + [
        "rsi_14_scaled",
        "mfi_14_scaled",
    ]
)

INDICATOR_FEATURES = [
    feature
    for feature in DEFAULT_FEATURES
    if feature not in DERIVED_FEATURES
]