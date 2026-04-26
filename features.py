import numpy as np
import pandas as pd

def build_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for a single ticker's price history DataFrame.

    Expected input columns:
    - date
    - ticker
    - open
    - high
    - low
    - close
    - adjusted_close
    - volume

    Output columns:
    - date
    - ticker
    - open
    - high
    - low
    - close
    - adjusted_close
    - volume
    - daily_return
    - daily_return_7d_avg
    """

    df = df.copy()
    df["daily_return"] = df["close"].pct_change()
    df["daily_return_7d_avg"] = df["daily_return"].rolling(window=7).mean()

    return df