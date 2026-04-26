# data_providers.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

import pandas as pd
import yfinance as yf


Interval = Literal["1d", "1wk", "1mo"]


@dataclass
class PriceDataRequest:
    ticker: str
    start_date: date
    end_date: date
    interval: Interval = "1d"


class MarketDataProvider(ABC):
    """
    Abstract base class for all market data providers.

    Every provider should implement the same public methods so the rest
    of the application can use market data without knowing the source.
    """

    @abstractmethod
    def get_price_history(self, request: PriceDataRequest) -> pd.DataFrame:
        """
        Return historical price data for one ticker.

        Expected columns:
        - date
        - ticker
        - open
        - high
        - low
        - close
        - adjusted_close
        - volume
        """
        raise NotImplementedError


class YFinanceProvider(MarketDataProvider):
    """
    Market data provider using yfinance.
    """

    def get_price_history(self, request: PriceDataRequest) -> pd.DataFrame:
        raw_df = yf.download(
            tickers=request.ticker,
            start=request.start_date,
            end=request.end_date,
            interval=request.interval,
            auto_adjust=False,
            progress=False,
        )

        return self._normalize_price_history(raw_df, request.ticker)

    def _normalize_price_history(self, raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Normalize yfinance output into the application's standard format.
        """

        expected_columns = [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ]

        if raw_df.empty:
            return pd.DataFrame(columns=expected_columns)

        df = raw_df.copy()

        # yfinance may return MultiIndex columns, especially with newer versions.
        # Example: ("Close", "VOLV-B.ST"). We only want the first level.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        df = df.rename(
            columns={
                "Date": "date",
                "Datetime": "date",
                "index": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
            }
        )

        df["ticker"] = ticker

        return df[expected_columns]


def get_market_data_provider(provider_name: str = "yfinance") -> MarketDataProvider:
    """
    Factory function for creating market data providers.
    """

    provider_name = provider_name.lower().strip()

    if provider_name == "yfinance":
        return YFinanceProvider()

    raise ValueError(f"Unsupported market data provider: {provider_name}")