from datetime import date

import pandas as pd
import pytest

from data_providers import (
    PriceDataRequest,
    YFinanceProvider,
    get_market_data_provider,
)


def test_get_market_data_provider_returns_yfinance_provider():
    provider = get_market_data_provider("yfinance")

    assert isinstance(provider, YFinanceProvider)


def test_get_market_data_provider_rejects_unknown_provider():
    with pytest.raises(ValueError):
        get_market_data_provider("unknown_provider")


def test_yfinance_provider_normalizes_price_history(monkeypatch):
    def fake_download(*args, **kwargs):
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 100.0],
                "Close": [104.0, 105.0],
                "Adj Close": [103.5, 104.5],
                "Volume": [1000000, 1200000],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )

        df.index.name = "Date"

        return df

    monkeypatch.setattr("data_providers.yf.download", fake_download)

    provider = YFinanceProvider()

    request = PriceDataRequest(
        ticker="VOLV-B.ST",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
        interval="1d",
    )

    price_df = provider.get_price_history(request)

    assert list(price_df.columns) == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]

    assert len(price_df) == 2
    assert price_df.loc[0, "ticker"] == "VOLV-B.ST"
    assert price_df.loc[0, "open"] == 100.0
    assert price_df.loc[1, "close"] == 105.0
    assert price_df.loc[1, "volume"] == 1200000


def test_yfinance_provider_handles_empty_response(monkeypatch):
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("data_providers.yf.download", fake_download)

    provider = YFinanceProvider()

    request = PriceDataRequest(
        ticker="UNKNOWN",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
        interval="1d",
    )

    price_df = provider.get_price_history(request)

    assert price_df.empty
    assert list(price_df.columns) == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]