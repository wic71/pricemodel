from datetime import date

from data_providers import PriceDataRequest, get_market_data_provider


def test_yfinance_volvo_returns_valid_price_dataset_with_five_rows():
    provider = get_market_data_provider("yfinance")

    request = PriceDataRequest(
        ticker="VOLV-B.ST",
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        interval="1d",
    )

    price_df = provider.get_price_history(request)
    first_five_rows = price_df.head(5)

    assert len(first_five_rows) == 5

    assert list(first_five_rows.columns) == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]

    assert first_five_rows["ticker"].eq("VOLV-B.ST").all()
    assert first_five_rows["close"].notna().all()
    assert first_five_rows["volume"].notna().all()