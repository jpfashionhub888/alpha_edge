import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import timedelta

DATA_DIR = Path("data/ohlcv")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_ticker(ticker: str):
    path = DATA_DIR / f"{ticker}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def backfill_ticker(ticker: str, years: int = 10):
    df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        return None

    df = df.reset_index()
    df.to_parquet(DATA_DIR / f"{ticker}.parquet", index=False)
    return df


def update_ticker(ticker: str):
    existing = load_ticker(ticker)

    if existing is None:
        return backfill_ticker(ticker)

    last_date = pd.to_datetime(existing["Date"]).max()

    new_data = yf.download(
        ticker,
        start=(last_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True
    )

    if new_data.empty:
        return existing

    new_data = new_data.reset_index()

    updated = pd.concat([existing, new_data], ignore_index=True)
    updated = updated.drop_duplicates(subset="Date")
    updated = updated.sort_values("Date")

    updated.to_parquet(DATA_DIR / f"{ticker}.parquet", index=False)

    return updated