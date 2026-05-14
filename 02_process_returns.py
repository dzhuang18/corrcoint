"""Snap raw quotes to 1-min grid, compute 1-min and 20-min overlapping returns."""

import numpy as np
import pandas as pd
from config import (
    TICKERS, RAW_DIR, PROCESSED_DIR,
    SNAP_INTERVAL_MIN, RETURN_WINDOW_20, OVERLAP_STEP_20,
)

MARKET_OPEN = pd.Timedelta(hours=9, minutes=30)
MARKET_CLOSE = pd.Timedelta(hours=16, minutes=0)


def load_raw(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}_quotes.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # time_m is a time string "HH:MM:SS.ffffff" — combine with date to get full timestamp
    df["time_m"] = df["date"] + pd.to_timedelta(df["time_m"].astype(str))
    df["midpoint"] = (df["bid"] + df["ask"]) / 2
    df = df[df["midpoint"] > 0].copy()
    return df


def snap_to_grid(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a DatetimeIndex-ed Series of snapped midpoints for one ticker."""
    records = []
    for date, day_df in df.groupby("date"):
        # build 1-min grid for this day
        grid = pd.date_range(
            date + MARKET_OPEN,
            date + MARKET_CLOSE,
            freq=f"{SNAP_INTERVAL_MIN}min",
        )
        day_df = day_df.set_index("time_m").sort_index()
        # keep only intraday quotes
        day_df = day_df.loc[
            (date + MARKET_OPEN) : (date + MARKET_CLOSE)
        ]
        # last valid midpoint in each minute bucket
        snapped = (
            day_df["midpoint"]
            .resample(f"{SNAP_INTERVAL_MIN}min", closed="right", label="right")
            .last()
            .reindex(grid)
        )
        # forward-fill within day only
        snapped = snapped.ffill()
        records.append(snapped)
    return pd.concat(records).rename(ticker)


def compute_1min_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns from consecutive 1-min prices; first obs of each day is overnight."""
    rets = np.log(prices / prices.shift(1))
    # zero out cross-day gaps that aren't valid overnight (multi-day weekend gaps)
    # overnight return is valid only when the prior row is 16:00 of the previous biz day
    # all other NaN rows (from missing snaps) stay NaN — drop them downstream
    return rets


def compute_20min_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Overlapping 20-min log returns with step=OVERLAP_STEP_20 minutes."""
    step = OVERLAP_STEP_20
    window = RETURN_WINDOW_20
    # align to step-minute boundary
    rets = np.log(prices / prices.shift(window))
    # keep only timestamps that fall on the step grid
    mask = (
        (prices.index.hour * 60 + prices.index.minute - 9 * 60 - 30) % step == 0
    )
    rets = rets.loc[mask]
    # need at least 'window' minutes of prior data
    rets = rets.dropna(how="all")
    return rets


def main():
    price_frames = {}
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        raw = load_raw(ticker)
        snapped = snap_to_grid(raw, ticker)
        price_frames[ticker] = snapped

    prices = pd.DataFrame(price_frames)
    prices.index.name = "timestamp"

    # 1-min returns
    ret1 = compute_1min_returns(prices)
    ret1 = ret1.dropna(how="all")

    # 20-min overlapping returns
    ret20 = compute_20min_returns(prices)

    prices.to_parquet(PROCESSED_DIR / "prices_1min.parquet")
    ret1.to_parquet(PROCESSED_DIR / "returns_1min.parquet")
    ret20.to_parquet(PROCESSED_DIR / "returns_20min_overlap5.parquet")

    print(f"Prices:        {prices.shape}  -> prices_1min.parquet")
    print(f"1-min returns: {ret1.shape}  -> returns_1min.parquet")
    print(f"20-min returns:{ret20.shape}  -> returns_20min_overlap5.parquet")
    print(f"NaN counts in 1-min returns:\n{ret1.isna().sum()}")


if __name__ == "__main__":
    main()
