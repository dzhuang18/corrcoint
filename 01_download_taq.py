"""Download TAQ millisecond quotes from WRDS and save as Parquet."""

import pandas as pd
import wrds
from config import TICKERS, START_DATE, END_DATE, RAW_DIR, WRDS_USERNAME

VALID_QU_COND = {"", "R", "A"}  # standard quote conditions to keep

def trading_dates(start: str, end: str) -> list[str]:
    dates = pd.bdate_range(start, end)
    return [d.strftime("%Y%m%d") for d in dates]


def fetch_ticker(db: wrds.Connection, ticker: str, dates: list[str]) -> pd.DataFrame:
    frames = []
    for date in dates:
        table = f"taqmsec.cqm_{date}"
        query = (
            f"SELECT time_m, sym_root, bid, ask, qu_cond "
            f"FROM {table} "
            f"WHERE sym_root = '{ticker}'"
        )
        try:
            df = db.raw_sql(query)
        except Exception as exc:
            print(f"  [{ticker}] {date}: skipped ({exc})")
            continue
        # drop non-standard quote conditions
        if "qu_cond" in df.columns:
            df = df[df["qu_cond"].isin(VALID_QU_COND)]
        df["date"] = pd.to_datetime(date, format="%Y%m%d")
        frames.append(df)
        print(f"  [{ticker}] {date}: {len(df):,} rows")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    dates = trading_dates(START_DATE, END_DATE)
    print(f"Trading dates: {dates}")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    for ticker in TICKERS:
        print(f"\nDownloading {ticker}...")
        df = fetch_ticker(db, ticker, dates)
        if df.empty:
            print(f"  No data for {ticker}")
            continue
        out = RAW_DIR / f"{ticker}_quotes.parquet"
        df.to_parquet(out, index=False)
        print(f"  Saved {len(df):,} rows → {out}")
    db.close()


if __name__ == "__main__":
    main()
