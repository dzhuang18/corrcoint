"""Engle-Granger and Johansen cointegration tests -- full-span and daily."""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from config import TICKERS, PROCESSED_DIR

MAR, HLT = TICKERS


def eg_test(y: pd.Series, x: pd.Series) -> dict:
    mask = y.notna() & x.notna()
    stat, pval, crit = coint(y[mask].astype(float), x[mask].astype(float))
    return {"stat": stat, "pval": pval, "crit_1pct": crit[0], "crit_5pct": crit[1], "crit_10pct": crit[2]}


def johansen_test(df: pd.DataFrame) -> dict:
    clean = df.dropna().astype(float)
    res = coint_johansen(clean, det_order=0, k_ar_diff=1)
    return {
        "trace_stat": res.lr1.tolist(),
        "trace_crit_90": res.cvt[:, 0].tolist(),
        "trace_crit_95": res.cvt[:, 1].tolist(),
        "trace_crit_99": res.cvt[:, 2].tolist(),
        "max_eig_stat": res.lr2.tolist(),
    }


def run_full_span(prices_1min: pd.DataFrame, prices_20min: pd.DataFrame) -> dict:
    results = {}
    for label, prices in [("1min", prices_1min), ("20min", prices_20min)]:
        p = prices[[MAR, HLT]].astype(float)
        p = p[(p > 0).all(axis=1)]
        log_p = np.log(p).dropna()
        results[f"eg_{label}"] = eg_test(log_p[MAR], log_p[HLT])
        results[f"johansen_{label}"] = johansen_test(log_p)
        eg = results[f"eg_{label}"]
        print(f"  EG [{label}]: stat={eg['stat']:.4f} pval={eg['pval']:.4f}")
        jo = results[f"johansen_{label}"]
        print(f"  Johansen [{label}]: trace={jo['trace_stat']}")
    return results


def run_daily(prices_1min: pd.DataFrame, prices_20min: pd.DataFrame) -> list[dict]:
    p1 = prices_1min[[MAR, HLT]].astype(float)
    log1 = np.log(p1[(p1 > 0).all(axis=1)].dropna())
    p20 = prices_20min[[MAR, HLT]].astype(float)
    log20 = np.log(p20[(p20 > 0).all(axis=1)].dropna())

    daily_results = []
    dates = log1.index.normalize().unique()
    for date in sorted(dates):
        day1 = log1.loc[log1.index.normalize() == date].dropna()
        day20 = log20.loc[log20.index.normalize() == date].dropna()
        row = {"date": date}

        if len(day1) >= 10:
            eg1 = eg_test(day1[MAR], day1[HLT])
            row["eg_1min_stat"] = eg1["stat"]
            row["eg_1min_pval"] = eg1["pval"]
            jo1 = johansen_test(day1)
            row["johansen_1min_trace"] = jo1["trace_stat"][0]
        else:
            row.update({"eg_1min_stat": np.nan, "eg_1min_pval": np.nan, "johansen_1min_trace": np.nan})

        if len(day20) >= 5:
            eg20 = eg_test(day20[MAR], day20[HLT])
            row["eg_20min_stat"] = eg20["stat"]
            row["eg_20min_pval"] = eg20["pval"]
            jo20 = johansen_test(day20)
            row["johansen_20min_trace"] = jo20["trace_stat"][0]
        else:
            row.update({"eg_20min_stat": np.nan, "eg_20min_pval": np.nan, "johansen_20min_trace": np.nan})

        daily_results.append(row)
        print(f"  {date.date()} EG 1-min pval={row.get('eg_1min_pval', float('nan')):.4f}")

    return daily_results


def main():
    prices_1min = pd.read_parquet(PROCESSED_DIR / "prices_1min.parquet")
    # 20-min prices: take prices at the 20-min return timestamps
    ret20 = pd.read_parquet(PROCESSED_DIR / "returns_20min_overlap5.parquet")
    prices_20min = prices_1min.reindex(ret20.index)

    print("Full-span cointegration:")
    full = run_full_span(prices_1min, prices_20min)

    print("\nDaily cointegration:")
    daily = run_daily(prices_1min, prices_20min)

    out = {
        "full_span": full,
        "daily": pd.DataFrame(daily).set_index("date"),
    }
    path = PROCESSED_DIR / "cointegration_results.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved → {path}")
    print(out["daily"])


if __name__ == "__main__":
    main()
