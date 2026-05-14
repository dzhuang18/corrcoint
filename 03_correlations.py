"""Static and rolling OLS regressions between MAR and HLT returns (HC3 SEs)."""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import TICKERS, PROCESSED_DIR, ROLLING_UPDATE_MINS

DIRECTIONS = [
    (TICKERS[0], TICKERS[1]),   # MAR ~ HLT
    (TICKERS[1], TICKERS[0]),   # HLT ~ MAR
]


def ols_hc3(y: pd.Series, x: pd.Series) -> dict:
    mask = y.notna() & x.notna()
    y_ = y[mask].astype(float)
    x_ = x[mask].astype(float)
    X = sm.add_constant(x_)
    res = sm.OLS(y_, X).fit(cov_type="HC3")
    return {
        "alpha": res.params.iloc[0],
        "beta": res.params.iloc[1],
        "se_beta": res.bse.iloc[1],
        "r2": res.rsquared,
        "nobs": int(res.nobs),
        "t_start": y_.index[0] if len(y_) else pd.NaT,
        "t_end": y_.index[-1] if len(y_) else pd.NaT,
    }


def static_regression(ret: pd.DataFrame, label: str) -> dict:
    results = {}
    for dep, indep in DIRECTIONS:
        key = (dep, indep, label, "static")
        results[key] = ols_hc3(ret[dep], ret[indep])
        print(f"  {dep}~{indep} [{label}] β={results[key]['beta']:.4f} R²={results[key]['r2']:.4f}")
    return results


def rolling_regression(ret: pd.DataFrame, label: str, step_mins: int) -> dict:
    """
    Roll a 1-trading-day window, stepping by step_mins observations.
    Window size = observations per trading day for each return type.
    """
    results = {}
    n = len(ret)
    # approx obs per day: 390 for 1-min, ~75 for 20-min step=5
    mins_per_day = 390 if label == "1min" else (390 // 5)
    window = mins_per_day

    for dep, indep in DIRECTIONS:
        rolls = []
        for start in range(0, n - window + 1, step_mins):
            chunk = ret.iloc[start : start + window]
            res = ols_hc3(chunk[dep], chunk[indep])
            rolls.append(res)
        key = (dep, indep, label, "rolling")
        results[key] = rolls
        if rolls:
            betas = [r["beta"] for r in rolls]
            print(f"  {dep}~{indep} [{label}] rolling: {len(rolls)} windows, β∈[{min(betas):.3f},{max(betas):.3f}]")
    return results


def main():
    ret1 = pd.read_parquet(PROCESSED_DIR / "returns_1min.parquet")
    ret20 = pd.read_parquet(PROCESSED_DIR / "returns_20min_overlap5.parquet")

    all_results = {}

    print("Static regressions — 1-min:")
    all_results.update(static_regression(ret1, "1min"))

    print("Static regressions — 20-min:")
    all_results.update(static_regression(ret20, "20min"))

    print("Rolling regressions — 1-min:")
    all_results.update(rolling_regression(ret1, "1min", step_mins=ROLLING_UPDATE_MINS))

    print("Rolling regressions — 20-min:")
    all_results.update(rolling_regression(ret20, "20min", step_mins=ROLLING_UPDATE_MINS // 5))

    out = PROCESSED_DIR / "regression_results.pkl"
    with open(out, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved {len(all_results)} result entries → {out}")


if __name__ == "__main__":
    main()
