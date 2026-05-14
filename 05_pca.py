"""Per-day PCA: calibrate on first 3 hours, then z-score factor scores post-12:30."""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import pickle
import numpy as np
import pandas as pd
from config import TICKERS, PROCESSED_DIR, PCA_CALIBRATION_HOURS

MAR, HLT = TICKERS
CALIB_END_HOUR = 9 + PCA_CALIBRATION_HOURS   # 12:30 → hour=12, minute=30
CALIB_END_MINUTE = 30


def calib_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    t = idx.hour * 60 + idx.minute
    calib_end = CALIB_END_HOUR * 60 + CALIB_END_MINUTE
    return t <= calib_end


def svd_pca(X: np.ndarray):
    """Return (eigenvalues, eigenvectors) from 2-col return matrix X."""
    X_c = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    eigenvalues = (s ** 2) / (len(X) - 1)
    return eigenvalues, Vt   # Vt rows are eigenvectors


def process_day(day_ret: pd.DataFrame) -> dict | None:
    """PCA for a single trading day."""
    day_ret = day_ret[[MAR, HLT]].astype(float).dropna()
    if len(day_ret) < 30:
        return None

    calib = calib_mask(day_ret.index)
    post = ~calib

    X_calib = day_ret.loc[calib].values
    if X_calib.shape[0] < 10:
        return None

    eigenvalues, Vt = svd_pca(X_calib)

    # project full-day returns onto eigenvectors
    X_all = day_ret.values
    scores_all = X_all @ Vt.T   # (n_obs, 2)

    # z-score from post-calibration onward using expanding mean/std
    post_idx = np.where(post)[0]
    if len(post_idx) == 0:
        return None

    scores_post = scores_all[post_idx, :]
    timestamps_post = day_ret.index[post_idx]

    # expanding z-score: for each t, use all post-calib obs up to and including t
    z = np.full_like(scores_post, np.nan)
    for i in range(1, len(scores_post)):  # need ≥2 obs for std
        window = scores_post[: i + 1]
        mu = window.mean(axis=0)
        sigma = window.std(axis=0, ddof=1)
        sigma = np.where(sigma == 0, np.nan, sigma)
        z[i] = (scores_post[i] - mu) / sigma

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": Vt,
        "scores_post": pd.DataFrame(scores_post, index=timestamps_post, columns=["f1", "f2"]),
        "z_scores": pd.DataFrame(z, index=timestamps_post, columns=["z1", "z2"]),
        "n_calib": int(X_calib.shape[0]),
    }


def main():
    ret1 = pd.read_parquet(PROCESSED_DIR / "returns_1min.parquet")

    daily_results = {}
    for date, day_df in ret1.groupby(ret1.index.normalize()):
        result = process_day(day_df)
        if result is None:
            print(f"  {date.date()}: insufficient data, skipped")
            continue
        daily_results[date] = result
        ev = result["eigenvalues"]
        nc = result["n_calib"]
        print(f"  {date.date()}: calib obs={nc}, eigenvalues=[{ev[0]:.6f}, {ev[1]:.6f}], sum={ev.sum():.6f}")

    # eigenvalue evolution across days
    ev_records = [
        {"date": d, "ev1": r["eigenvalues"][0], "ev2": r["eigenvalues"][1]}
        for d, r in daily_results.items()
    ]
    ev_df = pd.DataFrame(ev_records).set_index("date") if ev_records else pd.DataFrame()

    out = {"daily": daily_results, "eigenvalue_evolution": ev_df}
    path = PROCESSED_DIR / "pca_results.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved {len(daily_results)} day(s) → {path}")


if __name__ == "__main__":
    main()
