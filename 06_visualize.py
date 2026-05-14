"""All figures for the MAR/HLT correlation & cointegration analysis."""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import statsmodels.graphics.tsaplots as tsaplots
from config import TICKERS, PROCESSED_DIR, FIGURES_DIR

matplotlib.use("Agg")

MAR, HLT = TICKERS

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
C1, C2 = OKABE_ITO[4], OKABE_ITO[5]   # blue, vermillion

RC = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}
plt.rcParams.update(RC)


def save(fig: plt.Figure, name: str):
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ---------------------------------------------------------------------------
# A — Static regression scatter
# ---------------------------------------------------------------------------

def plot_static_scatter(regression_results: dict):
    for (dep, indep, label, kind), res in regression_results.items():
        if kind != "static":
            continue
        ret = pd.read_parquet(PROCESSED_DIR / f"returns_{label}.parquet" if label == "1min"
                              else PROCESSED_DIR / "returns_20min_overlap5.parquet")
        x = ret[indep].dropna()
        y = ret[dep].dropna()
        common = x.index.intersection(y.index)
        x, y = x.loc[common], y.loc[common]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(x, y, alpha=0.3, s=6, color=C1, linewidths=0)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, res["alpha"] + res["beta"] * x_line, color=C2, lw=1.5,
                label=f"β={res['beta']:.4f}  SE={res['se_beta']:.4f}  R²={res['r2']:.4f}")
        ax.set_xlabel(f"{indep} return ({label})")
        ax.set_ylabel(f"{dep} return ({label})")
        ax.set_title(f"Static OLS: {dep} ~ {indep}  [{label}]")
        ax.legend(loc="upper left")
        save(fig, f"A_static_{dep}_vs_{indep}_{label}")


# ---------------------------------------------------------------------------
# B — Rolling regression
# ---------------------------------------------------------------------------

def plot_rolling(regression_results: dict):
    # B1: one-snapshot scatter (median-R² window)
    for (dep, indep, label, kind), rolls in regression_results.items():
        if kind != "rolling" or not rolls:
            continue
        r2s = [r["r2"] for r in rolls]
        med_idx = int(np.argsort(r2s)[len(r2s) // 2])
        res = rolls[med_idx]

        ret_file = "returns_1min.parquet" if label == "1min" else "returns_20min_overlap5.parquet"
        ret = pd.read_parquet(PROCESSED_DIR / ret_file)
        chunk = ret.loc[res["t_start"]:res["t_end"]]
        x = chunk[indep].dropna()
        y = chunk[dep].dropna()
        common = x.index.intersection(y.index)
        x, y = x.loc[common], y.loc[common]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(x, y, alpha=0.4, s=8, color=C1, linewidths=0)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, res["alpha"] + res["beta"] * x_line, color=C2, lw=1.5,
                label=f"β={res['beta']:.4f}  R²={res['r2']:.4f}")
        ax.set_xlabel(f"{indep} return ({label})")
        ax.set_ylabel(f"{dep} return ({label})")
        ax.set_title(f"Rolling snapshot (median R²): {dep}~{indep} [{label}]")
        ax.legend(loc="upper left")
        save(fig, f"B1_rolling_snapshot_{dep}_vs_{indep}_{label}")

    # B2: beta evolution (4-panel)
    combo_keys = [
        (MAR, HLT, "1min"),
        (HLT, MAR, "1min"),
        (MAR, HLT, "20min"),
        (HLT, MAR, "20min"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    axes = axes.flatten()
    for ax, (dep, indep, label) in zip(axes, combo_keys):
        rolls = regression_results.get((dep, indep, label, "rolling"), [])
        if not rolls:
            ax.set_visible(False)
            continue
        t_end = [pd.Timestamp(r["t_end"]) for r in rolls]
        betas = [r["beta"] for r in rolls]
        se = [r["se_beta"] for r in rolls]
        betas = np.array(betas)
        se = np.array(se)
        ax.plot(t_end, betas, color=C1, lw=1.2)
        ax.fill_between(t_end, betas - 2 * se, betas + 2 * se, alpha=0.25, color=C1)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_title(f"β: {dep}~{indep} [{label}]")
        ax.set_ylabel("β")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Rolling OLS β evolution (±2 SE)", y=1.01)
    fig.tight_layout()
    save(fig, "B2_rolling_beta_evolution")


# ---------------------------------------------------------------------------
# C — ACF
# ---------------------------------------------------------------------------

def plot_acf(ret1: pd.DataFrame, ret20: pd.DataFrame):
    for ticker in TICKERS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        tsaplots.plot_acf(ret1[ticker].dropna(), lags=40, ax=ax1, color=C1, vlines_kwargs={"colors": C1})
        ax1.set_title(f"{ticker} ACF — 1-min returns")
        tsaplots.plot_acf(ret20[ticker].dropna(), lags=20, ax=ax2, color=C2, vlines_kwargs={"colors": C2})
        ax2.set_title(f"{ticker} ACF — 20-min overlapping returns")
        fig.suptitle(f"Autocorrelation — {ticker}")
        fig.tight_layout()
        save(fig, f"C_acf_{ticker}")


# ---------------------------------------------------------------------------
# D — PCA (per trading day)
# ---------------------------------------------------------------------------

def confidence_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def plot_pca(pca_results: dict):
    ret1 = pd.read_parquet(PROCESSED_DIR / "returns_1min.parquet")
    ev_df: pd.DataFrame = pca_results["eigenvalue_evolution"]
    daily: dict = pca_results["daily"]

    for date, day_res in sorted(daily.items()):
        day_ret = ret1.loc[ret1.index.normalize() == date, [MAR, HLT]].dropna()
        timestamps = day_ret.index
        t_num = (timestamps.hour * 60 + timestamps.minute).values  # minutes since midnight

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: scatter with ellipse
        ax = axes[0]
        sc = ax.scatter(day_ret[MAR], day_ret[HLT], c=t_num, cmap="viridis", s=8, alpha=0.7, linewidths=0)
        plt.colorbar(sc, ax=ax, label="Minute of day")
        Vt = day_res["eigenvectors"]
        ev = day_res["eigenvalues"]
        cov = (Vt.T * ev) @ Vt
        mean = day_ret[[MAR, HLT]].mean().values
        confidence_ellipse(mean, cov, ax, n_std=2.0,
                           edgecolor=C2, facecolor="none", lw=1.5, linestyle="--")
        ax.set_xlabel(f"{MAR} 1-min return")
        ax.set_ylabel(f"{HLT} 1-min return")
        ax.set_title(f"Return scatter + 2σ ellipse\n{date.date()}")

        # Panel 2: z-score time series
        ax = axes[1]
        z = day_res["z_scores"]
        ax.plot(z.index, z["z1"], color=C1, lw=1, label="Factor 1")
        ax.plot(z.index, z["z2"], color=C2, lw=1, label="Factor 2")
        for level in (-2, 2):
            ax.axhline(level, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Time")
        ax.set_ylabel("Z-score")
        ax.set_title(f"PCA factor z-scores (post-12:30)\n{date.date()}")
        ax.legend(loc="upper right")
        ax.tick_params(axis="x", rotation=30)

        # Panel 3: eigenvalue evolution across all days
        ax = axes[2]
        if not ev_df.empty:
            ax.plot(ev_df.index, ev_df["ev1"], color=C1, lw=1.5, marker="o", ms=5, label="EV 1")
            ax.plot(ev_df.index, ev_df["ev2"], color=C2, lw=1.5, marker="s", ms=5, label="EV 2")
            ax.axvline(date, color="black", lw=0.8, ls=":", alpha=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Eigenvalue evolution — all days")
        ax.legend(loc="upper right")
        ax.tick_params(axis="x", rotation=30)

        fig.suptitle(f"PCA Analysis — {date.date()}", fontsize=12)
        fig.tight_layout()
        save(fig, f"D_pca_{date.strftime('%Y%m%d')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ret1 = pd.read_parquet(PROCESSED_DIR / "returns_1min.parquet")
    ret20 = pd.read_parquet(PROCESSED_DIR / "returns_20min_overlap5.parquet")

    with open(PROCESSED_DIR / "regression_results.pkl", "rb") as f:
        reg = pickle.load(f)

    with open(PROCESSED_DIR / "pca_results.pkl", "rb") as f:
        pca = pickle.load(f)

    print("A — Static scatter plots:")
    plot_static_scatter(reg)

    print("B — Rolling regression plots:")
    plot_rolling(reg)

    print("C — ACF plots:")
    plot_acf(ret1, ret20)

    print("D — PCA plots:")
    plot_pca(pca)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
