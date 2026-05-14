"""Compile all results into a single PDF report."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pickle
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from config import TICKERS, PROCESSED_DIR, FIGURES_DIR, START_DATE, END_DATE

MAR, HLT = TICKERS
OUT_PDF = Path(__file__).parent / "output" / "report.pdf"

RC = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid": False,
    "font.family": "sans-serif",
    "font.size": 10,
}
plt.rcParams.update(RC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def blank_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def title_page(pdf: PdfPages):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    blank_ax(ax)
    ax.set_facecolor("#f7f7f7")

    ax.text(0.5, 0.72, "MAR / HLT", ha="center", va="center",
            fontsize=32, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.63, "Intraday Correlation & Cointegration Analysis",
            ha="center", va="center", fontsize=18, transform=ax.transAxes)
    ax.axhline(y=0.59, xmin=0.1, xmax=0.9, color="#333333", linewidth=0.8)
    lines = [
        f"Tickers:      {MAR} (Marriott International)  &  {HLT} (Hilton Worldwide)",
        f"Date range:   {START_DATE} – {END_DATE}  (excl. Good Friday Apr 3)",
        "Data source:  WRDS TAQ millisecond quotes  (taqmsec.cqm_YYYYMMDD)",
        "Returns:      1-min non-overlapping  |  20-min overlapping (step = 5 min)",
        "Methods:      OLS (HC3 SE)  ·  Engle-Granger  ·  Johansen  ·  PCA/SVD",
    ]
    for i, line in enumerate(lines):
        ax.text(0.5, 0.52 - i * 0.055, line, ha="center", va="center",
                fontsize=10.5, fontfamily="monospace", transform=ax.transAxes)

    ax.text(0.5, 0.08, f"Generated {pd.Timestamp.now().strftime('%Y-%m-%d')}",
            ha="center", va="center", fontsize=9, color="#888888",
            transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def section_heading(pdf: PdfPages, number: str, title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    blank_ax(ax)
    ax.text(0.08, 0.88, f"Section {number}", ha="left", va="center",
            fontsize=11, color="#888888", transform=ax.transAxes)
    ax.text(0.08, 0.82, title, ha="left", va="center",
            fontsize=22, fontweight="bold", transform=ax.transAxes)
    ax.axhline(y=0.79, xmin=0.08, xmax=0.92, color="#333333", linewidth=0.8)
    if subtitle:
        for i, line in enumerate(textwrap.wrap(subtitle, 90)):
            ax.text(0.08, 0.74 - i * 0.04, line, ha="left", va="center",
                    fontsize=10.5, color="#333333", transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def embed_figure(pdf: PdfPages, name: str, caption: str = ""):
    path = FIGURES_DIR / f"{name}.png"
    if not path.exists():
        return
    img = mpimg.imread(str(path))
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.imshow(img)
    blank_ax(ax)
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", fontsize=8.5, color="#555555",
                 style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def results_table(pdf: PdfPages, title: str, col_labels: list, row_labels: list,
                  cell_data: list, col_widths: list | None = None):
    n_rows = len(row_labels)
    fig_h = max(3.5, 1.2 + n_rows * 0.45)
    fig = plt.figure(figsize=(8.5, fig_h))
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    blank_ax(ax)
    ax.text(0.0, 1.0, title, ha="left", va="bottom",
            fontsize=12, fontweight="bold", transform=ax.transAxes)

    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    if col_widths:
        for (row, col), cell in tbl.get_celld().items():
            if col >= 0 and col < len(col_widths):
                cell.set_width(col_widths[col])
    # style header row
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#d0d8e8")
        tbl[0, col].set_text_props(fontweight="bold")
    # alternating row shading
    for row in range(1, n_rows + 1):
        color = "#f5f5f5" if row % 2 == 0 else "white"
        for col in range(-1, len(col_labels)):
            tbl[row, col].set_facecolor(color)
    tbl.scale(1, 1.4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 1 — Static correlations
# ---------------------------------------------------------------------------

def section_static(pdf: PdfPages, reg: dict):
    section_heading(pdf, "1", "Static Correlations",
                    "Full-sample OLS regressions with HC3 heteroskedasticity-robust "
                    "standard errors, run in both directions for 1-min and 20-min "
                    "overlapping (step=5) returns.")

    col_labels = ["Direction", "Return", "α", "β", "SE(β)", "R²", "N"]
    rows, cells = [], []
    for (dep, indep, label, kind), res in sorted(reg.items()):
        if kind != "static":
            continue
        rows.append(f"{dep} ~ {indep}")
        cells.append([
            f"{dep} ~ {indep}",
            label,
            f"{res['alpha']:.5f}",
            f"{res['beta']:.5f}",
            f"{res['se_beta']:.5f}",
            f"{res['r2']:.4f}",
            f"{res['nobs']:,}",
        ])
    results_table(pdf, "Table 1 — Static OLS Results", col_labels, rows, cells)

    for dep, indep in [(MAR, HLT), (HLT, MAR)]:
        for label in ("1min", "20min"):
            embed_figure(pdf, f"A_static_{dep}_vs_{indep}_{label}",
                         f"Figure: {dep} ~ {indep}, {label} returns — scatter + OLS fit")


# ---------------------------------------------------------------------------
# Section 2 — Rolling correlations
# ---------------------------------------------------------------------------

def section_rolling(pdf: PdfPages, reg: dict):
    section_heading(pdf, "2", "Rolling Correlations",
                    "OLS regressions on a rolling 1-trading-day window, stepped by "
                    "~60 min (1-min returns) or ~12 min (20-min returns). "
                    "26 windows per direction per return type.")

    col_labels = ["Direction", "Return", "β mean", "β min", "β max", "SE mean", "R² mean"]
    rows, cells = [], []
    for (dep, indep, label, kind), rolls in sorted(reg.items()):
        if kind != "rolling" or not rolls:
            continue
        betas = np.array([r["beta"] for r in rolls])
        ses   = np.array([r["se_beta"] for r in rolls])
        r2s   = np.array([r["r2"] for r in rolls])
        rows.append(f"{dep}~{indep} [{label}]")
        cells.append([
            f"{dep} ~ {indep}",
            label,
            f"{betas.mean():.4f}",
            f"{betas.min():.4f}",
            f"{betas.max():.4f}",
            f"{ses.mean():.4f}",
            f"{r2s.mean():.4f}",
        ])
    results_table(pdf, "Table 2 — Rolling OLS Summary", col_labels, rows, cells)

    embed_figure(pdf, "B2_rolling_beta_evolution",
                 "Figure: Rolling β evolution with ±2 SE bands across all windows")

    for dep, indep in [(MAR, HLT), (HLT, MAR)]:
        for label in ("1min", "20min"):
            embed_figure(pdf, f"B1_rolling_snapshot_{dep}_vs_{indep}_{label}",
                         f"Figure: Rolling snapshot (median-R² window) — {dep}~{indep} [{label}]")


# ---------------------------------------------------------------------------
# Section 3 — Autocorrelation
# ---------------------------------------------------------------------------

def section_acf(pdf: PdfPages):
    section_heading(pdf, "3", "Autocorrelation",
                    "Ljung-Box ACF plots for each ticker. Left panel: 1-min "
                    "non-overlapping returns (40 lags). Right panel: 20-min "
                    "overlapping returns, step=5 (20 lags). Bartlett confidence bands shown.")
    for ticker in TICKERS:
        embed_figure(pdf, f"C_acf_{ticker}",
                     f"Figure: ACF — {ticker} (left: 1-min, right: 20-min overlapping)")


# ---------------------------------------------------------------------------
# Section 4 — Cointegration
# ---------------------------------------------------------------------------

def section_coint(pdf: PdfPages, coint_res: dict):
    section_heading(pdf, "4", "Cointegration",
                    "Engle-Granger (EG) and Johansen trace tests on log-price series. "
                    "Full-span results use all 5 trading days. Daily results show "
                    "within-day intraday cointegration.")

    full = coint_res["full_span"]
    col_labels = ["Interval", "EG stat", "EG p-val", "Johansen trace r=0", "Johansen trace r≤1"]
    rows, cells = [], []
    for label in ("1min", "20min"):
        eg = full[f"eg_{label}"]
        jo = full[f"johansen_{label}"]
        rows.append(label)
        cells.append([
            label,
            f"{eg['stat']:.4f}",
            f"{eg['pval']:.4f}",
            f"{jo['trace_stat'][0]:.3f}",
            f"{jo['trace_stat'][1]:.3f}",
        ])
    results_table(pdf, "Table 3 — Full-Span Cointegration", col_labels, rows, cells)

    daily: pd.DataFrame = coint_res["daily"]
    col_labels2 = ["Date", "EG 1-min stat", "EG 1-min p", "EG 20-min stat",
                   "EG 20-min p", "Johansen 1-min", "Johansen 20-min"]
    rows2, cells2 = [], []
    for date, row in daily.iterrows():
        d = str(date.date())
        rows2.append(d)
        cells2.append([
            d,
            f"{row['eg_1min_stat']:.3f}" if pd.notna(row['eg_1min_stat']) else "—",
            f"{row['eg_1min_pval']:.4f}" if pd.notna(row['eg_1min_pval']) else "—",
            f"{row['eg_20min_stat']:.3f}" if pd.notna(row['eg_20min_stat']) else "—",
            f"{row['eg_20min_pval']:.4f}" if pd.notna(row['eg_20min_pval']) else "—",
            f"{row['johansen_1min_trace']:.3f}" if pd.notna(row['johansen_1min_trace']) else "—",
            f"{row['johansen_20min_trace']:.3f}" if pd.notna(row['johansen_20min_trace']) else "—",
        ])
    results_table(pdf, "Table 4 — Daily Cointegration", col_labels2, rows2, cells2)


# ---------------------------------------------------------------------------
# Section 5 — PCA
# ---------------------------------------------------------------------------

def section_pca(pdf: PdfPages, pca_res: dict):
    section_heading(pdf, "5", "Principal Component Analysis",
                    "Per-day SVD calibrated on the first 3 hours of trading (≈179 obs). "
                    "Two eigenportfolios extracted. Factor z-scores computed with an "
                    "expanding window from 12:30 onward.")

    ev_df: pd.DataFrame = pca_res["eigenvalue_evolution"]
    if not ev_df.empty:
        col_labels = ["Date", "EV 1", "EV 2", "EV 1 share (%)"]
        rows, cells = [], []
        for date, row in ev_df.iterrows():
            d = str(date.date())
            share = 100 * row["ev1"] / (row["ev1"] + row["ev2"])
            rows.append(d)
            cells.append([d, f"{row['ev1']:.6f}", f"{row['ev2']:.6f}", f"{share:.1f}%"])
        results_table(pdf, "Table 5 — PCA Eigenvalues by Day", col_labels, rows, cells)

    daily = pca_res["daily"]
    for date in sorted(daily.keys()):
        embed_figure(pdf, f"D_pca_{date.strftime('%Y%m%d')}",
                     f"Figure: PCA — {date.date()}  "
                     f"(left: scatter+ellipse, centre: z-scores, right: EV evolution)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(PROCESSED_DIR / "regression_results.pkl", "rb") as f:
        reg = pickle.load(f)
    with open(PROCESSED_DIR / "cointegration_results.pkl", "rb") as f:
        coint_res = pickle.load(f)
    with open(PROCESSED_DIR / "pca_results.pkl", "rb") as f:
        pca_res = pickle.load(f)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(OUT_PDF)) as pdf:
        title_page(pdf)
        section_static(pdf, reg)
        section_rolling(pdf, reg)
        section_acf(pdf)
        section_coint(pdf, coint_res)
        section_pca(pdf, pca_res)

        d = pdf.infodict()
        d["Title"] = "MAR/HLT Intraday Correlation & Cointegration Analysis"
        d["Author"] = "corrcoint"
        d["Subject"] = "TAQ intraday returns, OLS, cointegration, PCA"

    print(f"Report saved -> {OUT_PDF}")


if __name__ == "__main__":
    main()
